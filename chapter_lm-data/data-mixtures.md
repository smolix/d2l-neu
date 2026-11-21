# Data Mixtures
:label:`sec_lm_data_mixtures`

<!-- For maintainers. Plan: docs/llm/language_models_part_plan.md, "3.4 Data
Mixtures"; blueprint docs/llm/blueprints/ch03_data.md, section 3.4. Rights:
the code pool (CodeParrot files under per-file licences) is built locally and
never hosted; only its meta and per-file attribution are registered. Every
prose sentence that quotes a T1 number says "at T1 scale" or names T1, and
its paragraph carries the mark "campaign: lm-mix-runs-t2-v1", so the step
that registers the T2 run set finds and rewrites it. -->

The previous three sections decided which documents to keep. A model is not
trained on a set of documents, though, but on a stream of them, and someone
has to decide how often each kind of text appears in that stream. Suppose we
want the model of the next chapter to write prose, Python and the occasional
proof. How much of its training budget should go to web pages, how much to
code and how much to mathematics? Released models state their answers to the
percent. By its dataset card, the pretraining mixture of Olmo 3 is 76% web
text, 14% scientific PDFs, 7% code and 2.6% mathematical web pages
:cite:`OLMo3.2025`. Nobody derives such numbers. They are hyperparameters,
like a learning rate, with one difference that makes them expensive: every
candidate needs a pretraining run of its own.

The way out is the one :numref:`sec_scaling-laws` took for model size and
data: study small models and hope that they predict large ones. RegMix (Liu
and colleagues, 2024) and the data mixing laws of Ye and colleagues (2024)
make the hope into a method. Train many small *proxy* models on randomly
chosen mixtures, fit a function from the mixture weights to the loss on each
domain, and minimize that function. Two worries come with the method. A
mixture that looks best after a short run might be overtaken later, its loss
curve crossing that of another; and a mixture that suits a model of five
million parameters might not suit one a hundred times larger. This section
builds the method for three domains, web text, Python and mathematics. It
measures the noise that any fitted function has to beat, compares the mixture
it selects with uniform weights and with the weights of DoReMi, a method that
optimizes a different objective, and puts the first worry to 41 runs and the
second to a model four times larger. Temperature sampling across languages,
the value of repeated data and annealing at the end of training come along
the way, because each changes how a mixture should be read.

```{.python .input #data-mixtures-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #data-mixtures-extras}
%%tab pytorch
import collections
import gzip
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import time
import pandas as pd
import yaml
from scipy import optimize, spatial, stats
```

Everything here is computed for three domains, `web`, `code` and `math`, and
judged against two target distributions over them that are fixed in
`MIX_TARGETS` before any run. `t_avg` weighs the domains equally. `t_dep`,
$(0.7, 0.15, 0.15)$, stands for a deployment that reads mostly web text with
some code and mathematics; it is an illustration, not a measurement of any
real deployment, and it is fixed in advance because a target chosen after the
results can make almost any mixture look best. All training runs of the
section are made by one launcher, `tools/run_mix_campaign.py`, and stored in
a registered run set. A normal build reads that file and needs a GPU only for
a short live check. Setting `D2L_RERUN=mix-t1` retrains all 41 runs on the
GPUs named in `CUDA_VISIBLE_DEVICES`, which takes about a quarter of an hour
on four RTX 4090s, and writes the result under a new name.

```{.python .input #data-mixtures-config}
%%tab pytorch
#@save
MIX_DOMAINS = ('web', 'code', 'math')

#@save
MIX_TARGETS = {  # target weights over MIX_DOMAINS, fixed before any run
    't_avg': (1 / 3, 1 / 3, 1 / 3), 't_dep': (0.7, 0.15, 0.15)}

def say(*args, sep=' '):
    """print, each line wrapped to the page's 100 columns."""
    print('\n'.join(l if len(l) <= 100 else textwrap.fill(
        l, 100, subsequent_indent='    ', break_long_words=False,
        break_on_hyphens=False) for l in sep.join(map(str, args)).split('\n')))


DOMAINS = MIX_DOMAINS
T_AVG, T_DEP = (np.array(MIX_TARGETS[k]) for k in ('t_avg', 't_dep'))
ARTIFACTS = {'mix-t1': ('lm-mix-runs-t1-v1', 'lm-mix-runs-t1-v1.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
GPU = torch.cuda.is_available()
assert GPU or not RERUN, 'the mix-t1 regeneration trains on a GPU'
# The T2 campaign's logs replace the T1 run set once they are registered
T2 = 'lm-mix-runs-t2-v1' in d2l.DATA_HUB and not RERUN
RUNS_KEY = 'lm-mix-runs-t2-v1' if T2 else 'lm-mix-runs-t1-v1'
CONFIG = 't2' if T2 else 't1'
TAG = f'[at {CONFIG.upper()} scale]'
say(f'configuration {CONFIG}' + ('' if T2 else ' (the T2 campaign logs are '
    'not yet committed)'), '| regenerating:', ', '.join(sorted(RERUN))
    or 'none')
```

```{.python .input #data-mixtures-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-mix-web-20m-v1'] = (  # 20M reference-tokenizer tokens of
    # FineWeb-Edu's side buckets 1-7 (20.4); ODC-By
    d2l.LM_DATA_URL + 'lm-mix-web-20m-v1.tar',
    '02e073e66edbf4a7a83fa0936e4222a1ce3eab2f')
d2l.DATA_HUB['lm-mix-math-20m-v1'] = (  # 20M tokens of FineMath 4+ (20.4);
    # ODC-By
    d2l.LM_DATA_URL + 'lm-mix-math-20m-v1.tar',
    'aa1d5bf776548da1610d06148d339b33a57eb026')
d2l.DATA_HUB['lm-mix-val-1m-v1'] = (  # the web and mathematics validation
    # sets, 1M tokens each (20.4); ODC-By
    d2l.LM_DATA_URL + 'lm-mix-val-1m-v1.tar',
    '3b3aeb6a0b35c756b15d7fe05ea6a22ded3efd74')
d2l.DATA_HUB['lm-mix-code-meta-v1'] = (  # the code pool's meta and per-file
    # attribution, no code: CodeParrot's files are built locally (20.4)
    d2l.LM_DATA_URL + 'lm-mix-code-meta-v1.tar',
    '15ad995f32fb085771f2be54c5fd65b392c974c7')
d2l.DATA_HUB['lm-mix-web-150m-v1'] = (  # the T2 pools: 150M tokens of
    # FineWeb-Edu's side buckets 1-7 (20.4); ODC-By
    d2l.LM_DATA_URL + 'lm-mix-web-150m-v1.tar',
    'e38535606a46743f4d255d90cfab169f6f186297')
d2l.DATA_HUB['lm-mix-math-150m-v1'] = (  # 150M tokens of FineMath 4+
    # (20.4); ODC-By
    d2l.LM_DATA_URL + 'lm-mix-math-150m-v1.tar',
    '0b55675617bbd88e8cea96b99947e62a5ca76ddf')
d2l.DATA_HUB['lm-mix-val-2m-v1'] = (  # the T2 web and mathematics
    # validation sets, 2M tokens each (20.4); ODC-By
    d2l.LM_DATA_URL + 'lm-mix-val-2m-v1.tar',
    'dc2deae77f8ebc37a599d04c46b97ae229961edb')
d2l.DATA_HUB['lm-mix-code150-meta-v1'] = (  # the T2 code pool's meta and
    # attribution, no code (20.4)
    d2l.LM_DATA_URL + 'lm-mix-code150-meta-v1.tar',
    '139d6b2a3d463867cab48cd0b64986d7c0466996')
d2l.DATA_HUB['lm-mix-runs-t1-v1'] = (  # the 41 T1 runs, the selection,
    # DoReMi's weights and the T2 smoke (mix-t1 regeneration, 20.4)
    d2l.LM_DATA_URL + 'lm-mix-runs-t1-v1.json',
    '92abf76a5e9891ba3efe5c5daa5f2a021f05c9e4')
d2l.DATA_HUB['lm-mix-runs-t2s-v1'] = (  # a small-scale test of the T2 code
    # path on four RTX 4090s: the T2 design at half its tokens (20.4)
    d2l.LM_DATA_URL + 'lm-mix-runs-t2s-v1.json',
    'dac131e52de0e8b0d8ae883f5005eee6c3dcf60d')

# lm-mix-runs-t2-v1 is added after the T2 campaign
MIX_KEYS = ('lm-mix-web-20m-v1', 'lm-mix-math-20m-v1', 'lm-mix-val-1m-v1',
            'lm-mix-code-meta-v1', 'lm-mix-runs-t1-v1', 'lm-mix-web-150m-v1',
            'lm-mix-math-150m-v1', 'lm-mix-val-2m-v1',
            'lm-mix-code150-meta-v1', 'lm-mix-runs-t2s-v1',
            'lm-mix-runs-t2-v1')
UPSTREAM = ('lm-r2-tokenizer-v1',)
for name, keys in (('this section', MIX_KEYS), ('upstream', UPSTREAM)):
    say(f'{name}: registered', ', '.join(
        k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
        ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')
```

## Domain Weights Are Hyperparameters
:label:`subsec_lm_mix_weights`

A *mixture* over $m$ domains is a vector of weights $\mathbf{w}$ on the
simplex, $w_i \geq 0$ and $\sum_i w_i = 1$. Each training sequence is drawn
from domain $i$ with probability $w_i$; our sampler allocates the sequences so
that every domain stays within one sequence of its share at every step, which
removes the sampling noise of the shares themselves. Write $b_i(\mathbf{w})$
for the loss on held-out text of domain $i$ of a model trained on mixture
$\mathbf{w}$. We will judge mixtures by two objectives: the average
$\frac{1}{m}\sum_i b_i(\mathbf{w})$, which is the target-weighted loss
$\sum_i t_i b_i(\mathbf{w})$ with $\mathbf{t}$ set to `t_avg`, and the same
sum with $\mathbf{t}$ set to `t_dep`.

The losses must be comparable across domains, and losses per token are not.
A token of web prose stands for about 4.5 bytes of text under our tokenizer
and a token of Python for about 2.4, so a model can have a lower loss per
token on code and still predict code worse. We therefore report *bits per
byte*, as in :numref:`subsec_bits-per-byte`: the summed negative
log-likelihood in nats, divided by $\ln 2$ times the number of UTF-8 bytes
the scored tokens stand for (`bpb_from_nll`). Bits per byte do not depend on
the tokenizer, so a gain measured in them is a gain in prediction and not in
segmentation. To credit a difference in loss to the mixture, everything else
must be held fixed: the tokenizer, the model, the number of training tokens,
the learning-rate schedule and its peak, the evaluation sets and the seed
that orders the data.

The study comes in two tiers of the same code, fixed in `MIX_CONFIGS` before
any run. T1 trains proxies of 4.6 million parameters for $2^{24}$ tokens per
run and fits on one consumer GPU. T2 trains proxies of 20.3 million
parameters for $2^{27}$ tokens per run, eight times the tokens on a model four
times larger, and is packaged as a campaign for a larger machine. The page
reads the largest run set that is registered; the numbers quoted in the text
are those of T1, and each such sentence says so.

```{.python .input #data-mixtures-configs}
%%tab pytorch
#@save
ProxyCfg = collections.namedtuple('ProxyCfg', (
    'num_hiddens num_heads num_blks seq_len batch micro_batch lr '
    'warmup_frac schedule decay_frac final_lr_frac weight_decay betas dtype '
    'vocab_size'), defaults=(1024, 64, 16, 1e-3, 0.02, 'cosine', 0.2, 0.1,
                             0.1, (0.9, 0.95), 'bfloat16', 32768))

#@save
MIX_EDGES = (  # seed-free training mixtures near the corners of the simplex
    (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))

#@save
MIX_CONFIGS = {  # the study's two tiers, fixed before any run
    't1': dict(  # one RTX 4090
        name='t1', domains=MIX_DOMAINS, targets=MIX_TARGETS, target='t_dep',
        proxy=ProxyCfg(num_hiddens=128, num_heads=2, num_blks=2),
        tokens=2**24, pool='20m', val='1m', pool_tokens=20_000_000,
        val_tokens=1_000_000, keys={
            'web-20m': 'lm-mix-web-20m-v1', 'math-20m': 'lm-mix-math-20m-v1',
            'web-val-1m': 'lm-mix-val-1m-v1',
            'math-val-1m': 'lm-mix-val-1m-v1'},
        code_meta='lm-mix-code-meta-v1', runs_key='lm-mix-runs-t1-v1',
        train=((1.0, 6), (0.3, 6)), edges=MIX_EDGES, held=4,
        design_seeds=(0, 1), cos_seeds=(1, 2), arm_seeds=(0, 1, 2),
        max_weight=0.9, lr_grid=(5e-4, 1e-3, 2e-3, 4e-3), lr_tokens=2**23,
        lr_tie=1e-3, eval_at=tuple(k / 8 for k in range(1, 9)),
        arm_eval_at=tuple(k / 20 for k in range(1, 21)), curve_windows=256,
        log_every=10, ckpt_every=256, stage_frac=0.8, select='mixlaw',
        compile=False, doremi=dict(eta=1.0, smooth=1e-3)),
    't2': dict(  # the campaign package: run on a faster node
        name='t2', domains=MIX_DOMAINS, targets=MIX_TARGETS, target='t_dep',
        proxy=ProxyCfg(num_hiddens=320, num_heads=5, num_blks=8),
        tokens=2**27, pool='150m', val='2m', pool_tokens=150_000_000,
        val_tokens=2_000_000, keys={
            'web-150m': 'lm-mix-web-150m-v1',
            'math-150m': 'lm-mix-math-150m-v1',
            'web-val-2m': 'lm-mix-val-2m-v1',
            'math-val-2m': 'lm-mix-val-2m-v1'},
        code_meta='lm-mix-code150-meta-v1', runs_key='lm-mix-runs-t2-v1',
        train=((1.0, 8), (0.3, 8)), edges=MIX_EDGES, held=8,
        design_seeds=(0, 1), cos_seeds=(1, 2), arm_seeds=(0, 1, 2),
        max_weight=0.9, lr_grid=(5e-4, 1e-3, 2e-3, 4e-3), lr_tokens=2**25,
        lr_tie=1e-3, eval_at=tuple(k / 8 for k in range(1, 9)),
        arm_eval_at=tuple(k / 20 for k in range(1, 21)), curve_windows=256,
        log_every=10, ckpt_every=256, stage_frac=0.8, select='mixlaw',
        compile=True, doremi=dict(eta=1.0, smooth=1e-3))}
```

The web domain is FineWeb-Edu, the corpus that
:numref:`sec_lm_quality_filtering` explained how to build. The chapter splits
it into 256 buckets by a hash of each document's id, so the bucket of a
document depends on nothing but its id, and every section that uses the
corpus agrees on which document lies where.

```{.python .input #data-mixtures-bucket-def}
%%tab pytorch
#@save
def fwedu_bucket(doc_id):
    """The bucket (0-255) of a FineWeb-Edu document: the first 32 bits of
    the SHA-256 of its id string (as released, <urn:uuid:...>) mod 256."""
    return int(hashlib.sha256(doc_id.encode('utf-8')).hexdigest()[:8],
               16) % 256

#@save
FWEDU_BUCKETS = {  # bucket ranges of the chapter's FineWeb-Edu splits
    'val': range(0, 1),         # the reference model's validation set
    'side': range(1, 8),        # side pool: 20.4's and 20.5's web training text
    'side_val': range(8, 9),    # side validation
    'train': range(9, 256)}     # the reference model's training shards
```

The three pools follow. Web text comes from buckets 1 to 7, with bucket 8 as
its validation set. Code is Python from CodeParrot's cleaned GitHub files,
kept only where a file's own licence is permissive (MIT, Apache 2.0, BSD or
ISC); these files carry individual licences, so the book hosts only their
metadata and per-file attribution, and `tools/build_mix_pools.py` builds the
pool on the reader's machine. Mathematics is FineMath at its highest quality
level (4 and above), web pages about mathematics selected by a classifier
much like that of :numref:`sec_lm_quality_filtering`. Every pool is
tokenized with the reference tokenizer that :numref:`sec_lm_tokenizers`
trains, and every pool holds at least one run's full budget of tokens, so
even a run on a single domain reads each token once. Code is split into pool
and validation by repository and path, FineWeb-Edu by document id, so forks
and near-duplicates can land on both sides of the split; CodeParrot is
deduplicated exactly but not fuzzily. At the model sizes of this section and
a single epoch the effect is too small to matter, but a production pipeline
would split after the near-duplicate stage of :numref:`sec_lm_dedup`.

```{.python .input #data-mixtures-pools}
%%tab pytorch
C = d2l.MIX_CONFIGS[CONFIG]
POOLS, VALS = d2l.mix_pools(C)
for c in d2l.MIX_CONFIGS.values():  # a pure-domain run reads all its tokens
    assert c['pool_tokens'] >= c['tokens']
meta = {d: (POOLS[d].meta, VALS[d].meta) for d in POOLS if d in VALS}
if 'code' not in meta:  # CodeParrot's files are not hosted (their licences)
    say('code pool not built here: python tools/build_mix_pools.py '
        f'--config {CONFIG}; its meta comes from {C["code_meta"]}')
    tar = d2l.download_keyed(C['code_meta'])
    root = pathlib.Path(tar).with_suffix('')
    if not (root / 'meta.json').exists():
        root.mkdir(exist_ok=True)
        d2l.extract(tar, str(root))
    with open(root / 'meta.json') as f:
        m = json.load(f)
    meta['code'] = (m[f'code-{C["pool"]}.bin'], m[f'code-val-{C["val"]}.bin'])
POOL_TABLE = pd.DataFrame([dict(
    domain=d, split=split, tokens=m['n_tokens'], documents=m['n_docs'],
    bytes_per_token=m['n_bytes_scored'] / m['n_tokens'],
    tokenizer=m['tokenizer_sha256'][:12], here=d in POOLS)
    for d in DOMAINS for split, m in zip(('pool', 'val'), meta[d])])
say(POOL_TABLE.round(3).to_string(index=False))
assert POOL_TABLE.tokenizer.nunique() == 1, 'pools from different tokenizers'
assert (POOL_TABLE[POOL_TABLE.split == 'pool'].tokens >= C['tokens']).all()
for d, P in itertools.chain(POOLS.items(), VALS.items()):  # one EOS per doc
    eos = P.meta['eos_id']
    assert int((np.asarray(P.ids) == eos).sum()) == P.meta['n_docs'], d
for P, side in ((POOLS['web'], 'side'), (VALS['web'], 'side_val')):
    docs = pathlib.Path(P.ids.filename).with_suffix('.docs.jsonl.gz')
    with gzip.open(docs, 'rt') as f:
        ids = [json.loads(row)['doc_id'] for row in itertools.islice(f, 1000)]
    assert all(d2l.fwedu_bucket(i) in d2l.FWEDU_BUCKETS[side] for i in ids)
say('bucket rule: the first 1,000 web documents of the pool lie in '
    'buckets 1-7, those of the validation set (up to 1,000) in bucket 8')
```

Bytes per token differ by almost a factor of two between web text and code,
which is why the losses below are always per byte. The pools also differ in
documents: 20 million tokens are about 19,000 web pages, 12,000 mathematics
pages or 4,800 Python files, so the same weight buys very different numbers
of distinct documents in each domain.

### Temperature Sampling and Repetition
:label:`subsec_lm_mix_temperature`

Before training anything, it is worth asking what a mixture costs in data.
When the sources differ greatly in size, as languages do, sampling in
proportion to size starves the small ones, and sampling uniformly reads the
small ones over and over. *Temperature sampling* interpolates between the
two. Source $i$ with $n_i$ documents (or tokens) is sampled with probability

$$
p_i = \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}},
\qquad
e_i = \frac{p_i D}{n_i} = \frac{D\, n_i^{\alpha - 1}}{\sum_j n_j^{\alpha}},
$$
:eqlabel:`eq_lm_mix_epochs`

where $D$ is the number drawn in all and $e_i$ the number of epochs of source
$i$. At $\alpha = 1$ the sampling is proportional and every source is read
equally often, $e_i = D / \sum_j n_j$. At $\alpha = 0$ it is uniform. For any
$\alpha < 1$ the factor $n_i^{\alpha - 1}$ decreases with $n_i$, so the
smallest sources are repeated most. Multilingual BERT sampled languages with
$\alpha = 0.7$, and XLM-R and mT5 with $\alpha = 0.3$.

Repetition has a price. Muennighoff and colleagues trained models on data
repeated for many epochs and found that up to about four epochs
of repetition change the loss little compared with fresh data, while beyond
that the value of each further pass decays
:cite:`Muennighoff.Rush.Barak.ea.2023`. They summarized the effect in an
*effective* number of tokens. For $U$ unique tokens read for $E$ epochs, with
$R = E - 1$ repetitions,

$$
D' = U + U R^* \bigl(1 - e^{-R/R^*}\bigr),
$$
:eqlabel:`eq_lm_mix_effective`

with a fitted constant $R^* \approx 15.4$. For small $R$ the exponential is
close to $1 - R/R^*$ and $D' \approx U(1 + R) = UE$: repeats count as fresh
tokens. For $R \gg R^*$ the value saturates at $U(1 + R^*)$, and further
epochs add nothing. The form is the integral of $dD'/dR = U e^{-R/R^*}$, an
assumption that each repeat is worth a little less than the previous one; the
paper fits it, it does not derive it. Three short functions implement these
formulas and the sampler of random mixtures that the proxy runs will need.

```{.python .input #data-mixtures-mix-defs}
%%tab pytorch
#@save
def temperature_weights(sizes, alpha):
    """Sampling probabilities p_i proportional to n_i ** alpha: alpha = 1
    samples in proportion to size, alpha = 0 uniformly. sizes is a dict or
    a sequence; the result has the same form."""
    n = np.array(list(sizes.values()) if isinstance(sizes, dict) else sizes,
                 dtype=float)
    p = np.exp(alpha * np.log(n) - (alpha * np.log(n)).max())
    p = p / p.sum()
    return dict(zip(sizes, p)) if isinstance(sizes, dict) else p

#@save
def effective_tokens(unique, epochs, r_star=15.4):
    """Effective data D' = U + U R* (1 - exp(-R / R*)) of `epochs` passes
    over `unique` tokens (R = epochs - 1 repeats): the data-constrained law
    of Muennighoff et al. with their fitted R*. Less than one epoch counts
    what was read."""
    E = np.asarray(epochs, dtype=float)
    R = np.maximum(E - 1, 0)
    return unique * np.minimum(E, 1) + unique * r_star * (1 - np.exp(
        -R / r_star))

#@save
def sample_mixtures(n, domains, concentration, seed, max_weight=0.9,
                    min_weight=0.0):
    """n mixtures over the domains, (n, m), drawn from the symmetric
    Dirichlet(concentration) and kept only when every weight lies in
    [min_weight, max_weight] (rejection), in the order drawn."""
    rng, m, out = np.random.default_rng(seed), len(domains), []
    while len(out) < n:
        w = rng.dirichlet(np.full(m, concentration))
        if min_weight <= w.min() and w.max() <= max_weight:
            out.append(w)
    return np.array(out)
```

The first check uses three toy sources of 1,000, 100 and 10 documents, where
:eqref:`eq_lm_mix_epochs` can be compared with a simulated sampler. Then we
apply the formula to ten real languages: the nine FineWeb-2 languages that
:numref:`sec_lm_tokenizers` uses, with document counts read from the dataset
on 2026-10-28, and English from FineWeb, the English corpus that FineWeb-2
extends, read on 2026-11-16. The budget is $10^{10}$ documents, about a third
of all ten together.

```{.python .input #data-mixtures-temperature}
%%tab pytorch
n_toy, D_toy = np.array([1000, 100, 10]), 100_000  # D1 on three toy sources
p = d2l.temperature_weights(n_toy, 0.3)
e = p * D_toy / n_toy
counts = np.random.default_rng(0).multinomial(D_toy, p)
se = np.sqrt(D_toy * p * (1 - p)) / n_toy
assert abs(p.sum() - 1) < 1e-12 and (np.abs(counts / n_toy - e) < 3 * se).all()
assert e[2] > e[1] > e[0]
say('D1 (alpha 0.3, sizes 1000/100/10): p =', p.round(4), '| epochs',
    e.round(1), '| simulated', (counts / n_toy).round(1))
FW_ROWS = {  # documents per language (FineWeb-2 configs; English: FineWeb)
    'eng (FineWeb)': 25_886_364_489, 'rus_Cyrl': 699_118_293,
    'cmn_Hani': 636_092_280, 'deu_Latn': 496_029_661,
    'jpn_Jpan': 400_164_452, 'arb_Arab': 62_019_605,
    'kor_Hang': 60_904_429, 'tha_Thai': 35_921_239,
    'hin_Deva': 22_151_227, 'swh_Latn': 1_218_969}
D_DOCS = 10**10  # documents drawn in all
rows = {}
for alpha in (1.0, 0.7, 0.3, 0.0):
    p = d2l.temperature_weights(FW_ROWS, alpha)
    for lang, n in FW_ROWS.items():
        rows[lang, f'p (alpha {alpha:g})'] = p[lang]
        rows[lang, f'epochs (alpha {alpha:g})'] = p[lang] * D_DOCS / n
TEMP = pd.Series(rows).unstack()[[f'{k} (alpha {a:g})' for a in (
    1.0, 0.7, 0.3, 0.0) for k in ('p', 'epochs')]].loc[list(FW_ROWS)]
for k in ('p', 'epochs'):  # sampling probability, then epochs per language
    say(TEMP[[c for c in TEMP.columns if c.startswith(k + ' ')]].to_string(
        float_format=lambda v: f'{v:.3g}'))
say('read more than four times (the repeated-data limit below):',
    {c.split()[-1][:-1]: list(TEMP.index[TEMP[c] > 4])
     for c in TEMP.columns if c.startswith('epochs')})
alphas = np.linspace(0, 1, 21)
d2l.plot(alphas, [[d2l.temperature_weights(FW_ROWS, a)[lang] for a in alphas]
                  for lang in FW_ROWS], 'alpha', 'sampling probability',
         yscale='log', figsize=(5, 3.5))
legend = d2l.plt.gca().legend(list(FW_ROWS), loc='center left',
                              bbox_to_anchor=(1, 0.5))
```

The formula and the simulation agree, and the smallest toy source is read
about 1,400 times. The real languages show the same effect at scale. In
proportion to size, English takes 92% of the samples and every language is
read a third of a time. At $\alpha = 0.3$ English falls to about a third,
Swahili, with 1.2 million documents, rises from 0.004% to 1.7% of the
samples, and is read 143 times. Five of the ten languages pass four epochs
at $\alpha = 0.3$; at $\alpha = 0.7$ only Swahili does. Temperature sampling
does not create data. It decides how often the small languages are repeated,
and the next cell says what such repetition is worth.

For the effective-data law we take $U = 10^9$ unique tokens and evaluate four
epochs, the saturation, and the point at which counting repeats as fresh
tokens overstates their value by 10%.

```{.python .input #data-mixtures-repetition}
%%tab pytorch
U = 1e9
four = float(d2l.effective_tokens(U, 4)) / (4 * U)
limit = float(d2l.effective_tokens(U, 1001)) / (U * (1 + 15.4))
E_10 = optimize.brentq(lambda E: E * U / d2l.effective_tokens(U, E) - 1.1,
                       1.01, 50)
assert abs(four - 0.932) < 0.005 and abs(limit - 1) < 1e-3
say(f"D'(4 epochs) / (4U) = {four:.3f}; D'(R = 1000) / (U (1 + R*)) = "
    f'{limit:.6f}; U E overstates D\' by 10% at E = {E_10:.2f} '
    f'(R = {E_10 - 1:.2f})')
```

Four epochs keep about 93% of their nominal value, and the simple product
$UE$ is off by 10% only at about 4.8 epochs, which is why "up to four epochs
is almost free" is a fair summary of the law. The saturation is severe,
though. Swahili read 143 times is worth $U(1 + 15.4)$, about 11% of what 143
fresh copies would be, so temperature sampling at $\alpha = 0.3$ spends most
of Swahili's share on tokens that teach almost nothing. UniMax (Chung and
colleagues, 2023) draws the consequence and caps the epochs of each language
instead of choosing a temperature. In our three domains no pool is ever read
more than once, by construction, so repetition plays no part in the
experiments below; it returns in :numref:`sec_lm_synthetic_data`, where
rephrasing a small source is a way to make unique tokens out of repeated
ones.

## Proxy Runs and Regression
:label:`subsec_lm_mix_regmix`

RegMix turns the choice of weights into a regression problem in five steps.
Draw many mixtures at random from the simplex. Train a small proxy model on
each, with everything else held fixed. Measure the loss of each proxy on each
domain. Fit a function from the weights to these losses. Minimize the fitted
target loss and train the minimizing mixture. RegMix fitted gradient-boosted
trees to 512 proxies of one million parameters, then trained a model of one
billion parameters, a thousand times larger, on the chosen mixture. Our
version is much smaller, and checks each step against the noise of the
proxies themselves.

### The Proxy Model
:label:`subsec_lm_mix_proxy`

A proxy is the book's `d2l.GPT` with the reference tokenizer's vocabulary of
32,768, trained by AdamW with a warmup and a decaying learning rate as in
:numref:`sec_scheduler`. `train_proxy_lm` takes the pools, the weights and a
token budget, and its data stream depends only on the triple (seed, weights,
tokens): the pools are cut into windows of `seq_len + 1` tokens, each domain's
windows are read in a fresh random order per epoch, and the allocation of
sequences to domains follows the weights as above. A list of `stages` switches
to other weights at given fractions of the run, which is how the annealed arm
below is trained. Losses come back as summed negative log-likelihoods and
byte counts per domain, so they convert to bits per byte with
`bpb_from_nll`, and the function refuses a budget that would read any pool
more than `max_epochs` times instead of silently repeating tokens.

```{.python .input #data-mixtures-proxy-defs}
%%tab pytorch
#@save
def bpb_from_nll(sum_nll_nats, n_bytes):
    """Bits per byte from a summed negative log-likelihood in nats and the
    number of UTF-8 bytes the scored tokens stand for."""
    return float(sum_nll_nats) / (math.log(2) * n_bytes)

#@save
def train_proxy_lm(pools, weights, tokens, cfg, seed=0, eval_sets=None,
                   stages=None, log_every=50, *, max_epochs=1,
                   stop_tokens=None, eval_at=(0.25, 0.5, 0.75, 1.0),
                   curve_windows=None, byte_lengths=None, ckpt=None,
                   ckpt_every=256, resume=None, compile=False, device=None):
    """Train a proxy d2l.GPT on `tokens` tokens of the mixture `weights`
    ({domain: weight}) of token pools, switching to the weights of each
    (fraction, weights) of `stages` at that fraction of the run. Returns
    the log [step, tokens, loss, seconds], the evaluations at the
    fractions eval_at (per domain of eval_sets, on the first curve_windows
    windows), the final evaluation on all windows, the epochs each pool was
    read, throughput, parameter count and peak memory. stop_tokens ends the
    run early on the same schedule and stream; ckpt/resume save and restore
    a checkpoint every ckpt_every steps."""
    device = device or d2l.try_gpu()
    stream = d2l._ProxyStream(pools, weights, tokens, cfg.seq_len, cfg.batch,
                              seed, stages, max_epochs)
    steps, per_step = stream.steps, cfg.batch * cfg.seq_len
    stop = steps if stop_tokens is None else min(steps,
                                                 stop_tokens // per_step)
    model = d2l._proxy_model(cfg, seed).to(device)
    opt = d2l._proxy_optimizer(model, cfg)
    if eval_sets and byte_lengths is None:
        byte_lengths = d2l.token_byte_lengths(d2l.r2_tokenizer())
    evaluate = lambda n: {d: d2l._proxy_eval(model, getattr(v, 'ids', v), cfg,
                                             byte_lengths, n)
                          for d, v in (eval_sets or {}).items()}
    key = d2l._proxy_fingerprint(weights, stages, tokens, seed, cfg)
    start, out = 0, dict(log=[], evals=[], final=None, seconds=0.0)
    if resume is not None:
        start, out = d2l._proxy_load(resume, model, opt, key)
    fwd = torch.compile(model) if compile else model
    at = {max(1, round(f * steps)): f for f in eval_at}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    seconds = seconds0 = out['seconds']
    for step in range(start, stop):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g['lr'] = cfg.lr * d2l._lr_at(step, steps, cfg)
        X = torch.from_numpy(stream.batch(step)[0]).to(device)
        loss = 0.0
        for Xm in X.split(cfg.micro_batch):  # gradient accumulation
            l = d2l._per_token_nll(fwd, Xm, cfg).mean() * len(Xm) / len(X)
            l.backward()
            loss = loss + l.detach()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ((step + 1) % log_every == 0 or step + 1 in at or step + 1 == stop
                or ckpt and (step + 1) % ckpt_every == 0):
            loss = float(loss)  # waits for the GPU, so the clock is right
            seconds += time.perf_counter() - t0
            out['log'].append([step + 1, (step + 1) * per_step, loss,
                               round(seconds, 3)])
        else:
            seconds += time.perf_counter() - t0
        if step + 1 in at and eval_sets:
            out['evals'].append(dict(frac=at[step + 1], step=step + 1,
                                     tokens=(step + 1) * per_step,
                                     domains=evaluate(curve_windows)))
        if ckpt and ((step + 1) % ckpt_every == 0 or step + 1 == stop):
            out['seconds'] = seconds
            d2l._proxy_save(ckpt, model, opt, step + 1, out, key)
    if stop == steps and eval_sets:
        out['final'] = evaluate(None)
    out.update(steps=steps, stopped_at=stop, n_params=d2l.num_params(model),
               epochs=stream.epochs, epoch_starts=stream.epoch_starts,
               seconds=seconds, throughput=(stop - start) * per_step / max(
                   seconds - seconds0, 1e-9),
               peak_mem=(torch.cuda.max_memory_allocated(device) / 2**30
                         if device.type == 'cuda' else None))
    return out
```

Before any GPU run, four properties are checked on the CPU in 32-bit floating
point with a two-block model of width 32. The stream must be a function of
(seed, weights, tokens). The allocation must keep every domain within one
sequence of its share. A model that gives every token probability $1/32{,}768$
scores exactly 15 bits per token, so its bits per byte must be 15 divided by
the bytes per token. And the parameter counts of both tiers must match the
arithmetic of :numref:`sec_scaling-laws`, $V d + 12 L d^2$ for vocabulary $V$,
width $d$ and $L$ blocks.

```{.python .input #data-mixtures-proxy-checks}
%%tab pytorch
tiny = d2l.ProxyCfg(32, 2, 2, seq_len=64, batch=8, micro_batch=4,
                    dtype='float32')
two = {d: POOLS[d] for d in ('web', 'math')}
w2 = {'web': 0.7, 'math': 0.3}
hashes = lambda s: [hashlib.sha256(s.batch(k)[0].tobytes()).hexdigest()
                    for k in range(10)]
streams = [d2l._ProxyStream(two, w2, 2**14, 64, 8, seed) for seed in (0, 0, 1)]
assert hashes(streams[0]) == hashes(streams[1]) != hashes(streams[2])
runs = [d2l.train_proxy_lm(two, w2, 20 * 8 * 64, tiny, seed=0, log_every=1,
                           device=torch.device('cpu')) for _ in range(2)]
assert abs(runs[0]['log'][19][2] - runs[1]['log'][19][2]) <= 1e-6
s = d2l._ProxyStream(two, w2, 10**4 * 64, 64, 1, seed=0)  # 10^4 sequences
web = np.cumsum(s.dom == 0)
lag = np.abs(web - 0.7 * np.arange(1, len(web) + 1)).max()
assert lag <= 1 and abs(web[-1] - 7000) <= 3 * math.sqrt(1e4 * 0.7 * 0.3)
say(f'same (seed, weights, tokens): the same 10 batches and the same loss '
    f'at step 20 ({runs[0]["log"][19][2]:.6f}); web share after 10^4 '
    f'sequences {web[-1] / 1e4:.4f}, never more than {lag:.2f} sequences '
    'from 0.7 n')
uni = d2l._proxy_model(tiny._replace(dtype='float64')).double()
nn.init.zeros_(uni.token_emb.weight)  # tied output: every logit is 0
BYTE_LEN = d2l.token_byte_lengths(d2l.r2_tokenizer())
r = d2l._proxy_eval(uni, VALS['web'].ids, tiny, BYTE_LEN, 32,
                    torch.device('cpu'))
assert abs(r['bpb'] / (15 * r['n_tokens'] / r['n_bytes']) - 1) < 1e-9
say(f'uniform model: {r["bpb"]:.6f} bits per byte = 15 / '
    f'{r["n_bytes"] / r["n_tokens"]:.4f} bytes per token')
with torch.device('meta'):
    N = {k: d2l.num_params(d2l._proxy_model(c['proxy']))
         for k, c in d2l.MIX_CONFIGS.items()}
for k, c in d2l.MIX_CONFIGS.items():
    p = c['proxy']
    arith = p.vocab_size * p.num_hiddens + 12 * p.num_blks * p.num_hiddens**2
    assert abs(N[k] / arith - 1) < 0.005
    flop = d2l.proxy_flops(p, c['tokens'], N[k])
    say(f'{k}: {N[k]:,} parameters (arithmetic {arith:,}; embedding '
        f'{p.vocab_size * p.num_hiddens:,}); {flop:.2g} FLOP per run')
TIER = (f'{CONFIG.upper()}: {N[CONFIG] / 1e6:.1f}M-parameter proxies, '
        f'2^{round(math.log2(C["tokens"]))} tokens per run')
say('this page reads', TIER)
```

All four hold. The parameter counts say something about the proxies
themselves: nine tenths of the T1 proxy, 4.2 of its 4.6 million parameters,
is the embedding table, and its two transformer blocks hold about 400,000.
The T2 proxy is half embedding. A T1 proxy is close to a model of which token
follows which, with a little context on top, and what it prefers is evidence
about mixtures only for models that learn the same things first. We will
return to this when we ask what transfers.

### The Run Design
:label:`subsec_lm_mix_design`

The design draws the training mixtures from two Dirichlet distributions: six
with concentration 1, uniform on the simplex, and six with concentration 0.3,
which puts most of its mass near the edges where one weight is small. Three
fixed mixtures near the corners, $(0.8, 0.1, 0.1)$ and its permutations, are
added. Without them no training mixture would put more than about two thirds
on web text, and `t_dep`'s region would be reached only by extrapolation; a
design sampled only from the interior of the simplex cannot say much about
its edges. Four held-out mixtures are drawn from Dirichlet(1) with another
seed and never used for fitting. The rest of the runs measure the noise and
the arms: three pure-domain runs, two more seeds of the uniform mixture on the
schedule of the training runs, and three seeds each for the four arms of
:numref:`subsec_lm_mix_choose`. No weight exceeds 0.9.

```{.python .input #data-mixtures-design}
%%tab pytorch
TRAIN_W = np.vstack([d2l.sample_mixtures(n, DOMAINS, conc,
                                         C['design_seeds'][0], C['max_weight'])
                     for conc, n in C['train']] + [np.array(C['edges'])])
HELD_W = d2l.sample_mixtures(C['held'], DOMAINS, 1.0, C['design_seeds'][1],
                             C['max_weight'])
say(TAG, f'{len(TRAIN_W)} training mixtures (Dirichlet(1), Dirichlet(0.3), '
    f'{len(C["edges"])} near the corners) and {len(HELD_W)} held-out ones '
    f'(Dirichlet(1)), weights {DOMAINS}:')
say(pd.DataFrame(np.vstack([TRAIN_W, HELD_W]), columns=DOMAINS,
                 index=[f'train {k}' for k in range(len(TRAIN_W))]
                 + [f'held {k}' for k in range(len(HELD_W))]).round(3)
    .to_string())
SPECS = d2l.mix_run_specs(C)
for arm, W in (('train', TRAIN_W), ('held', HELD_W)):  # the launcher's design
    assert np.array_equal([s['weights'] for s in SPECS if s['arm'] == arm], W)
arms = collections.Counter(s['arm'] for s in SPECS)
say(f'{len(SPECS)} runs: ' + ', '.join(f'{n} {a}' for a, n in arms.items()))
```

The two held-out draws `held 0` and `held 1` came out almost equal by chance,
both with about 5% code, so the four held-out mixtures cover three distinct
points; any rank statistic over them is a description, not a test.

One regeneration writes the whole run set. The launcher first chooses the
peak learning rate on the uniform mixture from four values at half the run's
tokens, then trains the runs in dependency order on all visible GPUs, fits
the regression and chooses the selected mixture by code, trains DoReMi's
weights, and finally the arms that need them. A normal build reads the
registered result, and the cell asserts that it is the design above, trained
on these pools with this tokenizer.

```{.python .input #data-mixtures-regen}
%%tab pytorch
ROOT = pathlib.Path(d2l.__file__).resolve().parents[1]  # the book's repository
LAUNCHER = ROOT / 'tools' / 'run_mix_campaign.py'
if 'mix-t1' in RERUN:
    out_dir = os.path.join('..', 'data', 'lm', 'rerun', 'mix-t1')
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES')  # one worker per card
    assert gpus, 'name the cards of the regeneration in CUDA_VISIBLE_DEVICES'
    t0 = time.time()
    subprocess.run([sys.executable, str(LAUNCHER), '--config', 't1', '--gpus',
                    gpus, '--out', out_dir, '--smoke-t2'], check=True)
    say(f'mix-t1: {(time.time() - t0) / 60:.1f} min wall clock')
    d2l.lm_save_rerun(ARTIFACTS['mix-t1'][1], lambda p: shutil.copy(
        os.path.join(out_dir, 'lm-mix-runs-t1-v1.json'), p))
RUNS_PATH = d2l.lm_artifact(RUNS_KEY, ARTIFACTS['mix-t1'][1]
                            if 'mix-t1' in RERUN else None)
with open(RUNS_PATH) as f:
    RUNS = json.load(f)
REC = {r['run_id']: r for r in RUNS['runs']}
# The run set is the design MIX_CONFIGS fixes, trained on these pools
assert RUNS['config'] == CONFIG and RUNS['tokens'] == C['tokens']
assert RUNS['mix_config'] == json.loads(json.dumps(C))
assert [r['run_id'] for r in RUNS['runs']] == [s['run_id'] for s in SPECS]
for s in SPECS:
    if s['weights'] is not None:
        assert REC[s['run_id']]['weights'] == s['weights'], s['run_id']
for d in DOMAINS:  # a local build must not stand in for the registered pool
    for here, ran in zip(meta[d], (RUNS['pools'][d], RUNS['val'][d])):
        assert here['sha256'] == ran['sha256'], f'{d}: not the runs\' pool'
        assert (here['tokenizer_sha256'] == ran['tokenizer_sha256']
                == d2l.R2_TOKENIZER_SHA256)
SEL, DOREMI_W = RUNS['selection'], RUNS['doremi']['weights']
assert (SEL['kind'], SEL['target']) == (C['select'], C['target'])
for s in C['arm_seeds']:  # the arms trained the selection and DoReMi's weights
    assert REC[f'regsel-s{s}']['weights'] == SEL['weights']
    assert REC[f'staged-s{s}']['stages'] == [[C['stage_frac'], SEL['weights']]]
    assert REC[f'doremi-s{s}']['weights'] == DOREMI_W
say(TAG, f'{len(REC)} runs from {os.path.basename(RUNS_PATH)}; peak '
    f'learning rate {RUNS["lr_choice"]["chosen"]:g} (chosen on the uniform '
    'mixture: ' + ', '.join(f'{lr} -> {v:.4f} bpb' for lr, v in
                             RUNS['lr_choice']['mean_bpb'].items()) + ')')
```

At T1 scale the learning-rate choice is clear and interior: a peak of
$2 \times 10^{-3}$ beats its neighbours $10^{-3}$ and $4 \times 10^{-3}$ by
0.2 and 0.09 bits per byte on the uniform mixture. It is chosen once and used
for every run. Tuning it per mixture would be tempting, since each mixture
has its own best rate, but then a difference in loss would credit the mixture
with what the rate did.
<!-- campaign: lm-mix-runs-t2-v1 -->

The registered file could, in principle, be anything. The next cell retrains
the first sixteen steps of the pure-web run on this machine and compares its
losses with the logged ones.

```{.python .input #data-mixtures-live-check}
%%tab pytorch
if GPU and not RERUN:  # the regen cell checked the pool against the runs
    cfg = C['proxy']._replace(lr=REC['pure-web']['lr'])
    live = d2l.train_proxy_lm({'web': POOLS['web']}, {'web': 1.0},
                              C['tokens'], cfg, seed=0, log_every=10,
                              stop_tokens=2**20)
    logged = {s: l for s, _, l, _ in REC['pure-web']['log']}
    diff = max(abs(l - logged[s]) for s, _, l, _ in live['log'] if s % 10 == 0)
    say(TAG, f'pure-web, steps 10-{live["log"][-1][0]}: largest difference '
        f'from the registered losses {diff:.4f} nats (bf16; tolerance 0.02)')
    assert diff <= 0.02
else:
    say('live check skipped:', 'no GPU' if not GPU else
        'the regeneration above trained the same run')
```

The same regeneration also ran a short smoke test of the T2 configuration on
one RTX 4090: 100 steps with and without `torch.compile`, a checkpoint, a
killed process and a resume. It measures what the T2 campaign will cost.
*Model FLOP utilization* (MFU) is the share of the card's peak arithmetic
rate that the training step achieves, counting $6ND$ FLOP for $N$
parameters and $D$ tokens as in :numref:`sec_scaling-laws`.

```{.python .input #data-mixtures-smoke}
%%tab pytorch
SMOKE = RUNS.get('smoke')
two = lambda v: (v if not isinstance(v, float) else f'{v:.1e}' if 0 < abs(
    v) < 1e-3 else f'{float(f"{v:.2g}"):,g}')  # two significant figures
if SMOKE is None:
    say('no smoke record in this run set')
else:
    say(f'smoke of the {SMOKE["config"]} configuration on {SMOKE["device"]} '
        f'({SMOKE["steps"]} steps of {SMOKE["tokens_per_step"]:,} tokens):')
    say(pd.Series({k: two(SMOKE[k]) for k in (
        'n_params', 'tokens_per_s', 'tokens_per_s_compiled', 'mfu',
        'mfu_compiled', 'compile_seconds', 'peak_mem_gib',
        'resume_bf16_max_diff', 'rerun_bf16_max_diff', 'resume_fp32_max_diff',
        'campaign_compile', 'campaign_flop', 'predicted_a100_hours',
        'predicted_hours_here')}).to_string())
    assert SMOKE['resume_fp32_max_diff'] <= 1e-6
    assert SMOKE['resume_bf16_ok']  # 0.01 nats, deterministic kernels
```

The T2 proxy reaches about 30% MFU when compiled, peaks at 8.4 GiB, and
resumes from a checkpoint without any difference in loss. Two fresh runs of
the same 100 steps differ by up to 0.03 nats, because the fastest bf16
kernels on this card do not add in a fixed order; regenerated runs are
therefore close to the registered ones, not identical. The whole T2 design,
49 runs and about $10^{18}$ FLOP, comes to about 3 A100-hours at the measured
utilization, or a little over five hours on one RTX 4090.

Now the first worry of the opening. If the ranking of mixtures changed during
training, a short run could not be trusted to choose. For each of the 105
pairs of training mixtures we compare their `t_dep` bits per byte at a quarter
of the run and at its end, and count a pair as crossing only if both gaps
exceed the noise of a difference between two runs, $2\sqrt{2}$ seed standard
deviations. The training runs share one data-order seed, so this bound is
conservative. The cell plots the pair with the largest crossing, or, if none
qualifies, the pair with the largest early gap. It then prints the bits per
byte of the three pure-domain proxies on every domain, and the losses of the
uniform mixture per token and per byte.

```{.python .input #data-mixtures-opening-plot}
%%tab pytorch
def t_bpb(e, t=T_DEP):  # target-weighted bits per byte of one evaluation
    return float(sum(w * e[d]['bpb'] for w, d in zip(t, DOMAINS)))

def at_frac(rec, frac):
    return t_bpb(next(e for e in rec['evals'] if abs(e['frac'] - frac) < 1e-9)
                 ['domains'])

def seed_sd(frac):  # t_dep spread of the uniform seeds, pooled (3 df)
    groups = [[at_frac(REC[f'{a}-s{s}'], frac) for s in C[k]] for a, k in (
        ('uniform-cos', 'cos_seeds'), ('uniform-wsd', 'arm_seeds'))]
    return math.sqrt(sum((len(g) - 1) * np.var(g, ddof=1) for g in groups)
                     / sum(len(g) - 1 for g in groups))

SD = {f: seed_sd(f) for f in (0.25, 1.0)}  # a gap of two runs: sqrt(2) SD
beyond_noise = lambda g: all(abs(x) > 2 * math.sqrt(2) * SD[f]
                             for x, f in zip(g[:2], (0.25, 1.0)))
train = [r for r in RUNS['runs'] if r['arm'] == 'train']
gaps = [(at_frac(a, 0.25) - at_frac(b, 0.25), at_frac(a, 1) - at_frac(b, 1),
         a, b) for a, b in itertools.combinations(train, 2)]
cross = [g for g in gaps if g[0] * g[1] < 0]
real_cross = [g for g in cross if beyond_noise(g)]
G = (max(real_cross, key=lambda g: abs(g[1])) if real_cross else
     max(gaps, key=lambda g: abs(g[0])))
A, B = G[2:]
say(TAG, f'{len(cross)} of {len(gaps)} pairs of training mixtures cross, '
    f'{len(real_cross)} with both gaps above 2 sqrt(2) seed SDs (SD '
    f'{SD[0.25]:.4f} at 25%, {SD[1.0]:.4f} at the end); ' + (
        'the pair with the largest final gap' if real_cross else
        'none beyond the noise: the pair with the largest early gap')
    + f': gap {G[0]:+.4f} at 25%, {G[1]:+.4f} at the end')
d2l.plot([[e['tokens'] for e in r['evals']] for r in (A, B)],
         [[t_bpb(e['domains']) for e in r['evals']] for r in (A, B)],
         'tokens', 't_dep bits per byte', legend=[
             'w = (' + ', '.join(f'{x:.2f}' for x in r['weights']) + ')'
             for r in (A, B)], figsize=(4.5, 3))
MATRIX = pd.DataFrame([[REC[f'pure-{a}']['final'][b]['bpb'] for b in DOMAINS]
                       for a in DOMAINS], columns=DOMAINS,
                      index=[f'trained on {a}' for a in DOMAINS])
say(TAG, 'bits per byte on each validation set (columns) of the '
    'pure-domain runs (rows):')
say(MATRIX.round(3).to_string())
say('each proxy is best on its own domain:', all(
    MATRIX.values[i, i] == MATRIX.values[:, i].min() for i in range(3)))
u = REC['uniform-wsd-s0']['final']
say(TAG, 'uniform mixture (WSD, seed 0) per domain:')
say(pd.DataFrame({d: dict(
    bytes_per_token=u[d]['n_bytes'] / u[d]['n_tokens'],
    bits_per_token=u[d]['nll'] / u[d]['n_tokens'] / math.log(2),
    bits_per_byte=u[d]['bpb']) for d in DOMAINS}).round(3).to_string())
```

At T1 scale the worry does not materialize: about one pair in ten swaps
places between a quarter of the run and its end, and none by more than the
noise. The plotted pair is 0.15 bits per byte apart at both points. Early
rankings predict late ones here, which is what makes proxy runs usable at
all; it does not yet say that small models predict large ones.

The pure-domain matrix shows why mixing is a real choice. At T1 scale each
proxy is best on its own domain, and the costs off the diagonal are large and
asymmetric: the web proxy scores 5.2 bits per byte on Python, the code proxy
2.6 on web text, but the mathematics proxy reaches 2.15 on web text, close to
the web proxy's 2.05. FineMath is a collection of web pages, written in
English prose about mathematics, so most of what it teaches carries over to
web text. The last table makes the earlier warning concrete. Per token, code
looks like the easiest domain, 5.1 bits against 8.7 for web text; per byte it
is the harder one, 2.1 against 1.9, because a token of Python stands for less
text.
<!-- campaign: lm-mix-runs-t2-v1 -->

### Regression and Its Noise Floor
:label:`subsec_lm_mix_floor`

Three regressions map the weights to the per-domain bits per byte. *Ridge*
regression is linear in the weights. Since the three weights sum to one, the
three of them and an intercept are collinear, so ridge uses two free weights
and an intercept. *Gradient boosting*, an ensemble of shallow trees, is
RegMix's learner. The third is the *exponential mixing law* of Ye and
colleagues, one curve per domain $i$:

$$
b_i(\mathbf{w}) = c_i + k_i \exp\Bigl(\sum_j t_{ij} w_j\Bigr),
$$
:eqlabel:`eq_lm_mix_law`

with five parameters per domain, fitted by least squares
from several random starts. A negative $t_{ii}$ lets the loss on domain $i$
fall as its own weight rises and level off at $c_i$; the cross terms
$t_{ij}$ say how much the other domains help or hurt. `choose_mixture`
minimizes a fitted target loss over 5,000 random mixtures, polishes the best
one by Nelder-Mead with the weights written as a softmax, and stays inside the
convex hull of the training mixtures, where the regression was fitted. The
mixing law was fixed as the selecting regression before any run, for a reason
that has nothing to do with accuracy. A model linear in $\mathbf{w}$, like
ridge, predicts a target loss that is linear in $\mathbf{w}$, and a linear
function on a polytope has its minimum at a vertex. Ridge can only ever
choose a corner of the region it is given. The law's exponentials can place
the optimum inside.

```{.python .input #data-mixtures-reg-defs}
%%tab pytorch
#@save
def fit_mixture_regression(W, L, kind='ridge', **kw):
    """A predictor f(w) -> per-domain losses fitted on mixtures W (n, m)
    and their losses L (n, k): 'ridge' on the first m - 1 weights and an
    intercept (alpha=1e-3, intercept not shrunk), 'gbm' (scikit-learn
    gradient boosting per output, seed 0) or 'mixlaw' (L_i = c_i + k_i
    exp(sum_j t_ij w_j) per output, least squares from `starts` seeded
    starts, default 8; init=params of an earlier fit is tried first).
    f takes (m,) or (n', m); f.params holds the fitted parameters."""
    W, L = np.asarray(W, dtype=float), np.asarray(L, dtype=float)
    design = lambda w: np.hstack([np.ones((len(w), 1)), w[:, :-1]])
    if kind == 'ridge':
        X = design(W)
        P = kw.get('alpha', 1e-3) * np.eye(X.shape[1])
        P[0, 0] = 0
        params = np.linalg.solve(X.T @ X + P, X.T @ L)
        predict = lambda w: design(w) @ params
    elif kind == 'gbm':
        from sklearn.ensemble import GradientBoostingRegressor
        opts = dict(n_estimators=200, learning_rate=0.05, max_depth=2)
        models = [GradientBoostingRegressor(random_state=0, **{**opts, **kw})
                  .fit(W, L[:, i]) for i in range(L.shape[1])]
        params = None
        predict = lambda w: np.stack([g.predict(w) for g in models], -1)
    elif kind == 'mixlaw':
        from scipy.optimize import least_squares
        rng, params = np.random.default_rng(0), []
        law = lambda p, w: p[0] + p[1] * np.exp(w @ p[2:])
        init, starts = kw.get('init'), kw.get('starts', 8)
        for i, y in enumerate(L.T):
            x0 = [] if init is None else [np.asarray(init[i], dtype=float)]
            x0 += [np.concatenate([[rng.uniform(0, y.min())],
                                   [rng.uniform(0.1, 2)],
                                   rng.normal(0, 1, W.shape[1])])
                   for _ in range(starts - len(x0))]
            fits = [least_squares(lambda p: law(p, W) - y, x, xtol=1e-15,
                                  ftol=1e-15, gtol=1e-15, max_nfev=5000)
                    for x in x0]
            params.append(min(fits, key=lambda r: r.cost).x)
        params = np.array(params)
        predict = lambda w: np.stack([law(p, w) for p in params], -1)
    else:
        raise ValueError(f'unknown kind {kind!r}')
    def f(w):
        w = np.asarray(w, dtype=float)
        return predict(w[None])[0] if w.ndim == 1 else predict(w)
    f.params = params
    return f

#@save
def choose_mixture(f, target, n_grid=5000, seed=0, max_weight=0.9,
                   within=None):
    """The mixture w with no weight above max_weight that minimizes the
    fitted target loss target @ f(w): the best of n_grid Dirichlet(1)
    points, polished by Nelder-Mead on w = softmax(z). within (n, m), if
    given, bounds the choice to the convex hull of those mixtures (where
    f was fitted, so the choice never rests on extrapolation). Returns
    (w, the predicted target loss)."""
    from scipy.optimize import minimize
    from scipy.spatial import Delaunay
    t = np.asarray(target, dtype=float)
    hull = None if within is None else Delaunay(np.asarray(within)[:, :-1])
    inside = lambda w: (np.ones(len(w), bool) if hull is None else
                        hull.find_simplex(w[:, :-1]) >= 0)
    G = d2l.sample_mixtures(n_grid, t, 1.0, seed, max_weight)
    G = G[inside(G)]
    vals = f(G) @ t
    softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    def loss(z):
        w = softmax(z)
        if w.max() > max_weight or not inside(w[None])[0]:
            return np.inf
        return float(f(w) @ t)
    res = minimize(loss, np.log(G[vals.argmin()]), method='Nelder-Mead',
                   options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    if res.fun < vals.min():
        return softmax(res.x), float(res.fun)
    return G[vals.argmin()], float(vals.min())
```

How well can any regression do? Each measured loss carries noise of standard
deviation $\sigma$ from the data order, and the held-out loss we compare a
prediction with is just as noisy. For a model linear in $p$ parameters
fitted by least squares to $n$ runs with design matrix $\mathbf{X}$, the
prediction at a new point $\mathbf{x}$ is
$\hat y = \mathbf{x}^\top \hat{\boldsymbol\beta}$ with
$\hat{\boldsymbol\beta} - \boldsymbol\beta = (\mathbf{X}^\top
\mathbf{X})^{-1} \mathbf{X}^\top \boldsymbol\epsilon$, and the new
observation is $y = \mathbf{x}^\top \boldsymbol\beta + \epsilon'$ with
independent noise. The two error terms are independent, so

$$
\mathbb{E}\bigl[(\hat y - y)^2\bigr]
= \sigma^2 \bigl(1 + \mathbf{x}^\top (\mathbf{X}^\top \mathbf{X})^{-1}
\mathbf{x}\bigr),
\qquad
\frac{1}{n} \sum_{k=1}^{n} \mathbf{x}_k^\top (\mathbf{X}^\top
\mathbf{X})^{-1} \mathbf{x}_k = \frac{p}{n}.
$$
:eqlabel:`eq_lm_mix_floor`

The second identity is the trace of the hat matrix
$\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$, which is $p$.
Averaged over the design, the root-mean-square error against held-out runs
cannot fall below $\sigma\sqrt{1 + p/n}$, however good the model class. A
regression that reaches this *noise floor* has extracted everything the runs
contain; one that is far above it is wrong about the shape of the function.
The cell checks :eqref:`eq_lm_mix_floor` by simulation, estimates $\sigma$
from the two extra seeds of the uniform mixture, fits the three regressions
on the training runs and judges them only on the held-out ones.

Two design facts qualify the comparison. Every training and held-out run uses
the same data-order seed, so part of the noise is shared and the floor
estimated from independent seeds is an upper bound on the noise the held-out
errors face. And the same validation sets choose the learning rate, fit the
regression and score the held-out mixtures: "held out" means held-out
mixtures, not held-out text. Those choices have few degrees of freedom, a
learning rate and two free weights, so the bias they introduce is small, but
it is not zero. Panel (a) of the figure plots predicted against measured bits
per byte for each domain with the identity line and a band of two floors;
panel (b) places every run on the simplex, coloured by its `t_dep` loss, with
the selected mixture as a star.

```{.python .input #data-mixtures-regression}
%%tab pytorch
rng = np.random.default_rng(0)  # D4: n = 16 mixtures, linear truth
Wd, Wn = d2l.sample_mixtures(16, DOMAINS, 1.0, 0), d2l.sample_mixtures(
    200, DOMAINS, 1.0, 1)
beta, sigma = np.array([1.2, -0.3, 0.2]), 0.02
X, Xn = (np.hstack([np.ones((len(w), 1)), w[:, :2]]) for w in (Wd, Wn))
err = np.zeros(len(Wn))
for _ in range(5000):
    f = d2l.fit_mixture_regression(Wd, (X @ beta + sigma * rng.normal(
        size=16))[:, None], 'ridge', alpha=0.0)
    err += (f(Wn)[:, 0] - Xn @ beta - sigma * rng.normal(size=len(Wn)))**2
theory = sigma**2 * (1 + np.einsum('ij,jk,ik->i', Xn, np.linalg.inv(X.T @ X),
                                   Xn))
assert abs((err / 5000).mean() / theory.mean() - 1) < 0.03
say(f'D4: mean squared error at new mixtures {(err / 5000).mean():.3e}, '
    f'theory {theory.mean():.3e}; at the design points sigma^2 (1 + p/n) = '
    f'{sigma**2 * (1 + 3 / 16):.3e}')

final = lambda r: [REC[r]['final'][d]['bpb'] for d in DOMAINS]
cos = np.array([final(f'uniform-cos-s{s}') for s in C['cos_seeds']])
wsd = np.array([final(f'uniform-wsd-s{s}') for s in C['arm_seeds']])
sig_cos, sig_wsd = cos.std(0, ddof=1), wsd.std(0, ddof=1)
SIG_COS = math.sqrt((sig_cos**2).mean())  # pooled: 3 degrees of freedom
SIG_SEED = math.sqrt((sig_wsd**2).mean())  # pooled: 6
train_ids = [r['run_id'] for r in RUNS['runs'] if r['arm'] == 'train']
held_ids = [r['run_id'] for r in RUNS['runs'] if r['arm'] == 'held']
Wt, Lt = (np.array([REC[r]['weights'] for r in train_ids]),
          np.array([final(r) for r in train_ids]))
Wh, Lh = (np.array([REC[r]['weights'] for r in held_ids]),
          np.array([final(r) for r in held_ids]))
FLOOR = SIG_COS * math.sqrt(1 + 3 / len(train_ids))
FITS = {k: d2l.fit_mixture_regression(Wt, Lt, k)
        for k in ('ridge', 'gbm', 'mixlaw')}
PRED = {k: f(Wh) for k, f in FITS.items()}
noise = {'sigma_cos': sig_cos, 'sigma_seed (WSD)': sig_wsd,
         'floor sigma_cos sqrt(1 + p/n)': np.full(3, FLOOR)}
noise.update({f'RMSE {k}': np.sqrt(((P - Lh)**2).mean(0))
              for k, P in PRED.items()})
noise.update({f'MAE {k}': np.abs(P - Lh).mean(0) for k, P in PRED.items()})
noise.update({f'Spearman {k}': [stats.spearmanr(P[:, i], Lh[:, i])[0]
                                for i in range(3)] for k, P in PRED.items()})
say(TAG, f'{len(train_ids)} training and {len(held_ids)} held-out runs; '
    f'pooled sigma_cos {SIG_COS:.4f}, sigma_seed {SIG_SEED:.4f} bits per '
    f'byte (ratio {max(SIG_COS, SIG_SEED) / min(SIG_COS, SIG_SEED):.1f})')
say(pd.DataFrame(noise, index=DOMAINS).T.round(4).to_string())
FLOOR_CI = FLOOR * np.sqrt(3 / stats.chi2.ppf([0.975, 0.025], 3))  # 3 df
rmse = math.sqrt(((PRED[C['select']] - Lh)**2).mean())
say(f'floor {FLOOR:.4f} (95% interval from 3 degrees of freedom '
    f'{FLOOR_CI[0]:.4f}-{FLOOR_CI[1]:.4f}); {C["select"]} (the selecting '
    f'regression), held-out RMSE over all domains {rmse:.4f}:',
    'reaches only the noise floor' if rmse <= 1.5 * FLOOR else
    'does not reach the noise floor')

fig, axes = d2l.plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw=dict(
    width_ratios=(1, 1, 1, 1.5)))
for i, (ax, d) in enumerate(zip(axes, DOMAINS)):  # (a) one panel per domain
    lo, hi = Lh[:, i].min() - 3 * FLOOR, Lh[:, i].max() + 3 * FLOOR
    d2l.plot([P[:, i] for P in PRED.values()] + [[lo, hi]],
             [Lh[:, i]] * 3 + [[lo, hi]], f'predicted ({d})',
             'measured bits per byte' if i == 0 else None,
             legend=list(PRED) + ['identity'] if i == 0 else None,
             fmts=('o', 's', '^', 'k-'), axes=ax)
    ax.fill_between([lo, hi], [lo - 2 * FLOOR, hi - 2 * FLOOR],
                    [lo + 2 * FLOOR, hi + 2 * FLOOR], color='0.85', zorder=0)
ax2 = axes[3]  # (b) the simplex
ids = train_ids + held_ids + [r['run_id'] for r in RUNS['runs']
                              if r['arm'] in ('pure', 'uniform-cos')]
pts = d2l.barycentric(np.array([REC[r]['weights'] for r in ids]))
dots = ax2.scatter(*pts.T, c=[T_DEP @ final(r) for r in ids], cmap='viridis')
ax2.plot(*d2l.barycentric(np.array([RUNS['selection']['weights']])).T, '*',
         color='red', markersize=16, label='selected')
ax2.plot(*d2l.barycentric(np.vstack([np.eye(3), np.eye(3)[:1]])).T, 'k-',
         linewidth=0.8)
for (x, y), d in zip(d2l.barycentric(np.eye(3)), DOMAINS):
    ax2.annotate(d, (x, y), xytext=(0, -14 if y == 0 else 6),
                 textcoords='offset points', ha='center')
ax2.set_aspect('equal')
ax2.axis('off')
ax2.legend(loc='upper left', fontsize='small')
bar = fig.colorbar(dots, ax=ax2, label='t_dep bits per byte', shrink=0.8)
fig.tight_layout()
```

The simulation matches the formula to within a few tenths of a percent. On
the real runs, at T1 scale, the floor is about 0.02 bits per byte, and its
interval from three degrees of freedom is wide, 0.01 to 0.06. No regression
reaches it. Gradient boosting comes closest with a held-out error of about
0.03, ridge has about 0.06, and the mixing law about 0.07. The law's error is
almost all in one domain: on web text it predicts to a few thousandths, on
code it misses by about 0.12. Code's loss falls steeply as its weight rises
from zero, the web proxy scores 5.2 bits per byte on Python, and two of the
held-out mixtures sit at 5% code, the region where a single exponential
has the hardest time following it. The method's limit at this scale is therefore the shape of
the law, not the number of runs, since more runs lower the floor but not the
bias. A good fit to the training mixtures would have said nothing about
this: the law has five parameters per domain for fifteen training runs,
and gradient boosting can match its training points almost exactly.
<!-- campaign: lm-mix-runs-t2-v1 -->

## Choosing a Mixture
:label:`subsec_lm_mix_choose`

With a fitted function in hand, the choice is an optimization. The cell
minimizes each regression's prediction of the `t_dep` loss, the mixing law's
also for `t_avg` and without the bound to the training hull, and refits the
law on 200 bootstrap resamples of the training runs to see how firmly the data
determine the chosen weights.

```{.python .input #data-mixtures-choose}
%%tab pytorch
pick = lambda f, t, W=Wt: d2l.choose_mixture(f, t, max_weight=C[
    'max_weight'], within=W)
choice = {k: pick(f, T_DEP) for k, f in FITS.items()}
choice[f'{C["select"]}, t_avg'] = pick(FITS[C['select']], T_AVG)
choice[f'{C["select"]}, no hull'] = pick(FITS[C['select']], T_DEP, None)
assert np.allclose(choice[C['select']][0], SEL['weights'], atol=1e-6), (
    'the launcher chose another mixture')
assert abs(choice[C['select']][1] - SEL['predicted']) < 1e-9
HULL = spatial.Delaunay(Wt[:, :-1])  # the region the regression has seen
say(TAG, 'selected mixtures (weights web, code, math) and their predicted '
    'target-weighted bits per byte (the last one free of the hull bound):')
for k, (w, v) in choice.items():
    where = ('inside the training mixtures' if HULL.find_simplex(w[:-1]) >= 0
             else 'extrapolated')
    say(f'  {k:18s} w = ({", ".join(f"{x:.3f}" for x in w)}), predicted '
        f'{v:.4f}, {where}')
B = 200  # bootstrap refits; the mixing law starts from the full fit
warm = (dict(init=FITS['mixlaw'].params, starts=1)
        if C['select'] == 'mixlaw' else {})
rng, boot = np.random.default_rng(0), []
for _ in range(B):
    k = rng.integers(0, len(Wt), len(Wt))
    boot.append(pick(d2l.fit_mixture_regression(Wt[k], Lt[k], C['select'],
                                                 **warm), T_DEP, Wt[k]))
bw, bv = np.array([w for w, _ in boot]), np.array([v for _, v in boot])
lo, hi = np.percentile(bv, [2.5, 97.5])
say(f'{C["select"]} on {B} bootstrap resamples of the training runs: '
    f'predicted t_dep {np.median(bv):.4f} [{lo:.4f}, {hi:.4f}]; weights '
    'from ' + ', '.join(f'{d} {a:.2f}-{b:.2f}' for d, a, b in zip(
        DOMAINS, *np.percentile(bw, [2.5, 97.5], axis=0))))
```

The selected mixture at T1 scale is the result of this section that a reader
would least expect. For a target that puts 70% of its weight on web text, the
mixing law chooses about a third of each domain, $(0.34, 0.31, 0.35)$. The
reason is in the matrix above: web loss barely moves with the web weight,
because mathematics pages teach web prose almost as well as web pages do,
while the losses on code and mathematics fall steeply with their own weights.
The optimizer spends tokens where they lower the loss most, and a deployment
that reads mostly web text does not, at this scale, want mostly web text in
training. Without the hull bound the law picks the same point, so it is not
pressed against the edge of its data. Ridge, as promised, picks a corner, the
fixed mixture $(0.1, 0.8, 0.1)$; gradient boosting picks
$(0.24, 0.55, 0.22)$. For `t_avg` the law shifts weight from web text to
mathematics, $(0.20, 0.36, 0.43)$. The bootstrap is sobering: across
resamples each weight moves over a range of about 0.15, web from 0.25 to
0.38, code from 0.24 to 0.43. The runs determine a neighbourhood, not a
point.
<!-- campaign: lm-mix-runs-t2-v1 -->

Had we picked the regression after seeing the held-out errors, we would have
picked gradient boosting and a code-heavy mixture. The rule that the selecting
regression and the target are fixed in advance is what keeps two regressions,
two targets and a hull bound from becoming a menu.

### DoReMi
:label:`subsec_lm_mix_doremi`

DoReMi (Xie and colleagues, 2023) chooses the weights without a regression,
and with one difference in what it optimizes. It trains a reference model on
a default mixture, then trains a proxy whose loss weights $\boldsymbol\alpha$
over domains are adjusted during training to maximize the proxy's *excess*
loss over the reference, so that the proxy works hardest on the domains where
it lags the reference most. This is a minimax game,
$\min_\theta \max_{\boldsymbol\alpha \in \Delta} \sum_i \alpha_i
(\ell_i(\theta) - \ell_{\textrm{ref},i})$, and the step on the weights has a
closed form. With $g_{t,i}$ the excess loss of domain $i$ at step $t$,

$$
\boldsymbol\alpha_{t+1}
= \mathop{\mathrm{argmax}}_{\boldsymbol\alpha \in \Delta}
\langle \boldsymbol\alpha, \mathbf{g}_t \rangle
- \frac{1}{\eta} \textrm{KL}(\boldsymbol\alpha \,\|\, \boldsymbol\alpha_t)
\quad\Longrightarrow\quad
\alpha_{t+1,i} = \frac{\alpha_{t,i}\, e^{\eta g_{t,i}}}
{\sum_j \alpha_{t,j}\, e^{\eta g_{t,j}}},
$$
:eqlabel:`eq_lm_mix_doremi`

which follows from setting the derivative of the Lagrangian to zero: the
stationarity condition gives $\log(\alpha_i / \alpha_{t,i}) = \eta g_i$ plus a
constant, and normalization fixes the constant. The weights are then mixed
with a little of the uniform distribution and averaged over all steps, and
the average is the mixture that a larger model is trained on. The excess loss
is clipped at zero per token, against the reference model's loss on the same
tokens of the same batch, before it is averaged over a domain. The paper used
a proxy of 280 million parameters to set the weights of a model of 8 billion
on the Pile. Our miniature follows it with the T1 proxy, taking the
final checkpoint of the first uniform-mixture arm as the reference.

```{.python .input #data-mixtures-doremi-def}
%%tab pytorch
#@save
def doremi_weights(ref_model, proxy_model, domain_batches, steps, cfg,
                   eta=1.0, smooth=1e-3, *, lr=None):
    """DoReMi's domain weights: train proxy_model on the alpha-weighted
    domain losses while alpha follows each domain's excess loss over the
    frozen ref_model (per token, clipped at 0) by multiplicative weights.
    domain_batches(step) gives {domain: LongTensor (b, seq_len + 1)};
    lr(step) is the learning rate (default cfg.lr). Returns the
    step-averaged alpha and the per-step trace."""
    opt, trace, alpha = d2l._proxy_optimizer(proxy_model, cfg), [], None
    ref_model.eval()
    for step in range(steps):
        batches = domain_batches(step)
        if alpha is None:
            alpha = np.full(len(batches), 1 / len(batches))
        excess = []
        with torch.no_grad():  # both models on the same tokens
            for X in batches.values():
                nll = [torch.cat([d2l._per_token_nll(m, Xm, cfg) for Xm in
                                  X.split(cfg.micro_batch)])
                       for m in (proxy_model, ref_model)]
                excess.append(d2l._doremi_excess(*nll))
        alpha = d2l._doremi_step(alpha, excess, eta, smooth)
        for g in opt.param_groups:
            g['lr'] = cfg.lr if lr is None else lr(step)
        for a, X in zip(alpha, batches.values()):  # step on sum_i alpha_i l_i
            for Xm in X.split(cfg.micro_batch):
                (a * d2l._per_token_nll(proxy_model, Xm, cfg).mean()
                 * len(Xm) / len(X)).backward()
        nn.utils.clip_grad_norm_(proxy_model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        trace.append(dict(alpha=alpha.tolist(), excess=excess))
    return np.mean([t['alpha'] for t in trace], 0), trace
```

The cell checks the closed-form step against a numerical maximization, runs
the game on a toy with three quadratic losses whose saddle point is known,
and builds a batch on which clipping the mean excess loss of a domain would
give no signal at all while clipping per token does. It then prints the
weights of the registered run.

```{.python .input #data-mixtures-doremi}
%%tab pytorch
a_t, g, eta = np.array([0.5, 0.3, 0.2]), np.array([0.4, 1.0, -0.2]), 0.7
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
kl_obj = lambda z: -(softmax(z) @ g - np.sum(softmax(z) * np.log(
    softmax(z) / a_t)) / eta)  # <alpha, g> - KL(alpha || alpha_t) / eta
num = softmax(optimize.minimize(kl_obj, np.zeros(3), method='BFGS',
                                options=dict(gtol=1e-12)).x)
gap = np.abs(num - d2l._doremi_step(a_t, g, eta, 0.0)).max()
assert gap < 1e-8
c, theta, alpha, path = np.array([0.0, 1.0, 3.0]), 0.0, np.full(3, 1 / 3), []
for _ in range(20000):  # l_i = (theta - c_i)^2, reference loss 0
    alpha = d2l._doremi_step(alpha, (theta - c)**2, 0.05, 1e-3)
    theta -= 0.01 * 2 * alpha @ (theta - c)
    path.append(alpha)
grid = np.linspace(-1, 4, 501)
A_ub = -((grid[None] - c[:, None])**2).T  # max v s.t. sum_i a_i l_i >= v
lp = optimize.linprog(np.r_[np.zeros(3), -1], A_ub=np.c_[A_ub, np.ones(
    len(grid))], b_ub=np.zeros(len(grid)), A_eq=[[1, 1, 1, 0]], b_eq=[1],
                      bounds=[(0, 1)] * 3 + [(None, None)])
saddle = np.abs(np.mean(path, 0) - lp.x[:3]).max()
assert saddle < 0.02
say(f'D3: closed-form step = numerical maximizer to {gap:.1e}; averaged '
    f'weights {np.mean(path, 0).round(3)} vs the saddle '
    f'{lp.x[:3].round(3)} (largest gap {saddle:.3f})')
torch.manual_seed(0)
ref, prox = (d2l._proxy_model(tiny, seed) for seed in (0, 1))
Xd = torch.randint(0, tiny.vocab_size, (2, tiny.seq_len + 1))
with torch.no_grad():
    lp_, lr_ = (d2l._per_token_nll(m, Xd, tiny) for m in (prox, ref))
per_token = torch.clamp(lp_ - lr_, min=0).mean().item()
per_domain = max((lp_ - lr_).mean().item(), 0.0)
_, tr = d2l.doremi_weights(ref, prox, lambda k: {'d': Xd}, 1, tiny)
assert abs(tr[0]['excess'][0] - per_token) < 1e-6
assert abs(per_domain - per_token) > 1e-3  # domain-level clipping differs
say(f'excess loss of a random batch: per token {per_token:.4f}, from the '
    f'domain mean {per_domain:.4f}; doremi_weights used '
    f'{tr[0]["excess"][0]:.4f}')
DOREMI_W = np.array(RUNS['doremi']['weights'])
steps = [0, len(RUNS['doremi']['trace']) // 4, len(RUNS['doremi']['trace'])
         // 2, len(RUNS['doremi']['trace']) - 1]
say(TAG, 'DoReMi weights (step average):', dict(zip(
    DOMAINS, DOREMI_W.round(3).tolist())), '| alpha at steps', {
    s + 1: np.round(RUNS['doremi']['trace'][s]['alpha'], 3).tolist()
    for s in steps})
```

The step and the toy agree with their derivations. The clipping check shows
why the order matters: on the random batch the domain's mean excess loss is
zero, so clipping it gives nothing, while the per-token clip keeps the 0.05
nats by which the proxy is worse on some tokens. At T1 scale DoReMi raises
code and lowers web text, $(0.17, 0.47, 0.36)$. The trajectory behind that
average swings: code holds three quarters of the weight a quarter of the way
in and a quarter of it halfway. With 256 steps the average of so restless a
sequence is a noisy estimate; the paper's runs are many thousands of steps
long. Note also what DoReMi does not do. Its objective is the worst excess
loss over domains, not `t_dep` and not even the average loss, so there is no
reason to expect its mixture to win on either target.
<!-- campaign: lm-mix-runs-t2-v1 -->

### Four Arms Compared
:label:`subsec_lm_mix_arms`

Four mixtures are now trained with three data-order seeds each: uniform, the
regression's choice for `t_dep`, DoReMi's weights, and a staged arm that reads
the uniform mixture for 80% of its tokens and the regression's choice for the
rest. All four use the stable-then-decay schedule with the same peak, so that
only the data differ, and each arm's seed $s$ is compared with the uniform
arm's seed $s$, which saw its data in the same order. With three seeds no
bootstrap interval means much, so the rule is blunt: an arm differs from
uniform only if all three paired differences share a sign and their mean
exceeds two seed standard deviations of the uniform arm. The arms use a
different schedule from the training runs, so their losses are compared with
each other, never with the regression's predictions.

```{.python .input #data-mixtures-arms}
%%tab pytorch
ARMS = ('uniform', 'regsel', 'doremi', 'staged')
SEEDS = C['arm_seeds']
T = {a: np.array([T_DEP @ final(f'{"uniform-wsd" if a == "uniform" else a}'
                                f'-s{s}') for s in SEEDS]) for a in ARMS}
sig_t = T['uniform'].std(ddof=1)  # sigma_seed of t_dep bits per byte
rows = []
for a in ARMS:
    w = REC[f'{"uniform-wsd" if a == "uniform" else a}-s0']
    d = T[a] - T['uniform']
    real = a != 'uniform' and len(set(np.sign(d))) == 1 and abs(
        d.mean()) > 2 * sig_t
    rows.append(dict(arm=a, weights=np.round(w['weights'], 3).tolist(),
                     t_dep_mean=T[a].mean(),
                     diffs=np.round(d, 4).tolist(), real=real))
ARM_TABLE = pd.DataFrame(rows)
(frac, w_dec), = REC['staged-s0']['stages']
say(TAG, f'sigma_seed of t_dep bits per byte {sig_t:.4f} (3 uniform '
    'seeds); differences seed-paired with uniform (staged: the weights '
    f'below, then {np.round(w_dec, 3).tolist()} from {frac:.0%} of the '
    'tokens):')
say(ARM_TABLE.round(4).to_string(index=False))
ARM_DOMAINS = pd.DataFrame({a: dict(zip(DOMAINS, np.mean([final(
    f'{"uniform-wsd" if a == "uniform" else a}-s{s}') for s in SEEDS], 0)))
    for a in ARMS}).T  # DoReMi's objective is no single target
ARM_DOMAINS['t_dep'] = ARM_DOMAINS[list(DOMAINS)].values @ T_DEP
ARM_DOMAINS['t_avg'] = ARM_DOMAINS[list(DOMAINS)].values @ T_AVG
say(TAG, 'mean bits per byte of each arm over its seeds:')
say(ARM_DOMAINS.round(4).to_string())
ax = d2l.plot_intervals(
    list(ARMS), {'mean': ([T[a].mean() for a in ARMS],) * 3}, 'arm',
    't_dep bits per byte', reference=T['uniform'].mean(),
    marks={'seeds': ('.', {'mean': (np.repeat(np.arange(4), len(SEEDS)),
                                    np.concatenate([T[a] for a in ARMS]))})})
band = ax.axhspan(T['uniform'].mean() - 2 * sig_t,
                  T['uniform'].mean() + 2 * sig_t, color='0.85', zorder=0)
```

At T1 scale the regression's arm is ahead of uniform in all three seeds, by
about 0.007 bits per byte on `t_dep`, which is within the noise; that is no
surprise, since the selected mixture is nearly uniform. DoReMi's arm is
behind on `t_dep` by about 0.01 to 0.02. The rule calls that real, but with
little margin, and regenerating the run set moves single runs by up to about
0.02 bits per byte, so we read it as "worse on `t_dep` by about 0.01, at the
edge of the noise". The per-domain table explains the sign. DoReMi spent its
extra code weight well, lowering the code loss by about 0.1 bits per byte,
and paid with web text, which `t_dep` weighs by 0.7. On the average target the
same arm is better than uniform by about 0.025. One method, two targets, two
verdicts: a mixture is good only for a stated objective, and DoReMi's
objective is neither of ours.
<!-- campaign: lm-mix-runs-t2-v1 -->

## Annealing and Curricula
:label:`subsec_lm_mix_anneal`

The staged arm asks a different question: does it matter when in training a
mixture is read? Under the stable-then-decay schedule of
:numref:`sec_scheduler` the learning rate stays at its peak for most of
training and then falls to a small value over the last fifth of the steps,
and most of the drop in loss happens in that decay. A model fits most closely
what it reads while its learning rate falls, which suggests saving the best
data for the end. This is *annealing*, and it is now standard: SmolLM2's last
stage, the final 10% of its 11 trillion tokens, raises mathematics to 14% and
code to 24% of the mixture, using FineMath and educational code that were
held back until then to make them count most. The staged arm is the miniature of this recipe, and the cell
compares its per-domain curves with the uniform arm's, with the stage
boundary marked.

```{.python .input #data-mixtures-staged}
%%tab pytorch
curve = lambda a, d: np.mean([[e['domains'][d]['bpb'] for e in REC[
    f'{a}-s{s}']['evals']] for s in SEEDS], 0)
x = [e['tokens'] for e in REC['staged-s0']['evals']]
staged = REC['staged-s0']
n_steps, frac = staged['steps'], staged['stages'][0][0]
start = n_steps - round((1 - frac) * n_steps)  # first step of the decay
boundary = start * staged['tokens'] // n_steps
fig, axes = d2l.plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for ax, d in zip(axes, DOMAINS):
    d2l.plot(x, [curve('uniform-wsd', d), curve('staged', d)], 'tokens',
             f'{d} bits per byte', legend=['uniform', 'staged'], axes=ax)
    ax.axvline(boundary, color='black', linestyle=':', linewidth=1)
fig.tight_layout()
dec = [k for k, e in enumerate(REC['staged-s0']['evals'])
       if e['step'] > start]  # evaluations inside the decay
say(TAG, 'change of bits per byte during the decay, staged minus uniform '
    f'(mean over {len(SEEDS)} seeds; staged weights '
    f'{np.round(REC["staged-s0"]["stages"][0][1], 3).tolist()}):')
say({d: round(float((curve('staged', d) - curve('uniform-wsd', d))[dec[-1]]
                    - (curve('staged', d) - curve('uniform-wsd', d))
                    [dec[0] - 1]), 4) for d in DOMAINS})
```

At T1 scale the staged arm cannot show the effect, and for a plain reason:
it switches from the uniform mixture to the regression's choice, which is
nearly uniform, so the decay reads almost the same data in both arms. The
changes during the decay are a few thousandths of a bit per byte, and the
curves lie on top of each other. The arm shows that a staged run is one
argument to `train_proxy_lm`; what annealing on a different mixture does is
the subject of an exercise.
<!-- campaign: lm-mix-runs-t2-v1 -->

The harder question is what an annealing gain means. If the decay is run on
mathematics and question-answer pairs, scores on mathematics and
question-answering benchmarks rise, and they rise fast. Some of that rise can
be *recency*: the model has adapted to the format and vocabulary of what it
read last, not acquired a skill it lacked, and a gain of that kind fades when
training continues on other data. Two checks separate the cases. One measures
the loss on held-out data of the annealed domain that does not resemble any
benchmark; the other continues training on the general mixture after the
anneal and asks whether the gain survives. Blakeney and colleagues (2024)
proposed upsampling domains at the end of training as a cheap way to test
which data help, and the method is valuable exactly because it is fast; it
should not be read as a measure of what a model can do after the anneal
is over. Decontamination, as in :numref:`sec_lm_dedup`, matters doubly here,
since annealing data are often written to look like test items.

### Two Released Mixtures
:label:`subsec_lm_mix_cases`

Olmo 3 documents its data in unusual detail, and it reads best as a case
study of everything above rather than as a recipe to copy
:cite:`OLMo3.2025`. Its pool, Dolma 3, holds 9.3 trillion tokens, almost all
of it Common Crawl (8.1 trillion), with 970 billion tokens of scientific PDFs
converted to text, 137 billion of educational code, 34 billion of FineMath,
21 billion of arXiv and 4 billion of Wikipedia. The pretraining mixture takes
5.9 trillion tokens from it. By the two dataset cards' counts, Common Crawl is
subsampled to a little over half of its pool, while code is read about three
times, arXiv about two and a half times and FineMath about four and a half
times: the small, valuable sources are upsampled into the range where
:eqref:`eq_lm_mix_effective` starts to charge for repetition, and the report
states that it avoids repeating any domain more than a few times. The weights
come from our method at a larger scale: a swarm of proxies of 30 million
parameters trained for 3 billion tokens each, a hundred tokens per parameter,
on mixtures drawn from a Dirichlet distribution centred on the natural
proportions, with five times as many proxies as domains. A separate
regression per evaluation task predicts its bits per byte, and the mixture
minimizes the average prediction under the repetition constraint. After
pretraining comes a mid-training stage of 100 billion tokens, the Dolmino
mixture, which combines the highest-quality pretraining data with
mathematics, code, question answering, instruction following and reasoning
traces, the annealing of the previous paragraphs at scale; candidate data for
it are tested by short anneals of 10 billion tokens, half candidate, half
web. :numref:`sec_lm_continued_pretraining` takes up that stage.

SmolLM2 shows the other common pattern, a curriculum of four stages with
fixed weights inside each. The first 6 trillion tokens are web text and at
most 10% code, with no mathematics at all because the mathematics corpora
were too small to last; later stages add mathematics and educational code;
and the decay stage holds the highest-quality sets. Neither report claims
that its proportions are optimal for the final model: Olmo 3's are what its
proxies preferred under a repetition constraint, and SmolLM2's were revised
between stages from evaluations of the large run itself.

## Transfer Across Scale
:label:`subsec_lm_mix_transfer`

Everything so far was decided by models of 4.6 million parameters. Three
things limit how far such a decision carries. The first is the number of
tokens per parameter: the T1 proxies see about 4 tokens per parameter and the
T2 proxies about 7, against the roughly 20 of compute-optimal training
:cite:`hoffmann2022training` and 100 for the Olmo 3 swarm, and an
under-trained model has learned the easy statistics of every domain and few
of the hard ones. The second is the model itself: a proxy whose parameters
are mostly an embedding table learns which tokens follow which, and a model
that learns long-range structure may value code and mathematics quite
differently. The third is the tokenizer: the weights are shares of sequences
of tokens, and since bytes per token differ by domain, the same weights under
another tokenizer put a different share of text from each domain into the
run. RegMix chose for a model a thousand times larger than its proxies and
checked the choice against 64 other mixtures trained at the large size;
DoReMi transferred over a factor of 30. Without such a check a proxy's
ranking is a hypothesis about the large model, not a measurement of it.

The T2 configuration is the next step of that check, packaged in
`tools/lm_campaigns.yaml`: the same design with 16 Dirichlet training
mixtures and 8 held-out ones, at 20.3 million parameters and $2^{27}$ tokens
per run. The launcher runs it on as many GPUs as it is given, checkpoints
every run, stops itself after a time limit and resumes where it stopped. Once
its run set is registered, this page reads it in place of the T1 runs.

```{.python .input #data-mixtures-t2-package}
%%tab pytorch
c2 = d2l.MIX_CONFIGS['t2']
say('T2 campaign:', f'python tools/run_mix_campaign.py --config t2 '
    '--gpus 0,1,2,3 --out ../data/lm/rerun/mix-t2 (a faster node)')
CAMPAIGNS = yaml.safe_load((ROOT / 'tools' / 'lm_campaigns.yaml').read_text())
for e in CAMPAIGNS['campaigns']:  # tools/lm_campaigns.yaml
    say(f"{e['id']}:")
    for k in ('status', 'config', 'command', 'runs', 'flop',
              'predicted_a100_hours', 'memory_gib_per_run', 'measured'):
        if e.get(k) is not None:
            say(f'  {k}: {e[k]}')
n2 = len(d2l.mix_run_specs(c2))
if SMOKE is not None:
    say(f'from the smoke on {SMOKE["device"]}: {SMOKE["campaign_flop"]:.2g} '
        f'FLOP for {n2} runs; A100-hours at the measured MFU '
        f'{two(SMOKE["predicted_a100_hours"])}, peak memory per run '
        f'{two(SMOKE["peak_mem_gib"])} GiB')
say('writes:', ', '.join([c2['runs_key']] + sorted(set(c2['keys'].values()))
                         + [c2['code_meta']]))
```

A first reading of transfer is already possible. The T2 design was run at
half its tokens, $2^{26}$ per run, on four RTX 4090s, which took 46 minutes;
it is a test of the T2 code path and a first look at a larger proxy, not the
T2 result. The 22 mixtures that both scales trained, the pure-domain runs and
the shared design draws, can be ranked at both sizes, and the regression, the
choice and the arms can be repeated at the larger one.

```{.python .input #data-mixtures-t2-test}
%%tab pytorch
TEST_KEY = 'lm-mix-runs-t2s-v1'  # four RTX 4090s, 2026-11-21
TEST_TAG = '[small-scale T2 test, 2^26 tokens per run: not the T2 result]'
with open(d2l.lm_artifact(TEST_KEY)) as f:
    TEST = json.load(f)
TR = {r['run_id']: r for r in TEST['runs']}
assert TEST['mix_config'] == json.loads(json.dumps(dict(  # t2, half tokens
    c2, name='t2s', runs_key=TEST_KEY, tokens=2**26, lr_tokens=2**24,
    pools_of='t2')))
fin = lambda r: np.array([TR[r]['final'][d]['bpb'] for d in DOMAINS])
shared = [r for r in REC if r in TR and REC[r]['arm'] in (
    'pure', 'train', 'held')]  # T1's design draws are a prefix of T2's
assert all(REC[r]['weights'] == TR[r]['weights'] for r in shared)
t1_t = np.array([T_DEP @ final(r) for r in shared])
t2_t = np.array([T_DEP @ fin(r) for r in shared])
top = lambda v: ', '.join(shared[k] for k in np.argsort(v)[:3])
say(TEST_TAG, f'{N["t2"] / 1e6:.1f}M-parameter proxies; peak learning '
    f'rate {TEST["lr_choice"]["chosen"]:g} (T1: '
    f'{RUNS["lr_choice"]["chosen"]:g})')
say(f'  t_dep bits per byte of the {len(shared)} mixtures both scales '
    f'trained: Spearman {stats.spearmanr(t1_t, t2_t)[0]:.2f}')
say(f'  best three at T1: {top(t1_t)}; in the test: {top(t2_t)}')
ids = {a: [r['run_id'] for r in TEST['runs'] if r['arm'] == a]
       for a in ('train', 'held')}
Wt2, Wh2 = (np.array([TR[r]['weights'] for r in ids[a]])
            for a in ('train', 'held'))
Lt2, Lh2 = (np.array([fin(r) for r in ids[a]]) for a in ('train', 'held'))
f2 = d2l.fit_mixture_regression(Wt2, Lt2, c2['select'])
cos2 = np.array([fin(f'uniform-cos-s{s}') for s in c2['cos_seeds']])
floor2 = math.sqrt((cos2.std(0, ddof=1)**2).mean() * (1 + 3 / len(Wt2)))
rmse2 = np.sqrt(((f2(Wh2) - Lh2)**2).mean(0))
fmt = lambda w: '(' + ', '.join(f'{x:.3f}' for x in w) + ')'
say(f'  {c2["select"]} held-out RMSE per domain {rmse2.round(4).tolist()}'
    f' against a floor of {floor2:.4f}')
say(f'  selected for t_dep {fmt(TEST["selection"]["weights"])} (T1 '
    f'{fmt(SEL["weights"])}); DoReMi {fmt(TEST["doremi"]["weights"])} '
    f'(T1 {fmt(DOREMI_W)})')
wsd2 = {a: np.array([fin(f'{"uniform-wsd" if a == "uniform" else a}-s{s}')
                     for s in SEEDS]) for a in ARMS}
T2T = {a: wsd2[a] @ T_DEP for a in ARMS}
sig2 = T2T['uniform'].std(ddof=1)
for a in ARMS[1:]:
    d = T2T[a] - T2T['uniform']
    real = len(set(np.sign(d))) == 1 and abs(d.mean()) > 2 * sig2
    say(f'  {a}: t_dep minus uniform {np.round(d, 4).tolist()} (2 '
        f'sigma_seed {2 * sig2:.4f}), ' + ('beyond the noise' if real
                                           else 'within the noise')
        + f'; mean per domain {wsd2[a].mean(0).round(4).tolist()}')
say(f'  uniform per domain {wsd2["uniform"].mean(0).round(4).tolist()};'
    ' cosine schedule ' + str(cos2.mean(0).round(4).tolist()))
```

The rankings agree only in part. Over the 22 shared mixtures the rank
correlation of the `t_dep` losses between T1 and the half-length T2 test is
0.70, and only one of the three best mixtures at T1 scale is among the three
best in the test. The learning-rate choice moves from $2 \times 10^{-3}$ to $10^{-3}$, as
expected for a wider model. The mixing law still misses its floor of about 0.02,
by about 0.07 on code and 0.05 on mathematics. Its choice for `t_dep` moves toward web text,
$(0.44, 0.27, 0.30)$, a shift of about 0.1 in the web weight, about the width
of the T1 bootstrap range, and here the regression's arm beats uniform beyond
the noise, by about 0.01 bits per byte in all three seeds. DoReMi's weights
move most, to $(0.29, 0.30, 0.41)$, and its arm, like the staged one, is
within the noise of uniform. So the small proxies located the neighbourhood
of a good mixture and did not fix the point: a model four times larger
prefers more web text, and at that size its own choice beats uniform. Whether the trend continues to a model a hundred times larger, the
size at which mixtures are actually chosen, is a question that only a run at
that size can answer.
<!-- campaign: lm-mix-runs-t2-v1 -->

## Summary

A data mixture is a hyperparameter whose every trial costs a pretraining
run, so it is chosen with small proxy models: train proxies on random
mixtures, fit a function from weights to per-domain bits per byte, and
minimize a target loss fixed in advance. Bits per byte, not bits per token,
make the domains comparable, and the noise floor $\sigma\sqrt{1 + p/n}$ says
how well any fitted function can predict a new run. At T1 scale, early
rankings of mixtures predicted late ones, no regression reached the floor of
about 0.02 bits per byte, the exponential mixing law failed on code, and its
choice for a web-heavy target was close to uniform because mathematics pages
teach web prose almost as well as web pages do; DoReMi's worst-case objective
moved weight to code and lost slightly on that target. In a model four times
larger the ranking of mixtures agreed only in part (rank correlation 0.7),
and the chosen weights moved toward web text and beat uniform, so proxies
locate a region of good mixtures rather than a point. Temperature sampling
and annealing change the mixture over sources and over time; the first is
limited by the value of repeated data, which saturates after a few epochs,
and the second by the question of whether its gains are capability or
recency.
<!-- campaign: lm-mix-runs-t2-v1 -->

## Exercises

1. **Repetition in a budget.** A run of 100 billion tokens gives weight 0.2
   to a source of 2 billion unique tokens.
    1. How many epochs of the source does the run read? Using
       :eqref:`eq_lm_mix_effective`, how many effective tokens does the
       source contribute, and what share of its 20 billion nominal tokens is
       that? Confirm with `d2l.effective_tokens`.
    1. Find the weight above which the source's nominal tokens overstate its
       effective tokens by more than 10%. How does the answer change with the
       size of the run?
    1. The law has a single fitted constant. Name one property of a source,
       for example of a code corpus against a web crawl, that you would
       expect to change $R^*$, and say in which direction.
   :exlabel:`ex_lm_mix-repetition`

1. **The temperature for a target share.** Three languages have $n_1 = 1000$,
   $n_2 = 100$ and $n_3 = 10$ documents.
    1. For two languages, derive in closed form the $\alpha$ that gives the
       smaller one a share $s$ of the samples. When does no
       $\alpha \in [0, 1]$ work?
    1. For the three languages, find numerically the $\alpha$ that gives the
       smallest a share of 10%, and the number of epochs it is then read at
       $D = 10^5$. Show that the share of the smallest language is monotone
       in $\alpha$, so the answer is unique.
   :exlabel:`ex_lm_mix-temperature`

1. **What DoReMi's step does.** Use the update :eqref:`eq_lm_mix_doremi`
   before smoothing.
    1. Show that $\alpha_{t+1,i} \geq \alpha_{t,i}$ exactly when
       $g_{t,i} \geq \frac{1}{\eta} \log \sum_j \alpha_{t,j} e^{\eta g_{t,j}}$,
       and that this threshold lies above the weighted mean
       $\sum_j \alpha_{t,j} g_{t,j}$ by Jensen's inequality. Conclude that a
       domain whose excess loss is at or below the weighted mean never gains
       weight.
    1. Which weights are fixed points of the update? Why is the average of
       the iterates, rather than the last iterate, the natural output of the
       game?
    1. Because the excess loss is clipped at zero per token, a domain on
       which the proxy is better than the reference on average can still have
       $g_{t,i} > 0$. Construct a batch with this property and say whether the
       behaviour is desirable.
   :exlabel:`ex_lm_mix-doremi`

1. **Why ridge picks corners.**
    1. Show that a predicted target loss that is linear in $\mathbf{w}$ has
       its minimum over a polytope at a vertex. Which vertex of the hull of
       the training mixtures does ridge choose for `t_dep`, and why that one?
    1. [code] Count, over the 200 bootstrap resamples, how often ridge,
       gradient boosting and the mixing law choose a point within 0.05 of the
       hull boundary. What does the count say about using each of them to
       select?
   :exlabel:`ex_lm_mix-corners`

1. [code] **A law for code.** Plot the measured code loss of the training
   runs against the code weight. Propose a law that follows the steep drop
   near zero weight, for example
   $c + k (w_{\textrm{code}} + \epsilon)^{-\beta}$ for the code domain with the
   other domains' weights as a linear correction, fit it on the same training
   runs, and compare its held-out error per domain with the mixing law's and
   with the floor. Does its choice for `t_dep` move?
   :exlabel:`ex_lm_mix-code-law`

1. [code] **Annealing on mathematics.** Train the staged arm with the decay
   on the fixed mixture $(0.1, 0.1, 0.8)$ instead of the regression's choice,
   three seeds, paired with the uniform arm.
    1. Before running, predict the change of the mathematics and web losses
       during the decay.
    1. Measure both. Then continue each annealed model and its uniform
       partner for another 20% of the tokens on the uniform mixture at a low
       constant learning rate. How much of the mathematics gain survives?
       What does the answer say about recency?
   :exlabel:`ex_lm_mix-anneal`

1. [code] **Does the ranking transfer?** Train the T1 design with the T2
   proxy shape at T1's $2^{24}$ tokens per run on one GPU (about 40 minutes
   on an RTX 4090). Before running, predict whether the rank correlation of
   the `t_dep` losses with T1 will be closer to 1 or to the 0.70 of the
   half-length T2 test, which changes model size and tokens at once. Measure
   it, compare the selected weights at the two sizes, and say which of the
   two changes moves the selection more.
   :exlabel:`ex_lm_mix-transfer`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §20.4]{.kicker}

Data Mixtures<br>**domain weights · temperature and repetition · proxy regression · DoReMi · annealing · transfer**
:::
:::

::: {.slide title="Every Candidate Mixture Costs a Pretraining Run"}
[The problem]{.kicker}

Olmo 3's pretraining mixture, from its dataset card:

| source | share |
|---|---|
| web (Common Crawl) | 76% |
| scientific PDFs | 14% |
| code | 7% |
| mathematics web pages | 2.6% |

Nobody derives these numbers, and nobody can afford to try every candidate
at full scale.

::: {.d2l-note .warn}
Domain weights are hyperparameters. Fit them on small proxy runs.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compare losses across domains in bits per byte;
- compute what temperature sampling repeats and what repeats are worth;
- fit a regression from weights to loss and know its noise floor;
- choose a mixture for a fixed target, and say what DoReMi optimizes instead;
- judge whether a proxy's choice transfers.

**Running example:** 41 runs of a 4.6M-parameter GPT, $2^{24}$ tokens each,
on web text (FineWeb-Edu), Python (CodeParrot, permissive licences) and
mathematics (FineMath).
:::

::: {.slide title="Mixing Comes After Choosing and Before Cutting"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| step | section |
|---|---|
| get text | Document Extraction from Web Crawls |
| choose text | Quality Filtering |
| | Deduplication and Decontamination |
| mix text | **Data Mixtures** |
| | Synthetic and Rephrased Data |
| cut text | Tokenizers at Scale |
:::

::: {.col .narrow}
- **Here:** how often each kind of text is read.
- **Skips:** mid-training stages of released models.
:::
:::
:::

::: {.slide title="Per Token, Code Looks Easiest; per Byte, It Is Hardest"}
[Comparable losses]{.kicker}

The uniform mixture, one proxy:

| | web | code | math |
|---|---|---|---|
| bytes per token | 4.56 | 2.42 | 3.00 |
| bits per token | 8.73 | 5.13 | 6.63 |
| bits per byte | 1.92 | 2.12 | 2.21 |

$$\textrm{bpb} = \frac{\sum \textrm{NLL (nats)}}{\ln 2 \cdot \#\textrm{UTF-8 bytes}}$$

::: {.d2l-note}
Bits per byte do not depend on the tokenizer. Hold fixed: tokenizer, model,
tokens, schedule, evaluation sets, data order.
:::
:::

::: {.slide title="Temperature Sampling Repeats the Smallest Sources Most"}
[Many sources]{.kicker}

$p_i \propto n_i^{\alpha}$; epochs $e_i = p_i D / n_i \propto n_i^{\alpha - 1}$.

Ten languages, $10^{10}$ documents drawn:

| language | documents | share, $\alpha = 1$ | share, $\alpha = 0.3$ | epochs, $\alpha = 0.3$ |
|---|---|---|---|---|
| English | 25.9B | 0.915 | 0.345 | 0.13 |
| Hindi | 22M | 0.0008 | 0.042 | 18.7 |
| Swahili | 1.2M | 0.00004 | 0.017 | 143 |

::: {.d2l-note}
$\alpha = 0.7$ (multilingual BERT), $0.3$ (XLM-R, mT5). Sampling does not
create data; it chooses what is repeated.
:::
:::

::: {.slide title="Repeats Are Almost Free up to Four Epochs, Then Saturate"}
[Repetition]{.kicker}

$U$ unique tokens, $E$ epochs, $R = E - 1$ repeats:

$$D' = U + U R^* \bigl(1 - e^{-R/R^*}\bigr), \qquad R^* \approx 15.4$$

| epochs | $D' / (UE)$ |
|---|---|
| 4 | 0.93 |
| 4.8 | 0.91 |
| $\infty$ | $D' \to 16.4\, U$ |

::: {.d2l-note}
Small $R$: $D' \approx UE$. Large $R$: further epochs add nothing
(Muennighoff et al.'s fit).
:::
:::

::: {.slide title="Check: Swahili Read 143 Times. Worth How Many Fresh Copies?"}
[Check]{.kicker}

Use $D' = U + U R^* (1 - e^{-R/R^*})$ with $R^* = 15.4$ and $R = 142$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $e^{-142/15.4} \approx 10^{-4}$, so $D' \approx 16.4\,U$: about
11% of 143 fresh copies. Most of Swahili's share at $\alpha = 0.3$ is wasted;
UniMax caps the epochs per language instead.
:::
:::
:::

::: {.slide title="RegMix: Sample Mixtures, Train Proxies, Fit, Minimize"}
[The method]{.kicker}

| runs | mixtures | purpose |
|---|---|---|
| 15 training | 6 Dirichlet(1), 6 Dirichlet(0.3), 3 corners | fit |
| 4 held out | Dirichlet(1), another seed | judge the fit |
| 3 pure, 5 uniform | one domain; seeds | matrix; noise |
| 4 × 3 seeds | uniform, selected, DoReMi, staged | arms |

- Peak learning rate chosen once on uniform: $2 \times 10^{-3}$ (interior of
  four values), never per mixture.
- Targets fixed first: $t_{\textrm{avg}} = (\tfrac13, \tfrac13, \tfrac13)$,
  $t_{\textrm{dep}} = (0.7, 0.15, 0.15)$ over (web, code, math).
:::

::: {.slide title="Each Proxy Is Best on Its Own Domain"}
[Why mixing is a choice]{.kicker}

Bits per byte of the pure-domain proxies:

| trained on | web | code | math |
|---|---|---|---|
| web | **2.05** | 5.23 | 3.31 |
| code | 2.57 | **2.02** | 3.06 |
| math | 2.15 | 3.46 | **2.24** |

::: {.d2l-note}
Off-diagonal costs are large and asymmetric. FineMath is English prose about
mathematics, so it teaches web text almost as well as web text does.
:::
:::

::: {.slide title="Early Rankings Predict Late Ones in 105 Pairs"}
[Do curves cross?]{.kicker}

For every pair of the 15 training mixtures, compare the $t_{\textrm{dep}}$
loss at 25% of the run and at its end.

| | pairs |
|---|---|
| all pairs | 105 |
| order swaps | 10 |
| swaps beyond $2\sqrt2$ seed SDs at both points | 0 |

::: {.d2l-note}
At this scale a short run ranks mixtures as a full one does. Whether a small
model ranks them as a large one does is another question.
:::
:::

::: {.slide title="No Regression Can Beat σ√(1 + p/n)"}
[The noise floor]{.kicker}

Least squares, $n$ runs, $p$ parameters, noise $\sigma$, noisy held-out run:

$$\mathbb{E}\bigl[(\hat y - y)^2\bigr] = \sigma^2 \bigl(1 + \mathbf{x}^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}\bigr)$$

Average over the design: $\operatorname{tr}$ of the hat matrix is $p$, so
$\sigma^2 (1 + p/n)$.

- $\sigma$ from extra seeds of the uniform mixture: floor about 0.02 bits per
  byte (3 degrees of freedom: 0.01 to 0.06).
- Simulation: 4.79e-4 against 4.78e-4 predicted.
:::

::: {.slide title="On Held-Out Mixtures, Every Regression Misses the Floor"}
[Three regressions]{.kicker}

Mixing law: $b_i(\mathbf w) = c_i + k_i \exp\bigl(\sum_j t_{ij} w_j\bigr)$.

| held-out RMSE (bpb) | web | code | math |
|---|---|---|---|
| ridge | 0.03 | 0.06 | 0.07 |
| gradient boosting | 0.01 | 0.03 | 0.05 |
| mixing law | 0.003 | 0.12 | 0.04 |
| floor | 0.02 | 0.02 | 0.02 |

::: {.d2l-note .warn}
The law fails on code, whose loss drops steeply near zero code weight. More
runs lower the floor, not the bias.
:::
:::

::: {.slide title="A 70%-Web Target Selects About One Third Web"}
[Choosing]{.kicker}

Minimize the fitted $t_{\textrm{dep}}$ loss inside the hull of the training
mixtures:

| regression | (web, code, math) |
|---|---|
| mixing law (fixed in advance) | (0.34, 0.31, 0.35) |
| ridge | (0.10, 0.80, 0.10), a corner |
| gradient boosting | (0.24, 0.55, 0.22) |

200 bootstrap refits: web 0.25 to 0.38, code 0.24 to 0.43.

::: {.d2l-note}
Web loss barely moves with web weight; code and math fall with theirs. The
runs fix a neighbourhood, not a point.
:::
:::

::: {.slide title="Check: Why Does Ridge Always Choose a Corner?"}
[Check]{.kicker}

Ridge predicts each $b_i$ as linear in $\mathbf w$. The choice minimizes
$\sum_i t_i b_i(\mathbf w)$ over the hull of the training mixtures.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A sum of linear functions is linear, and a linear function on a
polytope is minimized at a vertex. Ridge can never select an interior
mixture; the mixing law's exponentials can.
:::
:::
:::

::: {.slide title="DoReMi Maximizes the Worst Excess Loss, Not Your Target"}
[A different objective]{.kicker}

$\min_\theta \max_{\boldsymbol\alpha \in \Delta} \sum_i \alpha_i (\ell_i(\theta) - \ell_{\textrm{ref},i})$;
excess $g_i$ clipped at 0 per token.

$$\alpha_{t+1,i} = \frac{\alpha_{t,i}\, e^{\eta g_{t,i}}}{\sum_j \alpha_{t,j}\, e^{\eta g_{t,j}}}$$

| | web | code | math |
|---|---|---|---|
| weight at step 65 | 0.03 | 0.75 | 0.22 |
| weight at step 129 | 0.15 | 0.25 | 0.60 |
| average over 256 steps | 0.17 | 0.47 | 0.36 |
:::

::: {.slide title="Four Arms: One Method, Two Targets, Two Verdicts"}
[Arms]{.kicker}

Three seeds each, paired with uniform; real only if all three differences
share a sign and the mean exceeds $2\sigma_{\textrm{seed}}$ (about 0.01).

| arm minus uniform | $t_{\textrm{dep}}$ | $t_{\textrm{avg}}$ | code |
|---|---|---|---|
| regression's choice | about −0.007 | about −0.008 | 0.00 |
| DoReMi | about +0.015 | about −0.025 | −0.10 |
| staged | about 0 | about 0 | 0.00 |

::: {.d2l-note}
DoReMi buys code with web text: worse on the web-heavy target, at the edge
of the noise; better on the average.
:::
:::

::: {.slide title="Annealing Reads the Best Data While the Learning Rate Falls"}
[Annealing]{.kicker}

- Stable-then-decay: peak rate for 80% of the steps, decay in the last 20%,
  where most of the loss drop happens.
- SmolLM2's last 10% of 11T tokens: 14% math, 24% code, held back until then.
- Our staged arm decays on the selected mixture, which is nearly uniform:
  no measurable change.

::: {.d2l-note .warn}
Benchmark gains may be recency. Check held-out loss of the domain, then train
on and see whether the gain survives.
:::
:::

::: {.slide title="Olmo 3 Runs the Same Method With Larger Proxies"}
[A released mixture]{.kicker}

| source | pool (tokens) | mix (tokens) | epochs |
|---|---|---|---|
| Common Crawl | 8.14T | 4.51T | 0.55 |
| code (Stack-Edu) | 137B | 409B | 3.0 |
| arXiv | 21.4B | 50.9B | 2.4 |
| FineMath 3+ | 34.1B | 151B | 4.4 |

- Weights from a swarm of 30M-parameter proxies, 3B tokens each, Dirichlet
  mixtures, one regression per task.
- Then 100B tokens of mid-training on the best data plus task data.
:::

::: {.slide title="Check: Nine Tenths of the Proxy Is Its Embedding Table"}
[Check]{.kicker}

4.2M of the 4.6M parameters are embeddings; it sees about 4 tokens per
parameter, against about 20 for compute-optimal training. What can its
ranking of mixtures tell you about a 1B model?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A hypothesis, not a measurement. It has learned which tokens
follow which, not long-range structure; a larger model may value code and
math differently. RegMix checked its choice against 64 mixtures trained at
the large size.
:::
:::
:::

::: {.slide title="Four Times Larger: Rank Correlation 0.70, the Choice Moves to Web"}
[Transfer]{.kicker}

A half-length run of the larger design: 20.3M parameters, $2^{26}$ tokens,
46 minutes on four RTX 4090s.

| | 4.6M proxy | 20.3M proxy |
|---|---|---|
| peak learning rate | $2 \times 10^{-3}$ | $10^{-3}$ |
| selected for $t_{\textrm{dep}}$ | (0.34, 0.31, 0.35) | (0.44, 0.27, 0.30) |
| selected minus uniform | within noise | about −0.01, all 3 seeds |
| DoReMi weights | (0.17, 0.47, 0.36) | (0.29, 0.30, 0.41) |

Rank correlation of 22 shared mixtures: 0.70.
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Compare domains in bits per byte; code is 5.1 bits per token but 2.1 per
  byte.
- Temperature $\alpha = 0.3$ reads Swahili 143 times, worth 11% of fresh data.
- Early rankings predict late ones: 0 of 105 pairs cross beyond the noise.
- Floor about 0.02 bpb; best regression 0.03, mixing law 0.07 (code).
- Web-heavy target selects about one third web; DoReMi moves weight to code.
- At 4× the size: rank correlation 0.70, the choice moves to web and wins.

**Next:** making unique tokens for domains that are too small.
:::
