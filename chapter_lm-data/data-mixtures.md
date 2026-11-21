# Data Mixtures
:label:`sec_lm_data_mixtures`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.4 Data Mixtures · T2").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.4.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Domain weights are hyperparameters, fitted on proxy runs.
**Opening:** Two small runs with different weights on web, code and mathematics have loss curves that cross: the mixture that wins early loses by the end. The reader cannot afford to try every mixture at full scale. The section treats the weights as hyperparameters, fits a model from weights to per-domain loss on small runs, and closes on whether what the small runs decide transfers to a larger model.
**Teaches:**
- Domain weights, temperature sampling across languages and domains (`d2l.temperature_weights`), and the effective number of tokens under repetition (`d2l.effective_tokens`), so a pool is always larger than the run needs.
- Proxy-model regression from sampled weights to per-domain bits per byte (`d2l.fit_mixture_regression`, ridge and an exponential mixing law), selection of a mixture for a target distribution; DoReMi's one difference (a worst-case excess-loss objective over domains, `d2l.doremi_weights`).
- Annealing on a higher-quality mixture at the end of training under a stable-then-decay schedule, and why its benchmark gains can be recency rather than capability.
- Dolma 3 as the open, documented mixture of a released model, read as a case rather than copied.
**Derivation:** none.
**Notebook:**
1. Build token pools for web (FineWeb-Edu side buckets), code (permissively licensed Python files) and mathematics (FineMath) with the reference tokenizer, each larger than the largest weight times the run budget, plus disjoint validation sets per domain.
2. Train proxy `d2l.GPT` models of about 20 million parameters at a fixed token budget on 16 training mixtures drawn from a Dirichlet, 8 held-out mixtures, and the uniform mixture with three data-order seeds (the noise floor); peak learning rate chosen once on the uniform mixture and fixed.
3. Fit the regression on the 16 runs and predict the 8 held-out mixtures' per-domain bits per byte; select the mixture that minimizes a stated target-weighted loss and train it with three seeds; train DoReMi's weights and a staged (annealed) arm with three seeds each.
4. Plot predicted against measured bits per byte per domain with the seed-noise band; a ternary plot of all runs; the arms as three seed points each against the uniform mean.
5. Claim that survives noise: the regression's predictions fall within the proxies' seed noise of the measured losses on the held-out mixtures (or the text reports that the regression only reaches the noise floor, which is itself the finding); an arm beats uniform only if all three seed-paired differences share a sign and the mean exceeds two floor standard deviations; Spearman rank correlation over 8 held-out mixtures is reported without a significance claim.
6. Reused pieces: `d2l.GPT`, `sec_scheduler`'s warmup plus cosine and the stable-then-decay shape, `sec_adamw`, `d2l.bits_per_byte` (1.4, imported), `d2l.paired_bootstrap` (1.5); the reference tokenizer of 3.6 (loaded, forward-referenced); new `d2l.ProxyCfg`, `d2l.train_proxy_lm`, `d2l.bpb_from_nll`, `d2l.sample_mixtures`, `d2l.temperature_weights`, `d2l.fit_mixture_regression`, `d2l.doremi_weights`, `d2l.effective_tokens`.
**Figures:** `fig_lm_mixture_regression` (plot: predicted against measured bits per byte on held-out mixtures per domain with the identity line and the seed band; a second panel with the ternary scatter and the selected mixture).
**Restate / forward:** `sec_scaling-laws` (loss follows a power law in parameters and tokens, so small proxies are informative about larger runs, and repeated data counts for less than fresh data). Forward: 4.4 (the mid-training stage where the mixture changes).
**Exercises:** 5 to 7. Load-bearing: derive the temperature that gives a target share to the smallest of three languages given their sizes (derive); measure the selected weights against proxy size (two proxy sizes) and report whether the ranking of mixtures transfers (measure); check that `d2l.effective_tokens` reproduces the repeated-data law's value at four epochs (check); open: replace the ridge regression by the exponential mixing law and compare held-out error at equal runs.
**Writer notes:**
- Proxy size: about 20 million parameters (the blueprint's 8.1M at T2 and 4.6M at T1 are the fallback if the budget cap binds). Sizing: at 2^27 tokens per run the 36 T2 runs cost about 6 × 2·10^7 × 1.3·10^8 × 36 FLOP, which at 15% utilization of an A100's dense bf16 peak is above the 3-hour cap; either cut to 12 training mixtures and 2^26 tokens, or `torch.compile` and larger micro-batches, or keep the smaller proxy. Tiny models are launch-bound, not FLOP-bound, so measure throughput before committing to the grid. T1 configuration: the same code with 20M-token pools, a 2^24-token budget, 12 + 4 + 9 runs, all under 15 minutes and 15 GiB.
- Exports with contracts (`01_interfaces.md`): `d2l.train_proxy_lm(pools, weights, tokens, cfg, seed=0, eval_sets=None, stages=None, log_every=50)` whose data stream depends only on `(seed, weights, tokens)`, returning per-domain nll and bytes (so losses convert with `d2l.bpb_from_nll(sum_nll_nats, n_bytes)`); it does not redefine `d2l.bits_per_byte`; `d2l.effective_tokens(unique, epochs, r_star=15.4)` is the data-constrained scaling law's form.
- Pools: flat little-endian `uint16` files with a `meta.json` (token count, NFC bytes, document count, tokenizer hash, source, revision, bucket rule); web from FineWeb-Edu buckets 1 to 7 with bucket 8 as validation; code from `codeparrot/codeparrot-clean` filtered to permissive per-file licences (the pool shrinks; the gated Stack is the production reference and cannot be fetched by a notebook); mathematics from `HuggingFaceTB/finemath` (ODC-By). FineMath's `token_count` is under another tokenizer; use reference-tokenizer counts written at build time.
- Reference and tolerance: the regression is judged only on held-out mixtures against the seed floor from three uniform seeds; arms compared seed-paired with the same data-order seed; with n = 3 no bootstrap interval is reported for arm comparisons, and the text says so.
- Pitfalls to state: comparing per-token losses across domains with different bytes per token; reading training-mixture R² as success; tuning the learning rate per mixture; choosing target weights after seeing results; a pool smaller than the run needs repeats tokens silently; sampling mixtures only from the interior of the simplex (Dirichlet with concentration 0.3 reaches the edges); extrapolating a proxy ranking to a model 100 times larger without a transfer check; annealing's gains as recency.
- Dolma 3 is read from the Olmo 3 report and the dataset card (`allenai/dolma3_pool`, ODC-By): the domains and their weights, the mid-training mix; no numbers in prose beyond what the card states.
- Forward reference to 3.6 (the reference tokenizer) is deliberate; 3.6's notebook reproduces the tokenizer hash. If the editor prefers no forward reference, the pools can be built with the GPT-2 tokenizer at the cost of a larger embedding, changing only the pool artifacts.
- Committed artifacts (T2): `lm-mix-runs-t2` (all run logs and per-domain results), the fitted regressors, the selected mixtures, the pools `lm-mix-{web,code,math}-{20m,150m}` and `lm-mix-val-{1m,2m}`.
- Do not re-teach: scaling-law fitting (4.6 and `sec_scaling-laws`), the stable-then-decay schedule (`sec_scheduler`).
**Sources:**
- Papers: [DoReMi](https://arxiv.org/abs/2305.10429) (group distributionally robust optimization of domain weights and their transfer); [RegMix](https://arxiv.org/abs/2407.01492) (regression over many small proxy runs; read its transfer claim and cite its scale); [Data Mixing Laws](https://arxiv.org/abs/2403.16952) (exponential mixing laws with scale extrapolation); [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (repetition, effective data, the four-epoch regime); [Olmo 3](https://arxiv.org/abs/2512.13961) (Dolma 3 and the staged mixture of a fully open model); [SmolLM2](https://arxiv.org/abs/2502.02737) (staged mixtures with web, code and mathematics at small scale).
- Code and documentation: [RegMix](https://github.com/sail-sg/regmix) and [DoReMi](https://github.com/sangmichaelxie/doremi) (the reference implementations); [Dolma 3 pool](https://huggingface.co/datasets/allenai/dolma3_pool) (the dataset card and its domain composition); [OLMo](https://github.com/allenai/OLMo) (the training code that consumed it).
- Courses, tutorials and blogs: [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (experiment planning and the mixture discussion); [CS336](https://cs336.stanford.edu/) (data mixing lecture and assignment A3's scaling-law fitting as the sibling exercise); [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (ablation discipline).
-->

<!-- Prose (stage 2): the opening. Two small runs with different weights on
web, code and mathematics whose loss curves cross (the plot comes from the
registered run set below); the reader cannot try every mixture at full
scale; the section fits a model from weights to per-domain loss on small
runs and closes on whether what they decide transfers. Every T1 number in
the prose carries "at T1 scale" and the HTML comment mark
"campaign: lm-mix-runs-t2-v1", so the post-campaign step finds it. -->

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
import time
import pandas as pd
from scipy import optimize, spatial, stats
```

<!-- Prose: the three domains, the two target distributions (stated as
illustrative and fixed before any run), and what a regeneration does. -->

```{.python .input #data-mixtures-config}
%%tab pytorch
#@save
MIX_DOMAINS = ('web', 'code', 'math')

#@save
MIX_TARGETS = {  # target weights over MIX_DOMAINS, fixed before any run
    't_avg': (1 / 3, 1 / 3, 1 / 3), 't_dep': (0.7, 0.15, 0.15)}

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
print(f'configuration {CONFIG}' + ('' if T2 else ' (the T2 campaign logs are '
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

# lm-mix-runs-t2-v1 is added after the T2 campaign
MIX_KEYS = ('lm-mix-web-20m-v1', 'lm-mix-math-20m-v1', 'lm-mix-val-1m-v1',
            'lm-mix-code-meta-v1', 'lm-mix-runs-t1-v1', 'lm-mix-web-150m-v1',
            'lm-mix-math-150m-v1', 'lm-mix-val-2m-v1',
            'lm-mix-code150-meta-v1', 'lm-mix-runs-t2-v1')
UPSTREAM = ('lm-r2-tokenizer-v1',)
for name, keys in (('this section', MIX_KEYS), ('upstream', UPSTREAM)):
    print(f'{name}: registered', ', '.join(
        k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
          ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')
```

## Domain Weights Are Hyperparameters
:label:`subsec_lm_mix_weights`

<!-- Prose: a mixture is a point on the simplex; each sequence draws its
domain with probability w; the two objectives (average held-out bits per
byte over domains, and the target-weighted sum with t_avg or t_dep); losses
in bits per byte (`bpb_from_nll`; one sentence restating
subsec_bits-per-byte), never per token; what to hold fixed when a gain is
credited to the mixture (tokenizer, model, tokens, schedule, evaluation
sets, data-order seed). The two tiers of the study follow. -->

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

<!-- Prose: the web pool comes from FineWeb-Edu's side buckets; the bucket
rule (one sentence: the bucket depends only on the document's id). -->

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

<!-- Prose: the three pools (web: FineWeb-Edu buckets 1-7, validation bucket
8; code: permissively licensed CodeParrot files, built locally and not
hosted; mathematics: FineMath 4+), each tokenized with the reference
tokenizer of sec_lm_tokenizers (forward reference), each larger
than the largest weight times the run. The table: tokens, documents, bytes
per token (C2: they differ by domain), tokenizer hash. One sentence: code is
split by repo/path and FineWeb-Edu by document id, so forks and
near-duplicates can fall on both sides of the pool/validation split
(CodeParrot-clean is deduplicated exactly, not fuzzily); at this model size
and one epoch the effect is negligible. -->

```{.python .input #data-mixtures-pools}
%%tab pytorch
C = d2l.MIX_CONFIGS[CONFIG]
POOLS, VALS = d2l.mix_pools(C)
for c in d2l.MIX_CONFIGS.values():  # a pure-domain run reads all its tokens
    assert c['pool_tokens'] >= c['tokens']
meta = {d: (POOLS[d].meta, VALS[d].meta) for d in POOLS if d in VALS}
if 'code' not in meta:  # CodeParrot's files are not hosted (their licences)
    print('code pool not built here: python tools/build_mix_pools.py '
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
print(POOL_TABLE.round(3).to_string(index=False))
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
print('bucket rule: the first 1,000 web documents of the pool lie in '
      'buckets 1-7, those of the validation set (up to 1,000) in bucket 8')
```

### Temperature Sampling and Repetition
:label:`subsec_lm_mix_temperature`

<!-- Prose: p_i proportional to n_i^alpha (alpha = 1 proportional, 0
uniform; multilingual practice near 0.3); epochs per source
e_i = p_i D / n_i (eq_lm_mix_epochs, three lines: the smallest
sources repeat most for alpha < 1); the repeated-data law
D' = U + U R* (1 - exp(-R/R*)) (restate sec_scaling-laws). The
three library functions: -->

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

<!-- Prose: the D1 check on three toy sources, then ten languages: the nine
FineWeb-2 languages of 20.6 (document counts read 2026-10-28) and English
from FineWeb, the English dataset FineWeb-2 extends (read 2026-11-16). -->

```{.python .input #data-mixtures-temperature}
%%tab pytorch
n_toy, D_toy = np.array([1000, 100, 10]), 100_000  # D1 on three toy sources
p = d2l.temperature_weights(n_toy, 0.3)
e = p * D_toy / n_toy
counts = np.random.default_rng(0).multinomial(D_toy, p)
se = np.sqrt(D_toy * p * (1 - p)) / n_toy
assert abs(p.sum() - 1) < 1e-12 and (np.abs(counts / n_toy - e) < 3 * se).all()
assert e[2] > e[1] > e[0]
print('D1 (alpha 0.3, sizes 1000/100/10): p =', p.round(4), '| epochs',
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
print(TEMP.to_string(float_format=lambda v: f'{v:.3g}'))
print('read more than four times (the repeated-data limit below):',
      {c.split()[-1][:-1]: list(TEMP.index[TEMP[c] > 4])
       for c in TEMP.columns if c.startswith('epochs')})
alphas = np.linspace(0, 1, 21)
d2l.plot(alphas, [[d2l.temperature_weights(FW_ROWS, a)[lang] for a in alphas]
                  for lang in FW_ROWS], 'alpha', 'sampling probability',
         yscale='log', figsize=(5, 3.5))
legend = d2l.plt.gca().legend(list(FW_ROWS), loc='center left',
                              bbox_to_anchor=(1, 0.5))
```

<!-- Prose: four epochs keep about 93% of their nominal value under the fit;
the saturation value; where U E overstates D' by 10%. -->

```{.python .input #data-mixtures-repetition}
%%tab pytorch
U = 1e9
four = float(d2l.effective_tokens(U, 4)) / (4 * U)
limit = float(d2l.effective_tokens(U, 1001)) / (U * (1 + 15.4))
E_10 = optimize.brentq(lambda E: E * U / d2l.effective_tokens(U, E) - 1.1,
                       1.01, 50)
assert abs(four - 0.932) < 0.005 and abs(limit - 1) < 1e-3
print(f"D'(4 epochs) / (4U) = {four:.3f}; D'(R = 1000) / (U (1 + R*)) = "
      f'{limit:.6f}; U E overstates D\' by 10% at E = {E_10:.2f} '
      f'(R = {E_10 - 1:.2f})')
```

## Proxy Runs and Regression
:label:`subsec_lm_mix_regmix`

<!-- Prose: RegMix's recipe (sample many mixtures, train small models, fit a
regression from mixture to loss, optimize it, train the optimum); the proxy
and its training loop. Losses go out as summed NLL and bytes per domain, so
they convert to bits per byte with `bpb_from_nll`. The stream: windows of
seq_len + 1 tokens, epochs as fresh permutations, a stratified domain
allocation, stages that switch weights; the pool-smaller-than-the-run
guard. -->

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

<!-- Prose: the checks below run on the CPU in fp32 with a 2-block, d = 32
model: the stream is a function of (seed, weights, tokens); the stratified
allocation holds every domain within one sequence of its share; a model
that predicts every token with probability 1/32,768 scores 15 bits per
token, so 15 / (bytes per token) bits per byte; and the parameter counts of
the two tiers against their arithmetic. -->

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
print(f'same (seed, weights, tokens): the same 10 batches and the same loss '
      f'at step 20 ({runs[0]["log"][19][2]:.6f}); web share after 10^4 '
      f'sequences {web[-1] / 1e4:.4f}, never more than {lag:.2f} sequences '
      'from 0.7 n')
uni = d2l._proxy_model(tiny._replace(dtype='float64')).double()
nn.init.zeros_(uni.token_emb.weight)  # tied output: every logit is 0
BYTE_LEN = d2l.token_byte_lengths(d2l.r2_tokenizer())
r = d2l._proxy_eval(uni, VALS['web'].ids, tiny, BYTE_LEN, 32,
                    torch.device('cpu'))
assert abs(r['bpb'] / (15 * r['n_tokens'] / r['n_bytes']) - 1) < 1e-9
print(f'uniform model: {r["bpb"]:.6f} bits per byte = 15 / '
      f'{r["n_bytes"] / r["n_tokens"]:.4f} bytes per token')
with torch.device('meta'):
    N = {k: d2l.num_params(d2l._proxy_model(c['proxy']))
         for k, c in d2l.MIX_CONFIGS.items()}
for k, c in d2l.MIX_CONFIGS.items():
    p = c['proxy']
    arith = p.vocab_size * p.num_hiddens + 12 * p.num_blks * p.num_hiddens**2
    assert abs(N[k] / arith - 1) < 0.005
    flop = d2l.proxy_flops(p, c['tokens'], N[k])
    print(f'{k}: {N[k]:,} parameters (arithmetic {arith:,}; embedding '
          f'{p.vocab_size * p.num_hiddens:,}); {flop:.2g} FLOP per run')
TIER = (f'{CONFIG.upper()}: {N[CONFIG] / 1e6:.1f}M-parameter proxies, '
        f'2^{round(math.log2(C["tokens"]))} tokens per run')
print('this page reads', TIER)
```

<!-- Prose: the run design. Training mixtures from Dirichlet(1) and
Dirichlet(0.3) (the second reaches the edges of the simplex) and three
fixed mixtures near the corners, (0.8, 0.1, 0.1) and its permutations
(without them no training mixture puts more than two thirds on web, and
t_dep's region would be reached only by extrapolation), held-out
mixtures from Dirichlet(1) with another seed, three pure-domain runs,
uniform seeds (two on the cosine schedule: the regression's noise floor;
three on WSD: the arms' floor), the learning rate chosen once on the
uniform mixture. -->

```{.python .input #data-mixtures-design}
%%tab pytorch
TRAIN_W = np.vstack([d2l.sample_mixtures(n, DOMAINS, conc,
                                         C['design_seeds'][0], C['max_weight'])
                     for conc, n in C['train']] + [np.array(C['edges'])])
HELD_W = d2l.sample_mixtures(C['held'], DOMAINS, 1.0, C['design_seeds'][1],
                             C['max_weight'])
print(TAG, f'{len(TRAIN_W)} training mixtures (Dirichlet(1), Dirichlet(0.3), '
      f'{len(C["edges"])} near the corners) and {len(HELD_W)} held-out ones '
      f'(Dirichlet(1)), weights {DOMAINS}:')
print(pd.DataFrame(np.vstack([TRAIN_W, HELD_W]), columns=DOMAINS,
                   index=[f'train {k}' for k in range(len(TRAIN_W))]
                   + [f'held {k}' for k in range(len(HELD_W))]).round(3).T
      .to_string())
SPECS = d2l.mix_run_specs(C)
for arm, W in (('train', TRAIN_W), ('held', HELD_W)):  # the launcher's design
    assert np.array_equal([s['weights'] for s in SPECS if s['arm'] == arm], W)
arms = collections.Counter(s['arm'] for s in SPECS)
print(f'{len(SPECS)} runs: ' + ', '.join(f'{n} {a}' for a, n in arms.items()))
```

<!-- Prose: one regeneration writes the whole run set: the launcher
tools/run_mix_campaign.py, which the T2 campaign uses as well, runs the
learning-rate choice, the runs, the regression's choice of the selected
mixture and DoReMi's weights, and, on the 4090, a short smoke run of the T2
configuration. A normal build reads the registered result. -->

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
    print(f'mix-t1: {(time.time() - t0) / 60:.1f} min wall clock')
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
print(TAG, f'{len(REC)} runs from {os.path.basename(RUNS_PATH)}; peak '
      f'learning rate {RUNS["lr_choice"]["chosen"]:g} (chosen on the uniform '
      'mixture: ' + ', '.join(f'{lr} -> {v:.4f} bpb' for lr, v in
                               RUNS['lr_choice']['mean_bpb'].items()) + ')')
```

<!-- Prose: the live check retrains the start of the pure-web run on this
machine and compares its losses with the registered log. -->

```{.python .input #data-mixtures-live-check}
%%tab pytorch
if GPU and not RERUN:  # the regen cell checked the pool against the runs
    cfg = C['proxy']._replace(lr=REC['pure-web']['lr'])
    live = d2l.train_proxy_lm({'web': POOLS['web']}, {'web': 1.0},
                              C['tokens'], cfg, seed=0, log_every=10,
                              stop_tokens=2**20)
    logged = {s: l for s, _, l, _ in REC['pure-web']['log']}
    diff = max(abs(l - logged[s]) for s, _, l, _ in live['log'] if s % 10 == 0)
    print(TAG, f'pure-web, steps 10-{live["log"][-1][0]}: largest difference '
          f'from the registered losses {diff:.4f} nats (bf16; tolerance 0.02)')
    assert diff <= 0.02
else:
    print('live check skipped:', 'no GPU' if not GPU else
          'the regeneration above trained the same run')
```

<!-- Prose: what the smoke of the T2 configuration measured on the 4090
(throughput with and without torch.compile, MFU against the card's dense
bf16 peak, peak memory, the kill-and-resume check), and the A100 hours it
predicts for the T2 campaign. -->

```{.python .input #data-mixtures-smoke}
%%tab pytorch
SMOKE = RUNS.get('smoke')
two = lambda v: (v if not isinstance(v, float) else f'{v:.1e}' if 0 < abs(
    v) < 1e-3 else f'{float(f"{v:.2g}"):,g}')  # two significant figures
if SMOKE is None:
    print('no smoke record in this run set')
else:
    print(f'smoke of the {SMOKE["config"]} configuration on {SMOKE["device"]} '
          f'({SMOKE["steps"]} steps of {SMOKE["tokens_per_step"]:,} tokens):')
    print(pd.Series({k: two(SMOKE[k]) for k in (
        'n_params', 'tokens_per_s', 'tokens_per_s_compiled', 'mfu',
        'mfu_compiled', 'compile_seconds', 'peak_mem_gib',
        'resume_bf16_max_diff', 'rerun_bf16_max_diff', 'resume_fp32_max_diff',
        'campaign_compile', 'campaign_flop', 'predicted_a100_hours',
        'predicted_hours_here')}).to_string())
    assert SMOKE['resume_fp32_max_diff'] <= 1e-6
    assert SMOKE['resume_bf16_ok']  # 0.01 nats, deterministic kernels
```

<!-- Prose: the question of the opening, answered by the run set: the pair
of training mixtures whose target-weighted bits per byte cross (the one
ahead at a quarter of the tokens is behind at the end) by more than the
seed noise at both points (a gap between two runs counts when it exceeds
2 sqrt(2) seed SDs; the training runs share data-order seed 0, so this
bound is conservative; if no pair qualifies the opening says so), and the 3 x 3
matrix of the pure-domain runs (C1); bits per token against bits per byte
(C2). -->

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
print(TAG, f'{len(cross)} of {len(gaps)} pairs of training mixtures cross, '
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
print(TAG, 'bits per byte on each validation set (columns) of the '
      'pure-domain runs (rows):')
print(MATRIX.round(3).to_string())
print('each proxy is best on its own domain:', all(
    MATRIX.values[i, i] == MATRIX.values[:, i].min() for i in range(3)))
u = REC['uniform-wsd-s0']['final']
print(TAG, 'uniform mixture (WSD, seed 0) per domain:')
print(pd.DataFrame({d: dict(
    bytes_per_token=u[d]['n_bytes'] / u[d]['n_tokens'],
    bits_per_token=u[d]['nll'] / u[d]['n_tokens'] / math.log(2),
    bits_per_byte=u[d]['bpb']) for d in DOMAINS}).round(3).to_string())
```

<!-- Prose: three regressions from weights to per-domain bits per byte:
ridge on two free weights and an intercept (the three weights sum to one,
so all three and an intercept are collinear), gradient boosting (RegMix's
learner) and the exponential mixing law of Ye et al.; choosing a mixture
minimizes the fitted target loss on a random grid, polished by Nelder-Mead
on a softmax parameterization, inside the convex hull of the training
mixtures, where the regression was fitted (the selected arm of the
launcher uses this function). Why the mixing law selects: a model linear
in w (ridge) has its optimum at a corner of whatever region it is given,
so its choice is a corner, never an interior mixture; the law's
per-domain exponentials can place one inside. -->

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

<!-- Prose: how well can any regression do? The noise floor of a
regression's prediction of a noisy observation, sigma sqrt(1 + p/n)
(eq_lm_mix_floor, the short derivation), checked by simulation;
the floor of the real runs from the uniform seeds; the three regressions
fitted on the training runs and judged only on the held-out ones (MAE per
domain against the floor, Spearman rho without a significance claim).
The verdict compares the RMSE of the selecting regression (the mixing law,
fixed before any run; ridge and GBM are comparisons, never picked by their
held-out error, which C4 reports on) with 1.5 floors, and the floor's
interval is printed: sigma_cos rests on three degrees of freedom. Two
design facts in one sentence each: every training and held-out run uses
data-order seed 0 (the blueprint's design: the shared seed's share of the
noise cancels in the intercept, so the floor from independent seeds is an
upper bound on the noise the held-out errors face); and the same
validation sets choose the learning rate, fit the regression and score
the held-out mixtures ("held-out" means held-out mixtures, not held-out
text; the selection has few degrees of freedom, three rates and two free
weights, so its bias is small, and the text says so rather than claims
it is zero).
Pitfalls: training-mixture R^2 read as success; tuning the learning rate
per mixture. The plot below: (a) predicted against measured with the
identity and the band of two floors, (b) every run on the simplex, coloured
by its t_dep bits per byte, with the selected mixture as a star. -->

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
print(f'D4: mean squared error at new mixtures {(err / 5000).mean():.3e}, '
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
print(TAG, f'{len(train_ids)} training and {len(held_ids)} held-out runs; '
      f'pooled sigma_cos {SIG_COS:.4f}, sigma_seed {SIG_SEED:.4f} bits per '
      f'byte (ratio {max(SIG_COS, SIG_SEED) / min(SIG_COS, SIG_SEED):.1f})')
print(pd.DataFrame(noise, index=DOMAINS).T.round(4).to_string())
FLOOR_CI = FLOOR * np.sqrt(3 / stats.chi2.ppf([0.975, 0.025], 3))  # 3 df
rmse = math.sqrt(((PRED[C['select']] - Lh)**2).mean())
print(f'floor {FLOOR:.4f} (95% interval from 3 degrees of freedom '
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
         color='red', markersize=16, label='selected for t_dep')
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

## Choosing a Mixture
:label:`subsec_lm_mix_choose`

<!-- Prose: optimize the fitted function for t_dep (and t_avg); the
interval from refitting on bootstrap resamples of the training runs; the
pitfall of choosing target weights after the results. -->

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
print(TAG, 'selected mixtures (weights web, code, math) and their predicted '
      'target-weighted bits per byte (the last one free of the hull bound):')
for k, (w, v) in choice.items():
    where = ('inside the training mixtures' if HULL.find_simplex(w[:-1]) >= 0
             else 'extrapolated')
    print(f'  {k:18s} w = ({", ".join(f"{x:.3f}" for x in w)}), predicted '
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
print(f'{C["select"]} on {B} bootstrap resamples of the training runs: '
      f'predicted t_dep {np.median(bv):.4f} [{lo:.4f}, {hi:.4f}]; weights '
      'from ' + ', '.join(f'{d} {a:.2f}-{b:.2f}' for d, a, b in zip(
          DOMAINS, *np.percentile(bw, [2.5, 97.5], axis=0))))
```

<!-- Prose: DoReMi's one difference: it minimizes the worst-case excess loss
over a reference model; its weight step is the closed-form maximizer of a
KL-regularized objective (one display); the excess loss is clipped per
token against the reference model on the same tokens. Our miniature
follows the paper: the reference is the uniform-mixture proxy (the
uniform-wsd-s0 run's final checkpoint). -->

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

<!-- Prose: the D3 checks on a three-domain quadratic toy (the closed-form
step against a numerical maximization; the averaged weights near the
saddle's), the per-token clipping on a batch where clipping the domain mean
would give no excess at all, and the weights of the registered run set. -->

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
print(f'D3: closed-form step = numerical maximizer to {gap:.1e}; averaged '
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
print(f'excess loss of a random batch: per token {per_token:.4f}, from the '
      f'domain mean {per_domain:.4f}; doremi_weights used '
      f'{tr[0]["excess"][0]:.4f}')
DOREMI_W = np.array(RUNS['doremi']['weights'])
steps = [0, len(RUNS['doremi']['trace']) // 4, len(RUNS['doremi']['trace'])
         // 2, len(RUNS['doremi']['trace']) - 1]
print(TAG, 'DoReMi weights (step average):', dict(zip(
    DOMAINS, DOREMI_W.round(3).tolist())), '| alpha at steps', {
    s + 1: np.round(RUNS['doremi']['trace'][s]['alpha'], 3).tolist()
    for s in steps})
```

<!-- Prose: the four arms (uniform, regression-selected, DoReMi, staged),
all on the WSD schedule with the same peak, three data-order seeds each,
compared seed-paired with uniform; a difference counts only if all three
share a sign and the mean exceeds two sigma_seed (C5, C6). The plot: three
seed points and the mean per arm, the uniform mean and a band of two
sigma_seed. -->

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
                     stages=[(f, np.round(v, 3).tolist())
                             for f, v in w['stages'] or []],
                     t_dep_mean=T[a].mean(),
                     diffs=np.round(d, 4).tolist(), real=real))
ARM_TABLE = pd.DataFrame(rows)
print(TAG, f'sigma_seed of t_dep bits per byte {sig_t:.4f} (3 uniform '
      'seeds); differences seed-paired with uniform:')
print(ARM_TABLE.round(4).to_string(index=False))
ax = d2l.plot_intervals(
    list(ARMS), {'mean': ([T[a].mean() for a in ARMS],) * 3}, 'arm',
    't_dep bits per byte', reference=T['uniform'].mean(),
    marks={'seeds': ('.', {'mean': (np.repeat(np.arange(4), len(SEEDS)),
                                    np.concatenate([T[a] for a in ARMS]))})})
band = ax.axhspan(T['uniform'].mean() - 2 * sig_t,
                  T['uniform'].mean() + 2 * sig_t, color='0.85', zorder=0)
```

## Annealing and Curricula
:label:`subsec_lm_mix_anneal`

<!-- Prose: stable-then-decay (restate sec_scheduler): the staged
arm reads the uniform mixture for 80% of its tokens and the selected one in
the 20% decay; per-domain bits per byte of the staged and uniform arms
(mean over the three seeds), with the stage boundary (C7); recency against
capability; Dolma 3 / Olmo 3's mid-training mix and SmolLM2's stages as
cases (numbers only as their cards state them). -->

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
print(TAG, 'change of bits per byte during the decay, staged minus uniform '
      f'(mean over {len(SEEDS)} seeds; staged weights '
      f'{np.round(REC["staged-s0"]["stages"][0][1], 3).tolist()}):')
print({d: round(float((curve('staged', d) - curve('uniform-wsd', d))[dec[-1]]
                      - (curve('staged', d) - curve('uniform-wsd', d))
                      [dec[0] - 1]), 4) for d in DOMAINS})
```

## What Transfers
:label:`subsec_lm_mix_transfer`

<!-- Prose: the limits of extrapolation (tokens per parameter, model size,
tokenizer) in one paragraph; which statements this section makes at which
scale (T1 now; the firm claims wait for the T2 logs); the T2 campaign
package. Whether proxy rankings transfer to a larger model is not measured
here (exercise). Forward to sec_lm_continued_pretraining only if
the label resolves when the prose is written. -->

```{.python .input #data-mixtures-t2-package}
%%tab pytorch
c2 = d2l.MIX_CONFIGS['t2']
print('T2 campaign:', f'python tools/run_mix_campaign.py --config t2 '
      '--gpus 0,1,2,3 --out ../data/lm/rerun/mix-t2 (a faster node)')
print((ROOT / 'tools' / 'lm_campaigns.yaml').read_text().strip())
n2 = len(d2l.mix_run_specs(c2))
if SMOKE is not None:
    print(f'from the smoke on {SMOKE["device"]}: {SMOKE["campaign_flop"]:.2g} '
          f'FLOP for {n2} runs; A100-hours at the measured MFU '
          f'{two(SMOKE["predicted_a100_hours"])}, peak memory per run '
          f'{two(SMOKE["peak_mem_gib"])} GiB')
print('writes:', ', '.join([c2['runs_key']] + sorted(set(c2['keys'].values()))
                           + [c2['code_meta']]))
```

## Summary

<!-- Prose (stage 2): 3-6 sentences, every T1 number with "at T1 scale". -->

## Exercises

<!-- Prose (stage 2): the six exercises of the specification (20.4 part 8). -->
