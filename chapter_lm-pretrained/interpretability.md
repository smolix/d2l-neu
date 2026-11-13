# Interpretability
:label:`sec_lm_interpretability`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.6 Interpretability · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Claims about what a model computes are tested by intervening on activations, not by reading them.
**Opening:** The reader asks Qwen3-0.6B where the Eiffel Tower stands and watches the answer's rank rise through the layers under the logit lens. Reading the residual stream suggests where the fact is retrieved; only overwriting it says whether that is where. The section builds the interventions and closes on what pushing a feature costs the model elsewhere.
**Teaches:**
- The residual stream read by the logit lens and the tuned lens (`d2l.logit_lens`, `d2l.TunedLens`); linear probes with a control task, so selectivity separates what a representation encodes from what the probe learns.
- Activation patching between a clean and a corrupted prompt as a causal effect per layer and position (`d2l.intervene`); attribution patching as its first-order estimate, with the gradient cached by the same hooks.
- Sparse autoencoders as dictionary learning on the residual stream: the objective, bounded decoder norms, the top-k variant; a small SAE trained in the notebook (`d2l.TopKSAE`, `d2l.train_sae`) measured against a released Qwen-Scope SAE by L0, fraction of variance unexplained and loss recovered (`d2l.sae_metrics`).
- One steering vector with its perplexity cost beside the behaviour change; chain-of-thought monitorability in one paragraph as the bridge to reasoning traces.
- Gemma Scope 2 transcoders and an attribution graph as one exercise (the only place a third family appears, and only there).
**Derivation:** Patching as a causal effect: the metric difference when one site's activation is replaced by its value under the corrupted prompt, summed over sites to recover the total effect; its first-order form, the inner product of the activation difference with the metric's gradient at that site, and when the linearization fails. The SAE objective: reconstruction plus an L1 penalty on codes with decoder columns constrained to unit norm (otherwise the penalty is defeated by scaling), and top-k as the L0-constrained alternative. Smallest numeric case: a two-site patch where the effects do not add, computed by hand.
**Notebook:**
1. Track the rank of the correct answer across layers on 40 committed recall prompts with the logit lens and with a tuned lens trained on cached activations (or loaded from the committed translators).
2. Fit a part-of-speech probe on one layer with a control task of random tags per word type and report accuracy and selectivity.
3. Patch `resid_post` and `attn_out` at every layer and position from the corrupted to the clean run and map the effect; then compute attribution patching from one backward pass and plot its agreement with the patched map.
4. Train a top-k SAE on one layer's residual stream of Qwen3-1.7B-Base over a few million cached tokens and compare its reconstruction and loss recovered with the released Qwen-Scope SAE for the same layer; label three features by top examples and call the labels hypotheses.
5. Add one steering direction (an SAE feature chosen by contrast and a difference-of-means vector) at a grid of strengths and plot the behaviour change against the perplexity cost on held-out English text, with a random direction of the same norm as the floor.
6. Claim that survives noise: a few (layer, position) cells carry most of the patching effect for the recall behaviour (a cluster bootstrap over prompts gives the interval); attribution patching agrees with patching in rank order outside the near-zero cells; the random direction yields no behaviour change below the cost cap, and both informed directions do, with the cheaper one stated only if the frontier areas' bootstrap intervals separate.
7. Reused pieces: `d2l.load_qwen3` and the module names of 1.2, `d2l.generate`, 1.4's cross-entropy for the cost, 1.5's `d2l.bootstrap_ci(cluster=)`.
**Figures:** `fig_lm_patching` (clean and corrupted prompts, the sites, and the patching map as a heat map); `fig_lm_lens_rank` (plot: rank of the correct answer by layer under both lenses); `fig_lm_sae_frontier` (plot: the steering frontier, behaviour change against perplexity cost, for the three directions).
**Restate / forward:** `sec_what-attention-computes` (induction heads were located by ablation; `tab_induction-evidence` orders behavioural, correlational and interventional evidence, and this section climbs it); `sec_mdl-svd-low-rank` (a linear probe is a rank-one read-out and the difference-of-means vector is its simplest case); `eq_stream-update` (every block adds to a shared stream the lens reads). Forward: Chapter 9 (monitoring reasoning traces).
**Exercises:** 6 to 8. Load-bearing: derive attribution patching from the first-order expansion and bound its error by the second derivative along the patch (derive); measure the steering frontier of a second feature and of the difference-of-means direction at matched perplexity cost (measure); load a Gemma Scope 2 transcoder for one Gemma layer, build a one-step attribution graph for one prompt, and compare with the SAE view (open); check that the logit lens applied to the last layer reproduces the model's logits exactly (check).
**Writer notes:**
- Exports: `d2l.intervene(model, sites, *, record=(), record_grad=False)` as a context manager over `d2l.GPT` with site kinds `resid_post, resid_mid, attn_out, mlp_out, head_z`; `d2l.logit_lens(model, hidden, layer=None)` (applies the final norm and the tied head, fp32); `d2l.TunedLens(d_model, num_layers)` with translators initialized at zero so training starts at the logit lens; `d2l.TopKSAE(d_model, d_sae, k)` with the layout and names of Qwen-Scope's files (`W_enc (d_sae, d_model)`, `b_enc`, `W_dec (d_model, d_sae)`, `b_dec`), `normalize_decoder_()`; `d2l.load_qwen_scope_sae(repo_id, layer, revision=None, ...)` loading one `layer{n}.sae.pt` with `weights_only=True`; `d2l.sae_metrics` excluding position 0 and norm outliers (the first-token sinks the Qwen-Scope report filtered).
- Models: patching, lenses and the probe run on the book's Qwen3-0.6B (28 layers, d = 1024); the SAE cells run on Qwen3-1.7B-Base through the same `d2l.load_qwen3`, because Qwen-Scope releases SAEs for Qwen3-1.7B-Base, Qwen3-8B-Base and Qwen3.5-2B-Base, not for 0.6B. One released layer is 537 MB in fp32; a layer sweep is optional. The `SMALL=True` path uses Qwen3-0.6B with its own notebook-trained SAE and no download.
- Licence: the Qwen-Scope card is tagged `license: other` with a use-restriction "Caution" paragraph while its LICENSE file is Apache-2.0 text; the notebook takes the stricter reading: run-time download only, no committed or redistributed SAE weights, the Caution quoted verbatim from the downloaded README before first use. If the editor judges the restriction incompatible with the book's licence, the released SAE becomes a pointer and the frontier uses the notebook's own SAE feature.
- Released-SAE convention: whether the encoder subtracts `b_dec` before `W_enc` is not stated on the card; decide by reconstruction, print both, and stop using the release if neither reconstructs (the instrument check fails loudly).
- Qwen-Scope is the notebook's suite; Gemma Scope 2 transcoders and the attribution graph are one exercise (no Gemma 3 run in the notebook); a monitorability paragraph is included; the five cells beyond patching (two lenses, probe, SAE, steering) are one cell each and patching is the section's experiment.
- Patching data: 40 committed (clean, corrupted, correct, distractor) prompt pairs of equal token length built from a landmark-and-city list; pairs where the clean top-1 is wrong are discarded and counted. Probe data: Universal Dependencies English EWT (CC BY-SA 4.0, pinned by commit), last sub-token per word. Contrast and cost data: FineWeb-Edu English (ODC-By) and Chinese Wikipedia (CC BY-SA 3.0) documents at run time.
- Pitfalls to state: a probe's accuracy is not the model's use of the property; early logit-lens outputs are partly a scaling artifact; patching a corrupted prompt that differs in more than one variable; applying an SAE trained on one checkpoint to another is an assumption checked by reconstruction; steering strength is not comparable across directions unless normalized; `.pt` files load with `weights_only=True` only.
- Hooks across versions: Transformers decoder-layer outputs changed between tensor and tuple; the oracle cell handles both and asserts the last-layer relation to the final-norm output; if TransformerLens lacks Qwen3 support at the pinned version, the oracle is Transformers with hooks.
- Committed artifacts: tuned-lens translators (`lm-tuned-lens-qwen3-0.6b`, about 57 MB in fp16), probe weights, the notebook's small SAEs (`lm-sae-r1-l14`, `lm-sae-q17b-l14`, fp16), patching effect tables, steering results as CSV; none of the released SAE.
- Read the exact Gemma Scope 2 transcoder repository and layer for the exercise from the Gemma Scope page at authoring. Run-time use of Qwen-Scope is approved (academic use) under the Caution terms above.
**Sources:**
- Papers: [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) (the tuned lens); [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (causal tracing, the template for the recall patching map); [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042) (metrics and methods); [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) (dictionary learning on the residual stream); [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) (top-k SAEs, the auxiliary loss, evaluation by loss recovered); [Qwen-Scope](https://arxiv.org/abs/2605.11887) (the released suite; its language-steering demonstration without a cost measurement, which the notebook adds); [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (transcoders and attribution graphs, the exercise); [ADAG: Automatically Describing Attribution Graphs](https://arxiv.org/abs/2604.07615) (automated description of attribution graphs, one pointer); [Persona Vectors](https://arxiv.org/abs/2507.21509) (steering directions for traits, the monitorability paragraph); [Representation Engineering](https://arxiv.org/abs/2310.01405) and [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (difference-of-means steering).
- Code and documentation: the released [Qwen-Scope SAE for Qwen3-1.7B-Base](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50) (file layout, config, the Caution paragraph); [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope) (Gemma Scope 2, transcoders; the exercise); [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (the patching oracle where it supports Qwen3); [SAELens](https://github.com/decoderesearch/SAELens) (SAE training conventions to compare against); [tuned-lens](https://github.com/EleutherAI/tuned-lens) (the reference implementation of the lens).
- Courses, tutorials and blogs: [ARENA](https://github.com/ARENA-education/ARENA_materials) (the curriculum this section follows: lenses, patching, SAEs); nostalgebraist, [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); Neel Nanda, [Attribution Patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) (the first-order estimate at scale).
-->

```{.python .input #interpretability-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #interpretability-extras}
%%tab pytorch
import contextlib
import gc
import hashlib
import io
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import re
import zipfile
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from scipy.stats import rankdata, spearmanr
from transformers import AutoTokenizer
```

<!-- Placeholder (prose, opening): the flags of the CPU path and of regeneration; Qwen3-0.6B in fp32, as loaded in the d2l-checkpoint section. -->

```{.python .input #interpretability-setup}
%%tab pytorch
SMALL = not torch.cuda.is_available()  # the CPU path: R1 only, less data
RETRAIN = os.environ.get('D2L_RERUN', '0') == '1'  # regenerate artifacts
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')
SIZES = (dict(pairs=10, heldout=32, steer=20, cost=20, contrast=60,
              label=40, words=(5_000, 2_000, 2_000), random=2) if SMALL else
         dict(pairs=40, heldout=200, steer=100, cost=200, contrast=300,
              label=100, words=(20_000, 5_000, 5_000), random=5))
TRAIN_TOKENS, LENS_STEPS, SAE_STEPS = 8_000_000, 250, 5_000  # RETRAIN only
torch.manual_seed(0)
R1 = 'Qwen/Qwen3-0.6B'
model = d2l.load_qwen3(R1, dtype=torch.float32).requires_grad_(False)
tok = AutoTokenizer.from_pretrained(R1, revision=model.revision)
dev, L = model.token_emb.weight.device, len(model.blks)
print(f'{R1} at {model.revision}: '
      f'{sum(p.numel() for p in model.parameters()):,} parameters')
```

<!-- Placeholder (prose, none; helpers): range reads of parquet files on the Hub, the cluster bootstrap over cities or documents, and the bookkeeping of the regeneration rule. -->

```{.python .input #interpretability-helpers}
%%tab pytorch
FS = HfFileSystem()

def read_rows(repo, path, revision, start, stop, column='text'):
    """Rows start, ..., stop - 1 of one column of a parquet file on the
    Hub, reading only the row groups that hold them (range requests);
    returns the values and the compressed bytes read."""
    with FS.open(f'datasets/{repo}@{revision}/{path}', 'rb') as f:
        pf = pq.ParquetFile(f)
        md, col = pf.metadata, pf.schema_arrow.names.index(column)
        ends = np.cumsum([md.row_group(g).num_rows
                          for g in range(md.num_row_groups)])
        groups = [g for g, e in enumerate(ends)
                  if e > start and e - md.row_group(g).num_rows < stop]
        values = pf.read_row_groups(groups, columns=[column]).column(0)
        first = ends[groups[0]] - md.row_group(groups[0]).num_rows
        nbytes = sum(md.row_group(g).column(col).total_compressed_size
                     for g in groups)
    return values.to_pylist()[start - first:stop - first], nbytes

def cluster_resample(clusters, n_boot=2000, seed=0):
    """Per replicate, the item indices of a resample of whole clusters
    (clusters holds one cluster id per item)."""
    ids, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(ids))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[g] for g in rng.integers(
            len(ids), size=len(ids))])

def pct_ci(values, alpha=0.05):
    """Percentile endpoints of bootstrap replicates (along axis 0)."""
    return np.nanquantile(values, [alpha / 2, 1 - alpha / 2], axis=0)

def fmt(mean, ci, digits=3):
    return f'{mean:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'

def save_rerun(name, write):
    """Rule R1: write(path) makes an unversioned file in RERUN_DIR, which
    the author uploads as the next -vN; prints its sha1."""
    os.makedirs(RERUN_DIR, exist_ok=True)
    path = os.path.join(RERUN_DIR, name)
    write(path)
    with open(path, 'rb') as f:
        print(f'wrote {path}, sha1 {hashlib.sha1(f.read()).hexdigest()}')

RESULTS, SUMMARY = {}, {}  # the members of lm-interp-results, by name

def write_results(path):
    """The results archive, byte-identical for identical members."""
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in sorted(RESULTS.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
```

<!-- Placeholder prose (stage 2): the section text is written after the runs. -->
The released sparse autoencoder is pinned like the models of
:numref:`sec_lm_hf_models`, by adding its repository to `d2l.HF_REVISIONS`.

```{.python .input #interpretability-revisions}
%%tab pytorch
#@save
d2l.HF_REVISIONS['Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'] = (
    'ce1a79d9c5163932d65c417380e53230e1086370')
```

<!-- Placeholder (stage 2): the probe's treebank, pinned to a commit, and the recall pairs; the trained artifacts are registered at capture. -->

```{.python .input #interpretability-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-ud-ewt-train'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-train.conllu',
    '59e9ff791f828f7cd3dcd2e55dc4f30ea7bbd5f9')
d2l.DATA_HUB['lm-ud-ewt-dev'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-dev.conllu',
    '1e3226c7b533354e4f79e7bae7a1aa56a7dd9c94')
d2l.DATA_HUB['lm-ud-ewt-test'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-test.conllu',
    '41b279558051f482d78fff4ea7b3c8c98c91ecd4')
# The recall pairs (tools/oneshot/build_lm_recall_pairs.py)
d2l.DATA_HUB['lm-interp-recall-v1'] = (
    d2l.LM_DATA_URL + 'lm-interp-recall-v1.json',
    'e3025b0216df5384d8a14f1e8e10c4cab080ebbf')
```

<!-- Placeholder (prose, ### The Residual Stream as a Sum): the data of the section, each subset resolved once by a declared rule (FineWeb-Edu, Chinese and English Wikipedia, the recall pairs), and the Han-character tokens of the vocabulary. -->

```{.python .input #interpretability-data}
%%tab pytorch
FINEWEB = ('HuggingFaceFW/fineweb-edu', 'sample/10BT/013_00000.parquet',
           '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
WIKI = {lang: ('wikimedia/wikipedia', f'20231101.{lang}/{shard}.parquet',
               'b04c8d1ceb2f5cd4588862100d08de323dccfbaa')
        for lang, shard in (('zh', 'train-00002-of-00006'),
                            ('en', 'train-00028-of-00041'))}
CJK = re.compile('[㐀-䶿一-鿿豈-﫿]')  # Han
ids_of = lambda text: tok(text, add_special_tokens=False).input_ids
prefix = lambda n, need: lambda ids: ids[:n] if len(ids) >= need else None

def clean_prompt(ids):  # 32 tokens without Han that survive a round trip
    text = tok.decode(ids[:32])
    keep = len(ids) >= 32 and not CJK.search(text) and ids_of(text) == ids[:32]
    return ids[:32] if keep else None

def walk(source, row, subsets, block=1000):
    """From row on, the first n qualifying documents of each subset in
    turn; subsets maps a name to (n, take), take(ids) -> ids or None."""
    out, texts, start = {}, [], row
    for name, (n, take) in subsets.items():
        docs, first, nbytes = [], row, 0
        while len(docs) < n:
            if row >= start + len(texts):  # the next block of rows
                start = row
                texts, b = read_rows(*source, row, row + block)
                nbytes += b
            if (ids := take(ids_of(texts[row - start]))) is not None:
                docs.append(ids)
            row += 1
        out[name] = torch.tensor(docs)
        print(f'{source[1]}, {name}: {n} docs from rows {first}-{row - 1}, '
              f'{nbytes / 1e6:.1f} MB read')
    return out

fineweb = walk(FINEWEB, 8000, {
    'held-out': (SIZES['heldout'], prefix(256, 257)),
    'steering': (SIZES['steer'], clean_prompt),
    'cost': (SIZES['cost'], prefix(129, 129))})
wiki = {lang: walk(WIKI[lang], 0, {
    'contrast': (SIZES['contrast'], prefix(128, 129)),
    'label': (SIZES['label'], prefix(128, 129))}) for lang in WIKI}
heldout, cost_docs = fineweb['held-out'], fineweb['cost']
steer_texts = tok.batch_decode(fineweb['steering'])
pieces = tok.batch_decode([[v] for v in range(len(tok))])
cjk = torch.zeros(model.token_emb.num_embeddings, dtype=torch.bool)
cjk[:len(tok)] = torch.tensor([CJK.search(p) is not None for p in pieces])
print(f'{int(cjk.sum()):,} of {len(tok):,} tokens hold a Han character')
with open(d2l.download_keyed('lm-interp-recall-v1')) as f:
    recall = json.load(f)['rows'][:SIZES['pairs']]
if RETRAIN:  # every 256-token window of rows 0 to 7,999, in order
    texts, nbytes = read_rows(*FINEWEB, 0, 8000)
    windows = [ids[i:i + 256] for ids in ids_of(texts)
               for i in range(0, len(ids) - 255, 256)]
    train_windows = torch.tensor(windows[:TRAIN_TOKENS // 256])
    print(f'training windows: {train_windows.numel():,} tokens, '
          f'{nbytes / 1e6:.1f} MB read')
FULL = None  # the full run's summary, shown beside the CPU path's numbers
if SMALL and not RETRAIN:
    with zipfile.ZipFile(d2l.download_keyed('lm-interp-results-v1')) as z:
        FULL = json.loads(z.read('summary.json'))
```

<!-- Placeholder (stage 2): reading and overwriting the five sites of every block through module hooks. -->

```{.python .input #interpretability-intervene}
%%tab pytorch
#@save
class _InterveneStop(Exception):
    """Raised right after the stop_after site, to end a forward early."""

#@save
class _Intervention:
    """The context manager that d2l.intervene returns."""
    KINDS = ('resid_post', 'resid_mid', 'attn_out', 'mlp_out', 'head_z')

    def __init__(self, model, sites, record, record_grad, stop_after):
        assert isinstance(model, d2l.GPT), 'intervene instruments a d2l.GPT'
        for blk in model.blks:
            assert blk.pre_norm, 'intervene needs pre-norm blocks'
            assert not (blk.training and blk.dropout.p > 0), (
                'dropout must be inactive: call model.eval()')
        self.model, self.record_grad = model, record_grad
        self.sites = {self._check(s): fn for s, fn in (sites or {}).items()}
        for site, fn in self.sites.items():
            if fn is None:  # recording is record='s job
                raise ValueError(f'site {site} maps to None; use record=')
        self.stop_after = None if stop_after is None else self._check(
            stop_after)
        self.record = {self._check(s) for s in record}
        if self.stop_after is not None:  # a stopped forward is for reading
            self.record.add(self.stop_after)
        self.cache = {s: [] for s in self.record}
        self.wanted = set(self.sites) | self.record

    def _check(self, site):
        if not (isinstance(site, tuple) and len(site) == 2
                and site[0] in self.KINDS
                and isinstance(site[1], (int, np.integer))
                and 0 <= site[1] < len(self.model.blks)):
            raise ValueError(f'not a site of this model: {site!r}')
        return site[0], int(site[1])

    @property
    def grad_cache(self):
        """The gradient at each recorded tensor; read after a backward."""
        return {s: [x.grad for x in xs] for s, xs in self.cache.items()}

    def _site(self, site, x):
        """Apply the site's fn to x and record the result. Returns fn's
        value (None without a fn) and a term to add to what the module
        passes on (None without record_grad)."""
        self.fired.add(site)
        fn, y, zero = self.sites.get(site), None, None
        if fn is not None:
            y = fn(x)
            if not (torch.is_tensor(y) and y.shape == x.shape
                    and y.dtype == x.dtype):
                raise ValueError(f'{site}: fn must return a tensor of shape '
                                 f'{tuple(x.shape)} and dtype {x.dtype}')
        if site in self.record:
            v = (x if y is None else y).detach()
            if self.record_grad:
                if not torch.is_grad_enabled():
                    raise RuntimeError('record_grad needs gradients enabled')
                # leaf - leaf.detach() is exactly 0: adding it leaves the
                # forward unchanged and makes leaf.grad the gradient there
                v.requires_grad_()
                zero = v - v.detach()
            self.cache[site].append(v)
        if site == self.stop_after:
            raise _InterveneStop(self)
        return y, zero

    def _write_hook(self, inner, outer, base):
        """Forward hook on a module whose output a block adds to its stream
        (W_o, the FFN): site inner sees that output, site outer the stream
        after the add, r = base[0] + output."""
        def hook(module, args, out):
            new = out
            if inner in self.wanted:
                y, zero = self._site(inner, out)
                new = new if y is None else y
                new = new if zero is None else new + zero
            if outer in self.wanted:
                r = base[0] + new  # the stream, as the block's add makes it
                y, zero = self._site(outer, r)
                if y is not None:  # the block's own add then yields about y,
                    new = new + (y - r)  # exactly r wherever y equals r
                new = new if zero is None else new + zero
            return None if new is out else new
        return hook

    def _z_hook(self, site):
        """Forward pre-hook on W_o: site head_z sees its input."""
        def hook(module, args):
            y, zero = self._site(site, args[0])
            new = args[0] if y is None else y
            new = new if zero is None else new + zero
            return None if new is args[0] else (new,)
        return hook

    @staticmethod
    def _store_hook(box):
        """Forward pre-hook that keeps the module's input in box[0]."""
        def hook(module, args):
            box[0] = args[0]
        return hook

    def __enter__(self):
        self.fired, self.handles = set(), []
        for l, blk in enumerate(self.model.blks):
            kinds = {kind for kind, layer in self.wanted if layer == l}
            W_o, hooks = blk.attention.W_o, []
            if 'head_z' in kinds:
                hooks.append(W_o.register_forward_pre_hook(
                    self._z_hook(('head_z', l))))
            if kinds & {'attn_out', 'resid_mid'}:
                x_in = [None]  # the block's input, the stream before it
                hooks += [
                    blk.norm1.register_forward_pre_hook(
                        self._store_hook(x_in)),
                    W_o.register_forward_hook(self._write_hook(
                        ('attn_out', l), ('resid_mid', l), x_in))]
            if kinds & {'mlp_out', 'resid_post'}:
                mid = [None]  # the stream after attention, as norm2 sees it
                hooks += [
                    blk.norm2.register_forward_pre_hook(
                        self._store_hook(mid)),
                    blk.ffn.register_forward_hook(self._write_hook(
                        ('mlp_out', l), ('resid_post', l), mid))]
            self.handles += hooks
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        if isinstance(exc, _InterveneStop):
            return exc.args[0] is self  # another context's stop goes on
        if exc_type is None:
            unfired = sorted(self.wanted - self.fired)
            if unfired:
                raise RuntimeError(f'sites that never fired: {unfired}')
        return False

#@save
def intervene(model, sites=None, *, record=(), record_grad=False,
              stop_after=None):
    """Context manager that reads and overwrites activations of a d2l.GPT
    with pre-norm blocks, in the full forward and in forward_cached. A
    site is (kind, layer), kind one of 'resid_post', 'resid_mid',
    'attn_out', 'mlp_out', 'head_z' (the per-head input of W_o, (B, T,
    heads * head_dim)). sites maps a site to fn(x) -> x_new (same shape and
    dtype), applied in every call; at a resid site the block's own add
    then carries x_new (rounded), and exactly x wherever x_new equals x.
    Nested contexts apply the outer fn first, and each context reads a
    block at its own hooks: an outer context's resid_mid or resid_post is
    the stream before an inner context's edit of attn_out or mlp_out in
    that block, so record and edit in one context. ctx.cache[site]
    lists, for each recorded site, one tensor per call (after fn). With
    record_grad=True, ctx.grad_cache[site] lists their gradients after
    the caller's backward. stop_after=site ends the forward right after
    that site (recorded) without an error. On a normal exit, a site that
    never fired raises."""
    return _Intervention(model, sites, record, record_grad, stop_after)
```

<!-- Placeholder (stage 2): the logit lens, the final norm and the head applied to any block. -->

```{.python .input #interpretability-logit-lens}
%%tab pytorch
#@save
def logit_lens(model, hidden):
    """The model's final norm and output head applied in fp32 to hidden
    (..., d), a stream read at any block; returns logits (..., V)."""
    norm, h = model.norm, hidden.float()
    f32 = lambda p: None if p is None else p.float()  # no affine, no bias
    if isinstance(norm, nn.RMSNorm):
        h = F.rms_norm(h, norm.normalized_shape, f32(norm.weight), norm.eps)
    else:  # nn.LayerNorm
        h = F.layer_norm(h, norm.normalized_shape, f32(norm.weight),
                         f32(norm.bias), norm.eps)
    head = model.lm_head if hasattr(model, 'lm_head') else model.token_emb
    return F.linear(h, head.weight.float())
```

<!-- Placeholder (prose, ### The Residual Stream as a Sum): check of eq_lm_dla on eight recall prompts (the contributions sum to the logit; the lens at the last block is the model), and the factor rho_L / rho_l by which the lens rescales an early block. -->

```{.python .input #interpretability-dla-check}
%%tab pytorch
assert not hasattr(model, 'lm_head')  # Qwen3-0.6B ties its output head
gamma, E = model.norm.weight, model.token_emb.weight
rho = lambda h: (h.pow(2).mean(-1) + model.norm.eps).sqrt()
sites = [(k, l) for l in range(L) for k in ('attn_out', 'mlp_out')]
blocks = [('resid_post', l) for l in range(L)]
ratios = []
for r in recall[:8]:
    X = torch.tensor([ids_of(r['template'].format(r['landmark']))], device=dev)
    city = ids_of(' ' + r['city'])[0]
    with torch.no_grad(), d2l.intervene(model, record=sites + blocks) as ctx:
        z = model(X)[0, -1]
        h0 = model.token_emb(X)[0, -1]
    h = torch.stack([ctx.cache[s][0][0, -1] for s in blocks])  # (L, d)
    parts = torch.stack([h0] + [ctx.cache[s][0][0, -1] for s in sites])
    c = parts @ (gamma * E[city]) / rho(h[-1])  # eq_lm_dla, one per part
    assert abs(float(c.sum() - z[city])) < 1e-3
    assert torch.allclose(d2l.logit_lens(model, h[-1]), z, rtol=0, atol=1e-4)
    ratios.append((rho(h[-1]) / rho(h[::4])).tolist())
print(pd.DataFrame(np.mean(ratios, 0)[None], index=['rho_27 / rho_l'],
                   columns=[f'block {l}' for l in range(0, L, 4)]).round(
    2).to_string())
```

<!-- Placeholder (prose, opening and ### Logit Lens and Tuned Lens): the opening picture, the final top-5 for the Eiffel prompt and the logit lens's top-1 token and the rank of " Paris" at every fourth block. -->

```{.python .input #interpretability-eiffel}
%%tab pytorch
eiffel = recall[0]['template'].format(recall[0]['landmark'])
paris = ids_of(' Paris')[0]
with torch.no_grad(), d2l.intervene(model, record=blocks) as ctx:
    p = F.softmax(model(torch.tensor([ids_of(eiffel)], device=dev))[0, -1], -1)
top = p.topk(5)
print(eiffel)
print(pd.DataFrame({
    'next token': map(repr, tok.batch_decode(top.indices[:, None])),
    'probability': top.values.tolist()}).round(3))
lens_rows = []
for l in [*range(0, L, 4), L - 1]:
    z = d2l.logit_lens(model, ctx.cache[blocks[l]][0][0, -1])
    rank = int((z > z[paris]).sum()) + 1
    lens_rows.append((l, repr(tok.decode(z.argmax())), rank))
print(pd.DataFrame(lens_rows, columns=['block', 'logit-lens top-1',
                                       'rank of " Paris"']).to_string(
    index=False))
```

<!-- Placeholder (stage 2): the tuned lens, one affine translator per block, trained toward the final distribution. -->

```{.python .input #interpretability-tuned-lens}
%%tab pytorch
#@save
class TunedLens(nn.Module):
    """One affine translator h + A_l h + b_l per block but the last (the
    identity), applied before the logit lens; A_l and b_l start at zero,
    so a fresh lens is the logit lens."""
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.translators = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(num_layers - 1))
        for t in self.translators:
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)

    def forward(self, hidden, layer):
        if not 0 <= layer < self.num_layers:
            raise IndexError(f'layer {layer} of {self.num_layers}')
        if layer == self.num_layers - 1:
            return hidden
        t = self.translators[layer]
        h = hidden.to(t.weight.dtype)
        return h + t(h)

    def _check(self, model):
        if len(model.blks) != self.num_layers:
            raise ValueError(f'a lens of {self.num_layers} layers for a '
                             f'model of {len(model.blks)} blocks')

    def logits(self, model, hidden, layer):
        self._check(model)
        return d2l.logit_lens(model, self(hidden, layer))

    def fit(self, model, batches, *, steps, lr=1e-3, frac=0.25, seed=0,
            chunk=512):
        """Adam on KL(final || lens), summed over the layers, at a random
        fraction frac of the positions >= 1 of each sequence of each batch
        (equal-length (B, T) ids) of the frozen model. Each layer's loss is
        backpropagated alone, chunk positions at a time, so the memory
        holds one chunk of logits. Returns the loss of each step."""
        assert not any(p.requires_grad for p in model.parameters()), (
            'freeze the model first: model.requires_grad_(False)')
        self._check(model)
        L, dev = self.num_layers, self.translators[0].weight.device
        sites = [('resid_post', l) for l in range(L)]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        gen, batches, losses = torch.Generator().manual_seed(seed), iter(
            batches), []
        for _ in range(steps):
            X = next(batches, None)
            if X is None:
                raise ValueError(f'batches ran out before {steps} steps')
            with torch.no_grad(), d2l.intervene(
                    model, record=sites, stop_after=sites[-1]) as ctx:
                model(X.to(model.token_emb.weight.device))
            B, T = X.shape
            n = max(1, round(frac * (T - 1)))  # positions per sequence
            cols = torch.stack([torch.randperm(T - 1, generator=gen)[:n] + 1
                                for _ in range(B)]).flatten()
            rows = torch.arange(B).repeat_interleave(n)
            H = [ctx.cache[s][0][rows, cols].to(dev, torch.float32)
                 for s in sites]  # each (B * n, d), detached
            del ctx
            optimizer.zero_grad()
            total = 0.0
            for i in range(0, len(rows), chunk):
                with torch.no_grad():  # the final distribution, the target
                    target = F.log_softmax(
                        d2l.logit_lens(model, H[-1][i:i + chunk]), -1)
                for l in range(L - 1):
                    lens = F.log_softmax(self.logits(model, H[l][i:i + chunk],
                                                     l), -1)
                    kl = F.kl_div(lens, target, reduction='sum',
                                  log_target=True) / len(rows)
                    kl.backward()
                    total += kl.item()
            optimizer.step()
            losses.append(total)
        return losses
```

<!-- Placeholder (prose, ### Logit Lens and Tuned Lens): the translators are fitted on the first million training tokens under RETRAIN; a normal build loads the committed ones. -->

```{.python .input #interpretability-tuned-lens-fit}
%%tab pytorch
lens = d2l.TunedLens(model.num_hiddens, L).to(dev)
if RETRAIN:  # LENS_STEPS batches of 16 windows: the first 1M tokens
    losses = lens.fit(model, train_windows[:16 * LENS_STEPS].split(16),
                      steps=LENS_STEPS)
    print(f'KL summed over blocks: {losses[0]:.2f} at the first step, '
          f'{losses[-1]:.2f} at the last')
    state = {k: v.half() for k, v in lens.state_dict().items()}
    save_rerun('lm-tuned-lens-qwen3-0.6b.pt', lambda p: torch.save(state, p))
else:
    key = 'lm-tuned-lens-qwen3-0.6b-v1'
    lens.load_state_dict(torch.load(d2l.download_keyed(key),
                                    weights_only=True))
    print(f'{key}: sha1 {d2l.DATA_HUB[key][1]}')
lens.requires_grad_(False);
```

<!-- Placeholder (prose, ### Logit Lens and Tuned Lens): the keep rule of the recall pairs, the rank of the correct city by block under both lenses (scale-free), and the KL to the final distribution on held-out text (with the scale); the plot below, caption from spec fig_lm_lens_rank. -->

```{.python .input #interpretability-lens-compare}
%%tab pytorch
def last_position(text):  # logits and the stream after each block, at -1
    with torch.no_grad(), d2l.intervene(model, record=blocks) as ctx:
        z = model(torch.tensor([ids_of(text)], device=dev))[0, -1]
    return z, torch.stack([ctx.cache[s][0][0, -1] for s in blocks])

LENSES = {'logit lens': lambda h, l: d2l.logit_lens(model, h),
          'tuned lens': lambda h, l: lens.logits(model, h, l)}
ranks, LD, kept = {name: [] for name in LENSES}, [], []
for r in recall:  # keep rule: correct top-1, and LD drops by 2 or more
    a, b = ids_of(' ' + r['city'])[0], ids_of(' ' + r['corrupt_city'])[0]
    z, H = last_position(r['template'].format(r['landmark']))
    z_x, _ = last_position(r['template'].format(r['corrupt_landmark']))
    LD.append((float(z[a] - z[b]), float(z_x[a] - z_x[b])))
    kept.append(int(z.argmax()) == a and LD[-1][0] - LD[-1][1] >= 2)
    for name, f in LENSES.items() if kept[-1] else ():
        zl = torch.stack([f(H[l], l) for l in range(L)])  # (L, V)
        ranks[name].append(((zl > zl[:, a:a + 1]).sum(-1) + 1).tolist())
cities = [r['city'] for r, k in zip(recall, kept) if k]

def first10(R):  # the first block where the median rank is at most 10
    hit = np.median(R, 0) <= 10
    return int(np.argmax(hit)) if hit.any() else np.nan

for name, R in ranks.items():
    R = ranks[name] = np.array(R)
    reps = [first10(R[idx]) for idx in cluster_resample(cities)]
    SUMMARY[f'{name}: first block of median rank <= 10'] = [
        float(first10(R)), *pct_ci(reps).tolist()]
    print(f'{name}: median rank at most 10 from block {first10(R)}, '
          f'95% interval {pct_ci(reps)}')
KL = {name: [[] for _ in range(L - 1)] for name in LENSES}
with torch.no_grad():
    for X in heldout[:, :128].split(4):  # 4 x 127 positions per chunk
        with d2l.intervene(model, record=blocks) as ctx:
            target = F.log_softmax(model(X.to(dev))[:, 1:].float(), -1)
        for l in range(L - 1):  # the last block: both lenses are exact
            for name, f in LENSES.items():
                lp = F.log_softmax(f(ctx.cache[blocks[l]][0][:, 1:], l), -1)
                KL[name][l].append(F.kl_div(lp, target, reduction='none',
                                            log_target=True).sum(-1).ravel())
doc = np.repeat(np.arange(len(heldout)), 127)

def mean_ci(values):  # over positions, resampling whole documents
    r = d2l.bootstrap_ci(torch.cat(values).cpu().numpy(), cluster=doc)
    return [r.mean, *r.ci]

KL = {name: np.array([mean_ci(v) for v in kls]) for name, kls in KL.items()}
fig, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.8), layout='constrained')
q = {name: np.percentile(R, [50, 25, 75], axis=0) for name, R in ranks.items()}
d2l.plot(list(range(L)), [q[n][0] for n in q] + [q[n][i] for n in q
                                                  for i in (1, 2)],
         'block', 'rank of the answer', legend=list(q), yscale='log',
         fmts=('C0-', 'C1-', 'C0--', 'C0--', 'C1--', 'C1--'), axes=axes[0])
axes[0].invert_yaxis()
d2l.plot(list(range(L - 1)), [KL[n][:, 0] for n in KL] + [
    KL[n][:, i] for n in KL for i in (1, 2)], 'block', 'KL to final (nats)',
         legend=list(KL), yscale='log', axes=axes[1],
         fmts=('C0-', 'C1-', 'C0:', 'C0:', 'C1:', 'C1:'))
```

<!-- Placeholder (prose, ### Linear Probes and Control Tasks): UD English EWT, every 10th training and every 5th dev and test sentence; each word read at its last token, from the embedding and after block 14. -->

```{.python .input #interpretability-probe-data}
%%tab pytorch
def read_conllu(path, stride, max_words):
    """Every stride-th sentence, up to max_words words: the text rebuilt
    with SpaceAfter=No, and per word (start, end, form, UPOS)."""
    with open(path, encoding='utf-8') as f:
        sentences, out, n = f.read().strip().split('\n\n'), [], 0
    for sentence in sentences[::stride]:
        text, words, end = '', [], 0
        for c in (line.split('\t') for line in sentence.split('\n')
                  if not line.startswith('#')):
            spaced = 'SpaceAfter=No' not in c[9]
            if '-' in c[0]:  # a multiword token; its words follow, unspaced
                end, space = int(c[0].split('-')[1]), spaced
            elif '.' not in c[0]:  # skip empty nodes
                words.append((len(text), len(text) + len(c[1]), c[1], c[3]))
                text += c[1]
                i = int(c[0])
                if i == end and space or i > end and spaced:
                    text += ' '
        if n + len(words) > max_words:
            break
        out.append((text.rstrip(), words))
        n += len(words)
    return out

def probe_features(sentences, batch=32):
    """Per word, at its last token: the embedding and the block-14 stream.
    Right padding is safe: a causal model never reads a later position."""
    emb, h14 = [], []
    for i in range(0, len(sentences), batch):
        chunk = [tok(t, return_offsets_mapping=True, add_special_tokens=False)
                 for t, _ in sentences[i:i + batch]]
        T = max(len(e.input_ids) for e in chunk)
        X = torch.tensor([e.input_ids + [0] * (T - len(e.input_ids))
                          for e in chunk], device=dev)
        with torch.no_grad(), d2l.intervene(
                model, stop_after=('resid_post', 14)) as ctx:
            model(X)
        H = ctx.cache[('resid_post', 14)][0]
        with torch.no_grad():
            E0 = model.token_emb(X)
        for j, (e, (_, ws)) in enumerate(zip(chunk, sentences[i:i + batch])):
            starts = [s for s, _ in e.offset_mapping]
            pos = [int(np.searchsorted(starts, w[1])) - 1 for w in ws]
            emb.append(E0[j, pos].cpu())
            h14.append(H[j, pos].cpu())
    return {'embedding': torch.cat(emb), 'block 14': torch.cat(h14)}

ud = {split: read_conllu(d2l.download_keyed(f'lm-ud-ewt-{split}'), stride, n)
      for split, stride, n in zip(('train', 'dev', 'test'), (10, 5, 5),
                                  SIZES['words'])}
feats = {split: probe_features(s) for split, s in ud.items()}
words = {split: [w for _, ws in s for w in ws] for split, s in ud.items()}
sentence_of = np.repeat(np.arange(len(ud['test'])),
                        [len(ws) for _, ws in ud['test']])
for split, s in ud.items():
    print(f'{split}: {len(words[split]):,} words, {len(s):,} sentences, '
          f'{len({w[2] for w in words[split]}):,} word types')
test_types = {w[2] for w in words['test']}
unseen = test_types - {w[2] for w in words['train']}
print(f'test word types unseen in training: '
      f'{len(unseen) / len(test_types):.2f}')
```

<!-- Placeholder (prose, ### Linear Probes and Control Tasks): linear probes for the tags and for five control tasks (one random tag per word type), with the majority tag per type as baseline; the table above (spec tab_lm_probe_control) and the paired selectivity difference. -->

```{.python .input #interpretability-probe}
%%tab pytorch
UPOS = sorted({w[3] for ws in words.values() for w in ws})
TYPES = sorted({w[2] for ws in words.values() for w in ws})  # all splits
WDS = (1e-4, 1e-3, 1e-2, 1e-1)
labels = {s: np.array([UPOS.index(w[3]) for w in words[s]]) for s in words}
freq = np.bincount(labels['train'], minlength=len(UPOS)) / len(words['train'])

def control(seed):  # one tag per word type, drawn from the train tags
    tag = dict(zip(TYPES, np.random.default_rng(seed).choice(
        len(UPOS), len(TYPES), p=freq)))
    return {s: np.array([tag[w[2]] for w in words[s]]) for s in words}

def fit_probe(X, y, wd):  # multinomial logistic regression, L-BFGS
    W = torch.zeros(X.shape[1], len(UPOS), device=dev, requires_grad=True)
    b = torch.zeros(len(UPOS), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()

def probe(layer, y):  # weight decay chosen on dev; correct per test word
    mu, sd = feats['train'][layer].mean(0), feats['train'][layer].std(0)
    X = {s: ((feats[s][layer] - mu) / sd).to(dev) for s in feats}
    Y = {s: torch.tensor(y[s], device=dev) for s in y}
    correct = lambda Wb, s: (X[s] @ Wb[0] + Wb[1]).argmax(-1) == Y[s]
    fits = {wd: fit_probe(X['train'], Y['train'], wd) for wd in WDS}
    wd = max(WDS, key=lambda w: float(correct(fits[w], 'dev').float().mean()))
    return correct(fits[wd], 'test').cpu().numpy(), (*fits[wd], mu, sd)

def majority(y):  # the most frequent train tag of each word type
    best = pd.DataFrame({'type': [w[2] for w in words['train']],
                         'tag': y['train']}).groupby('type')['tag'].agg(
        lambda t: t.value_counts().idxmax())
    default = np.bincount(y['train']).argmax()
    pred = np.array([best.get(w[2], default) for w in words['test']])
    return pred == y['test'], None

controls = [control(seed) for seed in range(5)]
table, selectivity, weights = {}, {}, {}
for name, f in (('majority tag per type', majority),
                ('embedding', lambda y: probe('embedding', y)),
                ('block 14', lambda y: probe('block 14', y))):
    task, weights[name] = f(labels)
    ctrl = np.mean([f(c)[0] for c in controls], 0)
    selectivity[name] = task.astype(float) - ctrl
    table[name] = [fmt(r.mean, r.ci) for r in (
        d2l.bootstrap_ci(v, cluster=sentence_of)
        for v in (task, ctrl, selectivity[name]))]
table = pd.DataFrame(table, index=['task accuracy', 'control accuracy',
                                   'selectivity']).T
print(table.to_string())
diff = d2l.paired_bootstrap(selectivity['block 14'], selectivity['embedding'],
                            cluster=sentence_of)
print(f'selectivity, block 14 minus embedding: {fmt(diff.mean, diff.ci)}')
SUMMARY['probe'] = {**table.to_dict('index'),
                    'difference': fmt(diff.mean, diff.ci)}
buffer = io.BytesIO()
torch.save({n: weights[n] for n in ('embedding', 'block 14')}, buffer)
RESULTS['probe.pt'] = buffer.getvalue()
if FULL is not None:
    print('full run:', json.dumps(FULL['probe'], indent=1))
```

<!-- Placeholder (prose, ### Clean and Corrupted Runs): the recall pairs the model answers, their cities, the LD on clean and corrupted prompts, and the eight position classes of each prompt. -->

```{.python .input #interpretability-recall-pairs}
%%tab pytorch
assert int(p.argmax()) == paris  # without a chat template, R1 says " Paris"
assert kept[0] and recall[0]['corrupt_landmark'] == 'Colosseum'
pairs = [r for r, k in zip(recall, kept) if k]
assert SMALL or (len(pairs) >= 30 and len(set(cities)) >= 15)
dropped = [r['landmark'] for r in recall if r not in pairs]
print(f'{len(pairs)} of {len(recall)} pairs kept, {len(set(cities))} '
      f'cities; discarded: {dropped}')
m_clean, m_corrupt = torch.tensor([ld for ld, k in zip(LD, kept) if k]).T
gap = m_clean - m_corrupt
print(f'mean LD: clean {m_clean.mean():.2f}, corrupted {m_corrupt.mean():.2f}')
CLASSES = ['subject', 'subject last', ' is', ' located', ' in', ' the',
           ' city', ' of']
by_length = {}
for i, r in enumerate(pairs):  # prompts of one length share their classes
    X, X_x = (ids_of(r['template'].format(r[k]))
              for k in ('landmark', 'corrupt_landmark'))
    assert len(X) == len(X_x) and tok.batch_decode(
        [[t] for t in X[-6:]]) == CLASSES[2:]
    by_length.setdefault(len(X), []).append(
        (i, X, X_x, ids_of(' ' + r['city'])[0],
         ids_of(' ' + r['corrupt_city'])[0]))
groups = [{'idx': torch.tensor([g[0] for g in G]),
           'X': torch.tensor([g[1] for g in G]),
           'X_x': torch.tensor([g[2] for g in G]),
           'ab': torch.tensor([g[3:] for g in G]),
           'cls': torch.tensor([-1] + [0] * (T - 8) + list(range(1, 8)))}
          for T, G in by_length.items()]  # 'The' is in no class (-1)
print({T: len(G) for T, G in by_length.items()}, '(prompt length: pairs)')
```

<!-- Placeholder (prose, ### The Patching Map): the noising sweep, three site kinds at every block and position class, one batched forward per site and prompt length; the checks that a complete cut gives NE 1 and a null patch NE 0. -->

```{.python .input #interpretability-patch}
%%tab pytorch
KINDS = ('resid_post', 'attn_out', 'mlp_out')
SITES = [(kind, l) for kind in KINDS for l in range(L)]
PATTERNS = [(c,) for c in range(8)] + ['all', ()]  # classes, all, none
for g in groups:  # the corrupted run, recorded at every site
    with torch.no_grad(), d2l.intervene(model, record=SITES) as ctx:
        model(g['X_x'].to(dev), logits_to_keep=1)
    g['acts'] = {s: ctx.cache[s][0] for s in SITES}

def masks(cls, patterns):  # (patterns, T): the positions each one patches
    return torch.stack([torch.ones_like(cls, dtype=torch.bool) if p == 'all'
                        else torch.isin(cls, torch.tensor(p, dtype=cls.dtype))
                        for p in patterns])

def patched_ld(site, patterns):
    """LD of each kept pair's clean prompt with the corrupted activation
    written at site, at each pattern's positions: (pairs, patterns)."""
    out, n = torch.empty(len(pairs), len(patterns)), len(patterns)
    for g in groups:
        M = masks(g['cls'], patterns).repeat(len(g['idx']), 1)[..., None]
        M, a_x = M.to(dev), g['acts'][site].repeat_interleave(n, 0)
        with torch.no_grad(), d2l.intervene(
                model, {site: lambda x: torch.where(M, a_x, x)}):
            z = model(g['X'].to(dev).repeat_interleave(n, 0),
                      logits_to_keep=1)[:, -1]
        ab = g['ab'].to(dev).repeat_interleave(n, 0)
        out[g['idx']] = (z.gather(1, ab) @ torch.tensor(
            [1.0, -1.0], device=dev)).view(-1, n).cpu()
    return out

NE = torch.empty(len(KINDS), L, len(PATTERNS), len(pairs))
def sweep():
    for i, site in enumerate(SITES):
        ld = patched_ld(site, PATTERNS)
        NE[i // L, i % L] = ((m_clean[:, None] - ld) / gap[:, None]).T
sweep_time = d2l.Benchmark(sweep, warmup=0, repeats=1).time
assert (NE[0, :, 8] - 1).abs().max() < 1e-3  # all of resid_post: NE = 1
assert NE[:, :, 9].abs().max() < 1e-3  # a null patch: NE = 0
NE = NE[:, :, :8].numpy()

def long_table(T, column):  # (kinds, blocks, classes, pairs) -> CSV bytes
    k, l, c, i = np.indices(T.shape).reshape(4, -1)
    return pd.DataFrame({
        'kind': np.array(KINDS)[k], 'block': l,
        'position': np.array(CLASSES)[c],
        'landmark': np.array([r['landmark'] for r in pairs])[i],
        column: T.ravel()}).to_csv(index=False).encode()

RESULTS['patching.csv'] = long_table(NE, 'ne')
print(f'effect tensor {NE.shape} (kinds, blocks, positions, pairs)')
print(f'{sweep_time:.1f} s for {NE.size:,} patched forwards '
      f'({NE[..., 0].size} per pair)')
```

<!-- Placeholder (prose, ### The Patching Map): single-site effects do not add: at the handover block, NE(subject last) + NE(final) against NE(both). -->

```{.python .input #interpretability-patch-additivity}
%%tab pytorch
S, FINAL = 1, 7  # position classes: the subject's last token, the final one

def handover_of(mean):  # first block where final overtakes subject last
    after = mean[0, :, FINAL] > mean[0, :, S]
    return int(np.argmax(after)) if after.any() else np.nan

mean = NE.mean(-1)  # (kinds, blocks, positions)
handover = handover_of(mean)
both = (m_clean - patched_ld(('resid_post', handover), [(S, FINAL)])[:, 0]
        ) / gap
print(f'block {handover}: NE(subject last) {mean[0, handover, S]:.2f} + '
      f'NE(final) {mean[0, handover, FINAL]:.2f} = '
      f'{mean[0, handover, S] + mean[0, handover, FINAL]:.2f}; '
      f'NE(both) {float(both.mean()):.2f}')
```

<!-- Placeholder (prose, ### The Patching Map): the map below with a city-cluster bootstrap: cells above the declared thresholds, the handover block with its interval, negative cells; caption from spec fig_lm_patching_map. -->

```{.python .input #interpretability-patch-map}
%%tab pytorch
reps = np.stack([NE[..., idx].mean(-1) for idx in cluster_resample(cities)])
lower = pct_ci(reps)[0]
SUMMARY['patching'] = {'handover': [handover, *pct_ci(
    [handover_of(r) for r in reps]).tolist()]}
for i, (kind, t) in enumerate(zip(KINDS, (0.5, 0.5, 0.1))):
    cells = [(int(l), CLASSES[c]) for l, c in zip(*np.nonzero(lower[i] > t))]
    SUMMARY['patching'][kind] = cells
    print(f'{kind}, lower bound above {t}: {cells}')
print(f'handover block {handover}, 95% interval '
      f'{SUMMARY["patching"]["handover"][1:]}')
print(f'{(mean < -0.05).sum()} cells with mean NE below -0.05, '
      f'minimum {mean.min():.2f}')
if FULL is not None:
    print('full run:', FULL['patching'])
d2l.show_heatmaps(torch.tensor(mean).clamp(min=0)[None], 'position', 'block',
                  titles=list(KINDS), figsize=(7, 4.5), xticklabels=CLASSES)
for ax in d2l.plt.gcf().axes[:len(KINDS)]:  # the names read vertically
    ax.tick_params('x', labelrotation=90)
```

<!-- Placeholder (prose, ### Attribution Patching): eq_lm_atp at every site from one clean forward and backward and one corrupted forward, timed against the sweep. -->

```{.python .input #interpretability-atp}
%%tab pytorch
def atp():
    """First-order NE at every site and position class, per group of
    equal-length prompts: a clean forward and backward, a corrupted one."""
    out = torch.empty(len(KINDS), L, 8, len(pairs))
    for g in groups:
        with d2l.intervene(model, record=SITES, record_grad=True) as clean:
            z = model(g['X'].to(dev), logits_to_keep=1)[:, -1]
            (z.gather(1, g['ab'].to(dev)) @ torch.tensor(
                [1.0, -1.0], device=dev)).sum().backward()
        with torch.no_grad(), d2l.intervene(model, record=SITES) as corrupt:
            model(g['X_x'].to(dev), logits_to_keep=1)
        M = masks(g['cls'], PATTERNS[:8]).float().to(dev)  # (8, T)
        for i, s in enumerate(SITES):
            delta = corrupt.cache[s][0] - clean.cache[s][0].detach()
            pe = -(delta * clean.grad_cache[s][0]).sum(-1)  # (pairs, T)
            out[i // L, i % L, :, g['idx']] = (M @ pe.T).cpu() / gap[g['idx']]
    return out.numpy()

atp_time = d2l.Benchmark(atp, warmup=1, repeats=3).time
NE_atp = atp()
RESULTS['atp.csv'] = long_table(NE_atp, 'atp_ne')
print(f'AtP tensor {NE_atp.shape}')
print(f'patching: {sweep_time:.2f} s for {3 * L * 8} forwards per pair; '
      f'AtP: {atp_time:.2f} s for 3 passes; ratio {sweep_time / atp_time:.0f}')
```

<!-- Placeholder (prose, ### Attribution Patching): rank agreement of AtP with exact patching outside the near-zero cells, the error by block band, and the plot below (caption from spec fig_lm_atp_vs_exact). -->

```{.python .input #interpretability-atp-agreement}
%%tab pytorch
def spearman(E, A):  # per kind and pooled, cells with |exact NE| > 0.05
    keep = np.abs(E) > 0.05
    pooled = spearmanr(E[keep], A[keep]).statistic
    return [spearmanr(E[k][keep[k]], A[k][keep[k]]).statistic
            for k in range(len(KINDS))] + [pooled]

exact, approx = NE.mean(-1), NE_atp.mean(-1)
point = spearman(exact, approx)
ci = pct_ci([spearman(NE[..., idx].mean(-1), NE_atp[..., idx].mean(-1))
             for idx in cluster_resample(cities)])
SUMMARY['atp spearman'] = {}
for j, name in enumerate([*KINDS, 'pooled']):
    SUMMARY['atp spearman'][name] = fmt(point[j], ci[:, j], 2)
    print(f'Spearman, {name}: {fmt(point[j], ci[:, j], 2)}')
print('mean absolute error by blocks:', {
    f'{b[0]}-{b[-1]}': round(float(np.abs(approx - exact)[:, b].mean()), 3)
    for b in np.array_split(np.arange(L), 4)})
if FULL is not None:
    print('full run:', FULL['atp spearman'])
ends = [min(exact.min(), approx.min()), max(exact.max(), approx.max())]
d2l.plot([exact[k].ravel() for k in range(3)] + [ends],
         [approx[k].ravel() for k in range(3)] + [ends], 'exact NE',
         'attribution patching NE', legend=list(KINDS),
         fmts=('o', 's', '^', 'k--'), figsize=(4, 4))
```

<!-- Placeholder (stage 2): a top-k sparse autoencoder with the parameter layout of the released files. -->

```{.python .input #interpretability-topk-sae}
%%tab pytorch
#@save
class TopKSAE(nn.Module):
    """Top-k sparse autoencoder with Qwen-Scope's parameter names and
    shapes: W_enc (d_sae, d_model), b_enc, W_dec (d_model, d_sae), b_dec.
    With center=True the encoder subtracts b_dec from its input first.
    seed fixes the random initial dictionary (default: the global RNG)."""
    def __init__(self, d_model, d_sae, k, *, center=False, seed=None):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.center = center  # a plain attribute: the encoder's convention
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        W = torch.randn(d_model, d_sae, generator=gen)
        W = W / W.norm(dim=0, keepdim=True)  # unit decoder columns
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def pre_acts(self, x):
        x = x.to(self.W_enc.dtype)
        return F.linear(x - self.b_dec if self.center else x, self.W_enc,
                        self.b_enc)

    @staticmethod
    def topk_relu(pre, k):
        """ReLU of the k largest entries of each row, zero elsewhere; equal
        values at the cut go to the lower indices."""
        cut = pre.topk(k, dim=-1).values[..., -1:]
        keep = pre >= cut
        ties = keep.sum(-1) > k  # rows where the cut value repeats
        if ties.any():
            p, c = pre[ties], cut[ties]
            above, at = p > c, p == c
            keep[ties] = above | (at & (at.cumsum(-1)
                                        <= k - above.sum(-1, keepdim=True)))
        return torch.where(keep, F.relu(pre), 0.0)

    def encode(self, x):
        return self.topk_relu(self.pre_acts(x), self.k)

    def decode(self, z):
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x, dead=None, k_aux=256):
        """Squared error normalized by the batch variance, and the auxiliary
        loss: the same for the residual x - x_hat reconstructed from the
        k_aux largest pre-activations of the dead features (a mask)."""
        x = x.to(self.W_enc.dtype).reshape(-1, self.d_model)
        pre = self.pre_acts(x)
        x_hat = self.decode(self.topk_relu(pre, self.k))
        nmse = lambda u, v: ((u - v) ** 2).sum(-1).mean() / (
            (v - v.mean(0)) ** 2).sum(-1).mean()
        aux = x_hat.new_zeros(())
        if dead is not None and dead.any():
            e = (x - x_hat).detach()
            z = self.topk_relu(pre.masked_fill(~dead, -torch.inf),
                               min(k_aux, int(dead.sum())))
            aux = nmse(F.linear(z, self.W_dec), e)
        return nmse(x_hat, x), aux

    @torch.no_grad()
    def normalize_decoder_(self):
        """Project each decoder column onto the unit sphere (columns already
        of norm 1 to within 1e-6 are left as they are)."""
        n = self.W_dec.norm(dim=0, keepdim=True)
        self.W_dec.div_(torch.where((n - 1).abs() > 1e-6, n.clamp_min(1e-12),
                                    1.0))

    @torch.no_grad()
    def remove_parallel_grad_(self):
        """Remove from W_dec.grad each column's component along the column,
        the part a unit-norm projection would undo."""
        W, G = self.W_dec, self.W_dec.grad
        if G is not None:
            G.sub_(W * (W * G).sum(0, keepdim=True) / (W * W).sum(
                0, keepdim=True))
```

<!-- Placeholder (stage 2): training the autoencoder on cached activations. -->

```{.python .input #interpretability-train-sae}
%%tab pytorch
#@save
def train_sae(sae, activations, *, steps, batch_size=4096, lr=2e-4,
              aux_coef=1/32, dead_after=200, seed=0):
    """Train a TopKSAE with Adam on random rows of activations, an (N, d)
    CPU tensor (fp16 allowed), with unit decoder columns and the auxiliary
    loss of the features that have not fired for dead_after steps. seed
    fixes the batches; the initial dictionary is the SAE's own (see
    TopKSAE's seed). Returns the loss of each step."""
    dev, dtype = sae.W_dec.device, sae.W_dec.dtype
    gen, N = torch.Generator().manual_seed(seed), len(activations)
    with torch.no_grad():  # b_dec at the data mean, W_enc = W_dec^T
        sample = activations[torch.randperm(N, generator=gen)[:65536]]
        sae.b_dec.copy_(sample.to(dev, dtype).mean(0))
        sae.W_dec.div_(sae.W_dec.norm(dim=0, keepdim=True))
        sae.W_enc.copy_(sae.W_dec.T)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    idle = torch.zeros(sae.d_sae, dtype=torch.long, device=dev)
    losses = []
    for _ in range(steps):
        x = activations[torch.randint(N, (batch_size,), generator=gen)].to(
            dev, dtype)
        mse, aux = sae.loss(x, idle >= dead_after)
        loss = mse + aux_coef * aux
        with torch.no_grad():  # steps since each feature last fired
            fired = (sae.encode(x) > 0).any(0)
            idle = torch.where(fired, 0, idle + 1)
        optimizer.zero_grad()
        loss.backward()
        sae.remove_parallel_grad_()
        optimizer.step()
        sae.normalize_decoder_()
        losses.append(loss.item())
    return losses
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): caching the block-14 stream and training an SAE on it under RETRAIN; the CPU path loads the one trained for Qwen3-0.6B. -->

```{.python .input #interpretability-sae-r1}
%%tab pytorch
def cache_stream(model, windows, layer, batch=32):
    """The stream after block layer at the positions >= 1 of windows,
    without norm outliers (above 10 times the batch median), as one fp16
    tensor in host memory."""
    site, n = ('resid_post', layer), 0
    acts = torch.empty(windows[:, 1:].numel(), model.num_hiddens,
                       dtype=torch.float16)  # filled in place: one copy
    for X in windows.split(batch):
        with torch.no_grad(), d2l.intervene(model, stop_after=site) as ctx:
            model(X.to(dev))
        h = ctx.cache[site][0][:, 1:].flatten(0, 1)
        norms = h.float().norm(dim=-1)
        h = h[norms <= 10 * norms.median()]
        acts[n:n + len(h)] = h.half().cpu()
        n += len(h)
    assert torch.isfinite(acts[:n]).all()
    return acts[:n]

def train_and_save(acts, name):  # TopKSAE(d, 8192, k=50), seed 0, fp16 file
    sae = d2l.TopKSAE(model.num_hiddens, 8192, 50, center=True,
                      seed=0).to(dev)
    losses = d2l.train_sae(sae, acts, steps=SAE_STEPS)
    print(f'loss at steps 0, {SAE_STEPS // 2}, {SAE_STEPS}: ' + ', '.join(
        f'{losses[i]:.3f}' for i in (0, SAE_STEPS // 2, -1)))
    state = {'state_dict': {k: v.half() for k, v in sae.state_dict().items()},
             'config': dict(d_model=sae.d_model, d_sae=sae.d_sae, k=sae.k,
                            center=sae.center), 'tokens': len(acts)}
    save_rerun(name, lambda p: torch.save(state, p))
    return sae, len(acts)

def load_sae(key):
    saved = torch.load(d2l.download_keyed(key), weights_only=True)
    sae = d2l.TopKSAE(**saved['config'])
    sae.load_state_dict(saved['state_dict'])
    print(f'{key}: sha1 {d2l.DATA_HUB[key][1]}')
    return sae.to(dev), saved['tokens']

if RETRAIN:
    sae_r1, r1_tokens = train_and_save(
        cache_stream(model, train_windows, 14), 'lm-sae-r1-l14.pt')
elif SMALL:
    sae_r1, r1_tokens = load_sae('lm-sae-r1-l14-v1')
else:
    print("this run reads Qwen3-1.7B-Base's SAE; R1's is the CPU path's")
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): Qwen3-1.7B-Base replaces Qwen3-0.6B, because the released dictionaries exist for it. -->

```{.python .input #interpretability-load-17b}
%%tab pytorch
if SMALL:
    print('CPU path: Qwen3-0.6B with its own SAE; the release is skipped')
else:
    HOST = 'Qwen/Qwen3-1.7B-Base'
    del model, lens, ctx  # a context keeps the model it instruments
    gc.collect()
    torch.cuda.empty_cache()
    model = d2l.load_qwen3(HOST).requires_grad_(False)  # bf16
    # the same vocabulary, so the token ids of the data cell stay valid
    assert AutoTokenizer.from_pretrained(
        HOST, revision=model.revision).get_vocab() == tok.get_vocab()
    print(f'{HOST} at {model.revision}: '
          f'{sum(p.numel() for p in model.parameters()):,} parameters')
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): under RETRAIN, the block-14 stream of Qwen3-1.7B-Base on the training windows. -->

```{.python .input #interpretability-act-cache}
%%tab pytorch
if RETRAIN and not SMALL:
    acts = cache_stream(model, train_windows, 14)
    print(f'{len(acts):,} positions cached, {acts.numel() * 2 / 1e9:.1f} GB '
          'of host memory')
else:
    print('no cache: the next cell loads the trained SAE')
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): our SAE for Qwen3-1.7B-Base, trained under RETRAIN and loaded otherwise. -->

```{.python .input #interpretability-sae-train-or-load}
%%tab pytorch
if SMALL:
    sae, sae_tokens = sae_r1, r1_tokens
elif RETRAIN:
    sae, sae_tokens = train_and_save(acts, 'lm-sae-q17b-l14.pt')
    del acts
else:
    sae, sae_tokens = load_sae('lm-sae-q17b-l14-v1')
```

<!-- Placeholder (prose, ### A Released Dictionary): Qwen-Scope's Caution paragraph, quoted from the README at the pinned revision, and the size of the release. -->

```{.python .input #interpretability-caution}
%%tab pytorch
SAE_REPO = 'Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'
revision = d2l.HF_REVISIONS[SAE_REPO]
with open(hf_hub_download(SAE_REPO, 'README.md', revision=revision)) as f:
    print(re.search(r'## Caution\n(.*?)\n## ', f.read(), re.S)[1].strip())
sizes = [s.size for s in HfApi().model_info(
    SAE_REPO, revision=revision, files_metadata=True).siblings
         if s.rfilename.endswith('.sae.pt')]
print(f'{len(sizes)} files of {sorted({round(b / 1e6) for b in sizes})} MB;'
      ' this notebook downloads one')
```

<!-- Placeholder (stage 2): one layer of the released dictionary, loaded safely. -->

```{.python .input #interpretability-load-qwen-scope}
%%tab pytorch
#@save
def load_qwen_scope_sae(repo_id, layer, revision=None, *, device=None,
                        dtype=torch.float32, center=False):
    """The TopKSAE of one layer of a Qwen-Scope repository, downloaded
    at the pinned revision and loaded with weights_only=True."""
    from huggingface_hub import hf_hub_download
    revision = revision or d2l.HF_REVISIONS[repo_id]
    get = lambda f: hf_hub_download(repo_id, f, revision=revision)
    with open(get('config.json')) as f:
        cfg = json.load(f)
    if cfg.get('model_type') != 'topk_sae':
        raise ValueError(f'not a top-k SAE: {cfg.get("model_type")!r}')
    if layer not in cfg['layers']:
        raise ValueError(f'{repo_id} has no layer {layer}: {cfg["layers"]}')
    state = torch.load(get(f'layer{layer}.sae.pt'), weights_only=True,
                       map_location='cpu')
    d, m = cfg['d_model'], cfg['d_sae']
    shapes = {'W_enc': (m, d), 'b_enc': (m,), 'W_dec': (d, m), 'b_dec': (d,)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        found = sorted(state) if isinstance(state, dict) else type(state)
        raise KeyError(f'expected the tensors {sorted(shapes)}, found {found}')
    for name, shape in shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f'{name}: shape {tuple(state[name].shape)}, '
                             f'expected {shape}')
    with torch.device('meta'):  # no random initialization to overwrite
        sae = TopKSAE(d, m, cfg['k'], center=center)
    sae.load_state_dict(state, assign=True)
    return sae.to(device=device, dtype=dtype)
```

<!-- Placeholder (stage 2): reconstruction and loss recovered, on the positions the statistics keep. -->

```{.python .input #interpretability-sae-metrics}
%%tab pytorch
#@save
@torch.no_grad()
def sae_metrics(sae, model, token_batches, layer):
    """Reconstruction statistics of sae on the stream after block layer,
    over token_batches (equal-length (B, T) ids), at the positions >= 1
    whose stream norm is at most 10 times the batch median: L0, pooled FVU,
    the next-token cross-entropy of the model, with x_hat spliced in, and
    with the mean activation, loss recovered, the fraction of features
    that never fire, each feature's firing rate, and per-row values:
    sse and sst over the n kept positions, mean ce_orig and ce_splice
    over the n_ce kept positions that have a next token (NaN if none).
    The cross-entropy goes one row at a time, so the fp32 buffers hold
    one row of logits."""
    if sae.W_dec.shape[0] != model.num_hiddens:
        raise ValueError(f'SAE width {sae.W_dec.shape[0]} != model width '
                         f'{model.num_hiddens}')
    site, dev = ('resid_post', layer), model.token_emb.weight.device
    token_batches = list(token_batches)  # read twice

    def ce(logits, X, keep):  # per row: summed loss and count, kept t
        loss = torch.stack([F.cross_entropy(row[:-1].float(), x[1:],
                                            reduction='none')
                            for row, x in zip(logits, X)])
        m = keep[:, :-1]
        return (loss * m).sum(1), m.sum(1)

    keeps, ce_orig, total, n = [], [], 0.0, 0
    for X in token_batches:  # pass 1: the clean run and the kept positions
        X = X.to(dev)
        with d2l.intervene(model, record=[site]) as ctx:
            logits = model(X)
        x = ctx.cache[site][0].float()
        norms = x.norm(dim=-1)
        keep = norms <= 10 * norms[:, 1:].median()
        keep[:, 0] = False
        keeps.append(keep)
        ce_orig.append(ce(logits, X, keep))
        del logits  # not alive during the next forward
        total = total + x[keep].double().sum(0)
        n += int(keep.sum())
    mean = (total / n).float()
    rows = {'sse': [], 'sst': []}
    fired = torch.zeros(sae.d_sae, device=sae.W_dec.device)
    nnz, ce_splice, ce_mean = 0, [], []
    for X, keep in zip(token_batches, keeps):  # pass 2: the two splices
        X, k = X.to(dev), keep[..., None]

        def splice(x):
            nonlocal nnz, fired
            x_hat, z = sae(x)
            xf, x_hat = x.float(), x_hat.float()
            rows['sse'].append((((x_hat - xf) ** 2).sum(-1) * keep).sum(1))
            rows['sst'].append((((xf - mean) ** 2).sum(-1) * keep).sum(1))
            active = (z > 0) & k
            nnz += int(active.sum())
            fired = fired + active.flatten(0, 1).sum(0)
            return torch.where(k, x_hat.to(x.dtype), x)

        with d2l.intervene(model, {site: splice}):
            ce_splice.append(ce(model(X), X, keep))
        with d2l.intervene(model, {site: lambda x: torch.where(
                k, mean.to(x.dtype), x)}):
            ce_mean.append(ce(model(X), X, keep))
    cat = lambda xs: torch.cat(xs).double().cpu().numpy()
    per_row = lambda pairs: cat([s / c for s, c in pairs])  # 0/0: NaN
    pooled = lambda pairs: (sum(float(s.sum()) for s, _ in pairs)
                            / sum(int(c.sum()) for _, c in pairs))
    ce_o, ce_s, ce_m = pooled(ce_orig), pooled(ce_splice), pooled(ce_mean)
    sse, sst = cat(rows['sse']), cat(rows['sst'])
    density = (fired / n).cpu().numpy()
    return {'l0': nnz / n, 'fvu': float(sse.sum() / sst.sum()),
            'ce_orig': ce_o, 'ce_splice': ce_s, 'ce_mean_ablate': ce_m,
            'loss_recovered': (ce_m - ce_s) / (ce_m - ce_o),
            'unfired_frac': float((density == 0).mean()), 'density': density,
            'n_tokens': n,
            'per_doc': {'sse': sse, 'sst': sst,
                        'n': cat([keep.sum(1) for keep in keeps]),
                        'n_ce': cat([c for _, c in ce_orig]),
                        'ce_orig': per_row(ce_orig),
                        'ce_splice': per_row(ce_splice)}}
```

<!-- Placeholder (prose, ### A Released Dictionary): the instrument check, both encoder conventions of the release by reconstruction on held-out text, and the declared rule that decides whether the release is used. -->

```{.python .input #interpretability-sae-instrument}
%%tab pytorch
eval_batches = list(heldout.split(8))  # 8 documents x 256 tokens each
USE_RELEASE = False
if SMALL:
    print('CPU path: no released SAE')
else:
    release = d2l.load_qwen_scope_sae(SAE_REPO, 14, device=dev)
    checks = {}
    for center in (False, True):  # app.py's raw input, then minus b_dec
        release.center = center
        checks[center] = m = d2l.sae_metrics(release, model, eval_batches, 14)
        print(f'center={center}: FVU {m["fvu"]:.3f}, loss recovered '
              f'{m["loss_recovered"]:.3f}, L0 {m["l0"]:.1f}')
    release.center = min(checks, key=lambda c: checks[c]['fvu'])
    m_release = checks[release.center]
    USE_RELEASE = m_release['fvu'] < 0.5 and m_release['loss_recovered'] > 0.5
    print(f'kept center={release.center}; release used: {USE_RELEASE}')
```

<!-- Placeholder (prose, ### A Released Dictionary): the table above (spec tab_lm_sae_compare), ours beside the release as context, not a contest; FVU and loss recovered are different instruments. -->

```{.python .input #interpretability-sae-compare}
%%tab pytorch
m_ours = d2l.sae_metrics(sae, model, eval_batches, 14)
rows = {'ours': (sae.d_sae, sae_tokens, m_ours)}
if not SMALL:  # the release states no token count
    rows['Qwen-Scope'] = (release.d_sae, None, m_release)
COLUMNS = {'L0': 'l0', 'FVU': 'fvu', 'CE orig': 'ce_orig',
           'CE splice': 'ce_splice', 'CE mean-ablate': 'ce_mean_ablate',
           'loss recovered': 'loss_recovered', 'unfired': 'unfired_frac'}
table = pd.DataFrame({name: {
    'width': f'{width:,}',
    'training tokens': 'not stated' if t is None else f'{t:,}',
    **{col: f'{m[key]:.3f}' for col, key in COLUMNS.items()}}
                      for name, (width, t, m) in rows.items()}).T
print(f'{m_ours["n_tokens"]:,} positions of {len(heldout)} documents')
print(table.to_string())
SUMMARY['sae'] = table.to_dict('index')
RESULTS['sae_metrics.json'] = json.dumps({
    name: {k: v for k, v in m.items() if k not in ('density', 'per_doc')}
    for name, (_, _, m) in rows.items()}, indent=1).encode()
if FULL is not None:
    print('full run:', json.dumps(FULL['sae'], indent=1))
```

<!-- Placeholder (prose, ### Features and Their Labels): a feature chosen by contrast (Chinese against English Wikipedia) and two by a density rule, their top contexts, and the contrast feature's label tested as a token rule beside the difference-of-means probe. -->

```{.python .input #interpretability-features}
%%tab pytorch
fsae = (release if USE_RELEASE else sae).requires_grad_(False)
density = (m_release if USE_RELEASE else m_ours)['density']  # held-out
SITE = ('resid_post', 14)

def stream(docs, batch=16):  # the block-14 stream at the positions >= 1
    out = []
    for X in docs.split(batch):
        with torch.no_grad(), d2l.intervene(model, stop_after=SITE) as ctx:
            model(X.to(dev))
        out.append(ctx.cache[SITE][0][:, 1:])
    return torch.cat(out)  # (docs, positions, d)

def fire_rate(H):  # the fraction of positions where each feature fires
    flat = H.flatten(0, 1)
    fired = sum((fsae.encode(h) > 0).sum(0) for h in flat.split(2048))
    return fired / len(flat)

def acts_of(H, feats):  # activations of the features feats, (..., feats)
    return torch.cat([fsae.encode(h)[:, feats] for h in H.flatten(0, 1).split(
        2048)]).view(*H.shape[:2], len(feats)).float()

H = {lang: {part: stream(docs) for part, docs in wiki[lang].items()}
     for lang in wiki}
rate = {lang: fire_rate(H[lang]['contrast']) for lang in H}
top = (rate['zh'] - rate['en']).topk(5).indices
print(pd.DataFrame({'feature': top.tolist(),
                    'rate zh': rate['zh'][top].tolist(),
                    'rate en': rate['en'][top].tolist()}).round(3))
feature = int(top[0])
d_mean = (H['zh']['contrast'].flatten(0, 1).float().mean(0)
          - H['en']['contrast'].flatten(0, 1).float().mean(0))
alive = np.flatnonzero((density >= 1e-3) & (density <= 1e-2))  # by index
FEATS = [feature, *alive[np.random.default_rng(0).permutation(len(alive))[:2]]]
docs = torch.cat([wiki['zh']['contrast'], wiki['en']['contrast'],
                  heldout[:, :128]])
A = torch.cat([acts_of(H[lang]['contrast'], FEATS) for lang in H]
              + [acts_of(stream(heldout[:, :128]), FEATS)])
for j, f in enumerate(FEATS):  # top-8 contexts: 11 tokens, then [token]
    print(f'feature {f}:')
    for i, t in (divmod(int(k), A.shape[1]) for k in A[..., j].ravel().topk(8)
                 .indices):
        before, at = tok.decode(docs[i, max(0, t - 10):t + 1]), tok.decode(
            docs[i, t + 1:t + 2])
        print(f'  {A[i, t, j]:6.1f}  {before}[{at}]'.replace('\n', '\\n'))
# the label as a token rule on held-out documents, beside the probe
label_ids = torch.cat([wiki[lang]['label'][:, 1:] for lang in H])
label = cjk[label_ids].numpy().ravel()  # the token itself holds Han
doc = np.repeat(np.arange(len(label_ids)), label_ids.shape[1])
H_test = torch.cat([H[lang]['label'] for lang in H])
score = {'feature': acts_of(H_test, [feature])[..., 0].cpu().numpy().ravel(),
         'probe': (H_test.float() @ d_mean).cpu().numpy().ravel()}
threshold = {'feature': 0.0, 'probe': float(np.mean(
    [(H[lang]['contrast'].float() @ d_mean).mean().item() for lang in H]))}

def auc(s, y):  # the Mann-Whitney statistic
    n = y.sum()
    return (rankdata(s)[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n))

reps = np.array([[auc(s[idx], label[idx]) for s in score.values()]
                 for idx in cluster_resample(doc)])
results = {}
for j, (name, s) in enumerate(score.items()):
    active = s > threshold[name]
    precision = d2l.bootstrap_ci(label[active], cluster=doc[active])
    recall_ = d2l.bootstrap_ci(active[label], cluster=doc[label])
    results[name] = [fmt(precision.mean, precision.ci),
                     fmt(recall_.mean, recall_.ci),
                     fmt(auc(s, label), pct_ci(reps[:, j]))]
print(pd.DataFrame(results, index=['precision', 'recall', 'AUC']).T)
d_auc = fmt(auc(score['feature'], label) - auc(score['probe'], label),
            pct_ci(reps[:, 0] - reps[:, 1]))
print(f'AUC, feature minus probe: {d_auc}')
print(f'features never firing: {(density == 0).sum():,}; '
      f'dense (above 0.1): {(density > 0.1).sum():,}')
SUMMARY['label test'] = {**results, 'AUC difference': d_auc}
```

<!-- Placeholder (prose, ### Steering Directions): three directions at block 14 (the feature's decoder column, the difference of means, random ones), and the median stream norm that makes their strengths comparable. -->

```{.python .input #interpretability-directions}
%%tab pytorch
directions = {
    'SAE feature': F.normalize(fsae.W_dec[:, feature].float(), dim=0),
    'difference of means': F.normalize(d_mean, dim=0)}
for seed in range(SIZES['random']):
    g = torch.Generator().manual_seed(seed)
    directions[f'random {seed}'] = F.normalize(
        torch.randn(model.num_hiddens, generator=g), dim=0).to(dev)
INFORMED, RANDOM = list(directions)[:2], list(directions)[2:]
n_bar = float(stream(cost_docs).float().norm(dim=-1).median())
cos = lambda u, v: float(u @ v)
print(f'median stream norm on English text: {n_bar:.1f}')
print(f'cosine of the feature and the difference of means: '
      f'{cos(*(directions[n] for n in INFORMED)):.2f}')
worst = max(abs(cos(directions[r], directions[i]))
            for r in RANDOM for i in INFORMED)
print(f'largest |cosine| of a random direction with either: {worst:.3f}')
if not SMALL:  # its decoder column is all we keep of the release
    del release, fsae
    gc.collect()
    torch.cuda.empty_cache()
```

<!-- Placeholder (prose, ### The Steering Frontier): eq_lm_steer on a grid of strengths for each direction, and a fixed Chinese prefix that changes no activation: behaviour (switched prompts) and cost (cross-entropy on English text). -->

```{.python .input #interpretability-steer}
%%tab pytorch
ALPHAS = ([0.0, 1.0, 4.0] if SMALL else
          [0.0] + [2 ** (j / 2) for j in range(-4, 7)])  # 0.25 to 8
PREFIX = '下面的文字将用中文继续。\n\n'  # declared before any run, not tuned
SWITCH = 0.3  # a continuation switched: at least 30% Han characters
for prefix in ('', PREFIX):  # one bucket of equal-length prompts each
    lengths = {len(ids_of(prefix + t)) for t in steer_texts}
    assert len(lengths) == 1
    print(f'one bucket: {len(steer_texts)} prompts of {lengths.pop()} tokens')
cjk_dev = cjk.to(dev)

def steer(v):  # adds v at every position but the first
    def fn(x):
        if x.shape[1] == 1:  # a decode step
            return x + v.to(x.dtype)
        return torch.cat([x[:, :1], x[:, 1:] + v.to(x.dtype)], 1)
    return fn

def steering(fn):
    return contextlib.nullcontext() if fn is None else d2l.intervene(
        model, {SITE: fn})

def han_fraction(text):
    chars = [c for c in text if not c.isspace()]
    return sum(CJK.match(c) is not None for c in chars) / max(len(chars), 1)

@torch.no_grad()
def doc_ce(prefix_ids, batch=16):
    """Per cost document, the mean cross-entropy of its tokens 1 to 128,
    plain and renormalized over the tokens without Han characters."""
    P, out = len(prefix_ids), []
    X = torch.cat([torch.tensor(prefix_ids, dtype=torch.long).expand(
        len(cost_docs), P), cost_docs], 1)
    for Xb in X.split(batch):  # 16 x 129 tokens: about 2,048 per chunk
        Xb = Xb.to(dev)
        z, y = model(Xb)[:, P:-1].float(), Xb[:, P + 1:]
        z_y = z.gather(-1, y[..., None])[..., 0]
        ce = z.logsumexp(-1) - z_y
        ce_rest = z.masked_fill(cjk_dev, -torch.inf).logsumexp(-1) - z_y
        rest = ~cjk_dev[y]
        out.append(torch.stack([ce.mean(1), (ce_rest * rest).sum(1)
                                / rest.sum(1)], 1).cpu())
    return torch.cat(out).numpy()  # (documents, 2)

def run(fn=None, prefix=''):  # Han fractions per prompt, CE per document
    with steering(fn):
        gens = d2l.generate(model, tok, [prefix + t for t in steer_texts],
                            max_new_tokens=48, batch_size=len(steer_texts))
        return (np.array([han_fraction(g.text) for g in gens]),
                doc_ce(ids_of(prefix)))

base, runs = run(), {}
for name, d in directions.items():
    runs[name, 0.0] = base  # alpha = 0 is one shared run
    for alpha in ALPHAS[1:]:
        runs[name, alpha] = run(steer(alpha * n_bar * d))
    print(f'{name}: switched at alpha 1: '
          f'{(runs[name, 1.0][0] >= SWITCH).mean():.2f}')
runs['prefix', 0.0] = run(prefix=PREFIX)
print(f'prefix: switched {(runs["prefix", 0.0][0] >= SWITCH).mean():.2f}')
RESULTS['steering_prompts.csv'] = pd.DataFrame(
    [(n, a, i, h) for (n, a), (hs, _) in runs.items()
     for i, h in enumerate(hs)],
    columns=['direction', 'alpha', 'prompt', 'han_fraction']
).to_csv(index=False).encode()
RESULTS['steering_docs.csv'] = pd.DataFrame(
    [(n, a, i, *c) for (n, a), (_, cs) in runs.items()
     for i, c in enumerate(cs)],
    columns=['direction', 'alpha', 'doc', 'ce', 'ce_without_han']
).to_csv(index=False).encode()
```

<!-- Placeholder (prose, ### The Steering Frontier): the frontier plot below (caption from spec fig_lm_steering_frontier), areas up to the cost cap with paired intervals, the random-direction floor, and the prefix against each direction at its own cost. -->

```{.python .input #interpretability-frontier}
%%tab pytorch
CAP = 0.5  # nats per token on English text
SW = {n: np.array([runs[n, a][0] >= SWITCH for a in ALPHAS])
      for n in directions}  # (alphas, prompts)
CE = {n: np.array([runs[n, a][1] - base[1] for a in ALPHAS])
      for n in directions}  # (alphas, documents, plain or without Han)
sw_p = runs['prefix', 0.0][0] >= SWITCH
ce_p = runs['prefix', 0.0][1] - base[1]

def envelope(cost, sw):  # the most switched at each cost or less
    order = np.argsort(cost, kind='stable')
    return np.clip(cost[order], 0, None), np.maximum.accumulate(sw[order])

def area(cost, sw):  # under the frontier, from cost 0 to CAP
    grid = np.linspace(0, CAP, 101)
    return np.trapezoid(np.interp(grid, *envelope(cost, sw)), grid)

def stats(ip, idoc):  # the declared statistics on resampled prompts, docs
    m = {n: (CE[n][:, idoc, 0].mean(1), SW[n][:, ip].mean(1))
         for n in directions}
    a = [area(*m[n]) for n in INFORMED]
    excess = max(m[n][1][np.flatnonzero(m[n][0] < CAP).max()]
                 for n in RANDOM) - m[RANDOM[0]][1][0]
    cp, sp = ce_p[idoc, 0].mean(), sw_p[ip].mean()
    return [*a, a[0] - a[1], excess,
            *(sp - np.interp(cp, *envelope(*m[n])) for n in INFORMED)]

n_p, n_d = len(steer_texts), len(cost_docs)
point = stats(np.arange(n_p), np.arange(n_d))
reps = np.array([stats(ip, idoc) for ip, idoc in zip(
    cluster_resample(range(n_p)), cluster_resample(range(n_d), seed=1))])
ci = pct_ci(reps)
names = [f'area, {INFORMED[0]}', f'area, {INFORMED[1]}',
         'area difference', 'random excess',
         *(f'prefix minus {n}' for n in INFORMED)]
SUMMARY['steering'] = {n: fmt(v, ci[:, j]) for j, (n, v) in
                       enumerate(zip(names, point))}
SUMMARY['steering']['random excess, upper 95% bound'] = float(
    np.quantile(reps[:, 3], 0.95))
for n, v in SUMMARY['steering'].items():
    print(f'{n}: {v}')
if FULL is not None:
    print('full run:', json.dumps(FULL['steering'], indent=1))
cost = {n: CE[n][..., 0].mean(1) for n in directions}
sw = np.array([SW[n].mean(1) for n in RANDOM])
c_rand = np.mean([cost[n] for n in RANDOM], 0)
d2l.plot([cost[INFORMED[0]], cost[INFORMED[1]], c_rand, c_rand,
          [ce_p[:, 0].mean()], c_rand, [CAP, CAP]],
         [SW[INFORMED[0]].mean(1), SW[INFORMED[1]].mean(1), sw.mean(0),
          sw.min(0), [sw_p.mean()], sw.max(0), [0, 1]],
         'cross-entropy increase on English (nats per token)',
         'fraction switched to Chinese',
         legend=[*INFORMED, f'random (mean of {len(RANDOM)})',
                 'random (min, max)', 'prefix'],
         fmts=('C0o-', 'C1s-', 'C2x-', 'C2:', 'C3*', 'C2:', 'k--'),
         figsize=(5, 3.5))
RESULTS['summary.json'] = json.dumps(SUMMARY, indent=1, default=float,
                                     ensure_ascii=False).encode()
if RETRAIN:
    save_rerun('lm-interp-results.zip', write_results)
```

<!-- Placeholder (prose, ### The Steering Frontier): continuations at each informed direction's operating point, beside the unsteered model and the prefix, including the Eiffel prompt. -->

```{.python .input #interpretability-steer-samples}
%%tab pytorch
samples = [*steer_texts[:3], eiffel]
conditions = {'unsteered': (None, '', 0.0),
              'prefix': (None, PREFIX, ce_p[:, 1].mean())}
for n in INFORMED:  # the smallest alpha that switches half below the cap
    below = cost[n] < CAP
    hit = np.flatnonzero(below & (SW[n].mean(1) >= 0.5))
    i = hit[0] if len(hit) else np.flatnonzero(below).max()
    print(f'{n}: alpha {ALPHAS[i]:g}'
          + ('' if len(hit) else ' (the largest below the cap; no half)'))
    conditions[n] = (steer(ALPHAS[i] * n_bar * directions[n]), '',
                     CE[n][i, :, 1].mean())
for name, (fn, prefix, renorm) in conditions.items():
    with steering(fn):
        gens = d2l.generate(model, tok, [prefix + s for s in samples],
                            max_new_tokens=48)
    print(f'--- {name} (cost without Han: {renorm:+.3f} nats per token)')
    for s, g in zip(samples, gens):
        d2l.print_wrapped(f'...{s[-30:]}|{g.text}'.replace('\n', ' '))
```
