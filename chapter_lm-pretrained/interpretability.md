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
import io
import itertools
import json
import os
import pandas as pd
import re
import unicodedata
import zipfile
from huggingface_hub import HfApi, hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoTokenizer
```

<!-- Placeholder (prose, opening): the flags of the CPU path and of regeneration (D2L_RERUN names the artifacts to regenerate: the tuned lens, the two SAEs, the results archive; 1 means all four); Qwen3-0.6B in fp32, as loaded in the d2l-checkpoint section. The plumbing (Hub range reads, the cluster bootstrap, the regeneration bookkeeping) is library code in interpretability-lib.md, not shown. -->

```{.python .input #interpretability-setup}
%%tab pytorch
SMALL = not torch.cuda.is_available()  # the CPU path: Qwen3-0.6B only
ARTIFACTS = {'tuned-lens': 'lm-tuned-lens-qwen3-0.6b-v1',
             'sae-06b': 'lm-sae-r1-l14-v1', 'sae-17b': 'lm-sae-q17b-l14-v1',
             'results': 'lm-interp-results-v3'}
RERUN = d2l.lm_rerun(ARTIFACTS)  # D2L_RERUN: the artifacts to regenerate
assert not (SMALL and RERUN), 'regenerating the artifacts needs a GPU'
SIZES = (dict(pairs=10, heldout=32, steer=20, cost=20, contrast=60,
              label=40, words=(5_000, 2_000, 2_000), random=2,
              wds=(1e-3, 1e-2)) if SMALL else
         dict(pairs=40, heldout=200, steer=100, cost=200, contrast=300,
              label=100, words=(20_000, 5_000, 5_000), random=5,
              wds=(1e-4, 1e-3, 1e-2, 1e-1)))
TRAIN_TOKENS, LENS_STEPS, SAE_STEPS = 8_000_000, 250, 5_000  # RERUN only
RESULTS = {}  # the members of the results archive, by name
SUMMARY = {'config': {'small': SMALL}}  # its summary.json
torch.manual_seed(0)
MODEL_ID = 'Qwen/Qwen3-0.6B'
model = d2l.load_qwen3(MODEL_ID, dtype=torch.float32).requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model.revision)
dev, L = model.token_emb.weight.device, len(model.blks)
print(f'SMALL={SMALL}, regenerating: {", ".join(sorted(RERUN)) or "none"}, '
      f'device {dev}')
print(f'{MODEL_ID} at {model.revision}: '
      f'{sum(p.numel() for p in model.parameters()):,} parameters')
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

<!-- Placeholder (stage 2): the probe's treebank, pinned to a commit, the recall pairs, and the files of the regeneration run (the tuned lens, the two SAEs, the results archive). -->

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
# What the regeneration run (D2L_RERUN) trained and measured
d2l.DATA_HUB['lm-tuned-lens-qwen3-0.6b-v1'] = (
    d2l.LM_DATA_URL + 'lm-tuned-lens-qwen3-0.6b-v1.pt',
    'e1a9275950de634b8f8a6e7dd516de662755834e')
d2l.DATA_HUB['lm-sae-r1-l14-v1'] = (
    d2l.LM_DATA_URL + 'lm-sae-r1-l14-v1.pt',
    'b7d9589fda439d7bdd9e52572a24c6ea58efb263')
d2l.DATA_HUB['lm-sae-q17b-l14-v1'] = (
    d2l.LM_DATA_URL + 'lm-sae-q17b-l14-v1.pt',
    '11982ca32f0382eb9bf594faee58e808d61be12d')
d2l.DATA_HUB['lm-interp-results-v3'] = (
    d2l.LM_DATA_URL + 'lm-interp-results-v3.zip',
    '75606ff34dc535f0705dfeb9194e4791c2820484')
```

<!-- Placeholder (prose, ### The Residual Stream as a Sum): the data of the section, each subset resolved once by a declared rule (FineWeb-Edu, Chinese and English Wikipedia, the recall pairs), and the Han-character tokens of the vocabulary. Wikipedia files are ordered by article id and neighbours share topics (hundreds of consecutive biographies of generals), so the documents come from 20 blocks spread over each file. The held-out label documents come from 20 other Wikipedia blocks, halfway between the contrast blocks, so they are other articles on other topics, not the contrast documents' neighbours. -->

```{.python .input #interpretability-data}
%%tab pytorch
FINEWEB = ('HuggingFaceFW/fineweb-edu', 'sample/10BT/013_00000.parquet',
           '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
WIKI = {lang: ('wikimedia/wikipedia', f'20231101.{lang}/{shard}.parquet',
               'b04c8d1ceb2f5cd4588862100d08de323dccfbaa')
        for lang, shard in (('zh', 'train-00002-of-00006'),
                            ('en', 'train-00028-of-00041'))}
STRIDE = {'zh': 11_000, 'en': 7_500}  # 20 blocks spread over each shard
HAN = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')  # ideographs
ids_of = lambda text: tok(text, add_special_tokens=False).input_ids

def first_tokens(n, min_len):  # the rule: a document's first n tokens
    return lambda ids: ids[:n] if len(ids) >= min_len else None

def clean_prompt(ids):  # 32 tokens without Han that survive a round trip
    text = tok.decode(ids[:32])
    keep = len(ids) >= 32 and not HAN.search(text) and ids_of(text) == ids[:32]
    return ids[:32] if keep else None

def walk(label, source, rows, subsets):  # the first qualifying documents
    docs, seen = d2l.lm_walk_docs(ids_of, source, rows, subsets)
    k = seen['blocks']
    print(f'{label}: ' + ', '.join(f'{name} {len(d)}' for name, d in
                                   docs.items())
          + f' docs from rows {seen["rows"][0]}-{seen["rows"][1]} '
          f'({k} block{"s" * (k > 1)}, {seen["nbytes"] / 1e6:.1f} MB read)')
    return docs

fineweb = walk('FineWeb-Edu', FINEWEB, itertools.count(8000), {
    'held-out': (SIZES['heldout'], first_tokens(256, 257)),
    'steering': (SIZES['steer'], clean_prompt),
    'cost': (SIZES['cost'], first_tokens(129, 129))})
wiki = {lang: {
    **walk(f'Wikipedia {lang}', WIKI[lang], d2l.lm_round_robin(STRIDE[lang]),
           {'contrast': (SIZES['contrast'], first_tokens(128, 129))}),
    **walk(f'Wikipedia {lang}', WIKI[lang],
           d2l.lm_round_robin(STRIDE[lang], first=STRIDE[lang] // 2),
           {'label': (SIZES['label'], first_tokens(128, 129))})}
        for lang in WIKI}
heldout, cost_docs = fineweb['held-out'], fineweb['cost']
steer_texts = tok.batch_decode(fineweb['steering'])
pieces = tok.batch_decode([[v] for v in range(len(tok))])  # each token alone
han_tokens = torch.zeros(model.token_emb.num_embeddings, dtype=torch.bool)
han_tokens[:len(tok)] = torch.tensor([HAN.search(p) is not None
                                      for p in pieces])
print(f'{int(han_tokens.sum()):,} of {len(tok):,} tokens hold a Han '
      'character')
with open(d2l.download_keyed('lm-interp-recall-v1')) as f:
    recall = json.load(f)['rows'][:SIZES['pairs']]
if RERUN - {'results'}:  # every 256-token window of rows 0 to 7,999
    texts, nbytes = d2l.lm_hub_rows(*FINEWEB, 0, 8000)
    windows = [ids[i:i + 256] for ids in ids_of(texts)
               for i in range(0, len(ids) - 255, 256)]
    train_windows = torch.tensor(windows[:TRAIN_TOKENS // 256])
    print(f'training windows: {train_windows.numel():,} tokens, '
          f'{nbytes / 1e6:.1f} MB read')
FULL = None  # the full run's summary, shown beside the CPU path's numbers
if SMALL:
    with zipfile.ZipFile(d2l.lm_artifact(ARTIFACTS['results'])) as z:
        FULL = json.loads(z.read('summary.json'))
    assert not FULL['config']['small'], 'the archive of a CPU-path run'
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
        self.model, self.record_grad = model, record_grad
        self.sites, self.record, self.stop_after = d2l._intervene_args(
            model, sites, record, stop_after)
        self.cache = {s: [] for s in self.record}
        self.wanted = set(self.sites) | self.record

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
    """Context manager over a d2l.GPT that overwrites the sites (kind,
    layer) -> fn(x) and records the sites in record."""
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
                   columns=[f'block {l}' for l in range(0, L, 4)]).to_string(
    float_format='{:.1f}'.format))
```

<!-- Placeholder (prose, opening and ### Logit Lens and Tuned Lens): the opening picture, the final top-5 for the Eiffel prompt and the logit lens's top-1 token and the rank of " Paris" at every fourth block. -->

```{.python .input #interpretability-eiffel}
%%tab pytorch
eiffel = recall[0]['template'].format(recall[0]['landmark'])
paris = ids_of(' Paris')[0]
with torch.no_grad(), d2l.intervene(model, record=blocks) as ctx:
    p = F.softmax(model(torch.tensor([ids_of(eiffel)], device=dev))[0, -1], -1)
top5 = p.topk(5)
print(eiffel)
print(pd.DataFrame({
    'next token': map(repr, tok.batch_decode(top5.indices[:, None])),
    'probability': top5.values.tolist()}).round(3).to_string(index=False))
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
        """Adam on KL(final || lens) summed over the layers, at a random
        fraction frac of the positions >= 1 of each (B, T) batch of ids;
        returns the loss of each step."""
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
            # each layer's loss is backpropagated alone, chunk positions at
            # a time: the memory holds one chunk of logits
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

<!-- Placeholder (prose, ### Logit Lens and Tuned Lens): the translators are fitted on the first million training tokens under D2L_RERUN; a normal build loads the committed ones. -->

```{.python .input #interpretability-tuned-lens-fit}
%%tab pytorch
lens = d2l.TunedLens(model.num_hiddens, L).to(dev)
if 'tuned-lens' in RERUN:  # LENS_STEPS batches of 16 windows: 1M tokens
    losses = lens.fit(model, train_windows[:16 * LENS_STEPS].split(16),
                      steps=LENS_STEPS)
    print(f'KL summed over blocks: {losses[0]:.2f} at the first step, '
          f'{losses[-1]:.2f} at the last')
    state = {k: v.half() for k, v in lens.state_dict().items()}
    d2l.lm_save_rerun('lm-tuned-lens-qwen3-0.6b.pt',
                      lambda p: torch.save(state, p))
    lens.load_state_dict(state)  # go on with the fp16 values it publishes
else:
    key = ARTIFACTS['tuned-lens']
    lens.load_state_dict(torch.load(d2l.download_keyed(key),
                                    map_location='cpu', weights_only=True))
    print(f'{key}: sha1 {d2l.DATA_HUB[key][1]}')
lens.requires_grad_(False);
```

<!-- Placeholder (prose, ### Logit Lens and Tuned Lens): the keep rule of the recall pairs, and the rank of the correct city by block under both lenses (scale-free): the first block where the median rank is at most 10, with a city-cluster interval. -->

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
    key = f'{name}: first block of median rank <= 10'
    lo, hi = d2l.percentile_ci([first10(R[idx]) for idx in
                                d2l.cluster_resample(cities)], names=[key])
    SUMMARY[key] = [float(first10(R)), lo, hi]
    print(f'{name}: median rank at most 10 from block {first10(R)}, '
          f'95% interval [{lo:.0f}, {hi:.0f}]')
    if FULL is not None:
        print(f'  full run: {FULL[key]}')
```

<!-- Placeholder (prose, ### Logit Lens and Tuned Lens): the KL to the final distribution on held-out text (with the scale), with document-cluster intervals; the plot below, caption from spec fig_lm_lens_rank (the 95% bands of panel (b) are narrower than the lines, and the dashed lines of panel (a) are the quartiles: the caption says both). -->

```{.python .input #interpretability-lens-kl}
%%tab pytorch
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
kl_doc = np.repeat(np.arange(len(heldout)), 127)

def mean_ci(values):  # over positions, resampling whole documents
    r = d2l.bootstrap_ci(torch.cat(values).cpu().numpy(), cluster=kl_doc)
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

<!-- Placeholder (prose, ### Linear Probes and Control Tasks): UD English EWT, every 10th training and every 5th dev and test sentence; each word read at its last token, from the embedding and after block 14. "Unseen" counts word types (about half); by word occurrence the share is about a fifth: never write "47% of test words". -->

```{.python .input #interpretability-probe-data}
%%tab pytorch
ud = {split: d2l.read_conllu(d2l.download_keyed(f'lm-ud-ewt-{split}'),
                             stride, n)
      for split, stride, n in zip(('train', 'dev', 'test'), (10, 5, 5),
                                  SIZES['words'])}
feats = {split: d2l.lm_word_features(model, tok, s, 14)
         for split, s in ud.items()}
words = {split: [w for _, ws in s for w in ws] for split, s in ud.items()}
sentence_of = np.repeat(np.arange(len(ud['test'])),
                        [len(ws) for _, ws in ud['test']])
for split, s in ud.items():
    print(f'{split}: {len(words[split]):,} words, {len(s):,} sentences, '
          f'{len({w[2] for w in words[split]}):,} word types')
test_types = {w[2] for w in words['test']}
unseen = test_types - {w[2] for w in words['train']}
print(f'unseen in training: {len(unseen) / len(test_types):.2f} of the test '
      f'word types, {np.mean([w[2] in unseen for w in words["test"]]):.2f} '
      'of the test words')
```

<!-- Placeholder (prose, ### Linear Probes and Control Tasks): linear probes for the tags and for five control tasks (one random tag per word type), with the majority tag per type as baseline; the table above (spec tab_lm_probe_control) and the paired selectivity difference. -->

```{.python .input #interpretability-probe}
%%tab pytorch
UPOS = sorted({w[3] for ws in words.values() for w in ws})
TYPES = sorted({w[2] for ws in words.values() for w in ws})  # all splits
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
    fits = {wd: fit_probe(X['train'], Y['train'], wd) for wd in SIZES['wds']}
    wd = max(fits, key=lambda w: float(correct(fits[w], 'dev').float().mean()))
    return correct(fits[wd], 'test').cpu().numpy(), (*fits[wd], mu, sd)

def majority(y):  # the most frequent train tag of each word type
    pred = d2l.type_tag_baseline([w[2] for w in words['train']], y['train'],
                                 [w[2] for w in words['test']])
    return pred == y['test'], None

controls = [control(seed) for seed in range(5)]
probe_table, selectivity, weights = {}, {}, {}
for name, f in (('majority tag per type', majority),
                ('embedding', lambda y: probe('embedding', y)),
                ('block 14', lambda y: probe('block 14', y))):
    task, weights[name] = f(labels)
    ctrl = np.mean([f(c)[0] for c in controls], 0)
    selectivity[name] = task.astype(float) - ctrl
    probe_table[name] = [d2l.fmt_ci(r.mean, r.ci) for r in (
        d2l.bootstrap_ci(v, cluster=sentence_of)
        for v in (task, ctrl, selectivity[name]))]
probe_table = pd.DataFrame(probe_table, index=[
    'task accuracy', 'control accuracy', 'selectivity']).T
print(probe_table.to_string())
top_tag = int(freq.argmax())  # the majority class, from the training tags
majority_rate = float((labels['test'] == top_tag).mean())
print(f'majority class {UPOS[top_tag]}: {majority_rate:.3f} of test words')
diff = d2l.paired_bootstrap(selectivity['block 14'], selectivity['embedding'],
                            cluster=sentence_of)
print(f'selectivity, block 14 minus embedding: {d2l.fmt_ci(diff.mean, diff.ci)}')
SUMMARY['probe'] = {**probe_table.to_dict('index'),
                    'difference': d2l.fmt_ci(diff.mean, diff.ci),
                    'majority class rate': round(majority_rate, 3)}
buffer = io.BytesIO()
torch.save({n: weights[n] for n in ('embedding', 'block 14')}, buffer)
RESULTS['probe.pt'] = buffer.getvalue()
if FULL is not None:
    print('full run:', json.dumps(FULL['probe'], indent=1))
```

<!-- Placeholder (prose, ### Clean and Corrupted Runs): the recall pairs the model answers, their cities, the LD on clean and corrupted prompts, and the eight position classes of each prompt. -->

```{.python .input #interpretability-recall-pairs}
%%tab pytorch
assert int(p.argmax()) == paris  # without a chat template: " Paris"
assert kept[0] and recall[0]['corrupt_landmark'] == 'Colosseum'
pairs = [r for r, k in zip(recall, kept) if k]
assert SMALL or (len(pairs) >= 30 and len(set(cities)) >= 15)
dropped = [r['landmark'] for r in recall if r not in pairs]
print(f'{len(pairs)} of {len(recall)} pairs kept, {len(set(cities))} cities'
      + (f'; discarded: {", ".join(dropped)}' if dropped else ''))
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
print('pairs per prompt length: ' + ', '.join(
    f'{T} tokens {len(by_length[T])}' for T in sorted(by_length)))
```

<!-- Placeholder (prose, ### The Patching Map): the noising sweep, three site kinds at every block and position class, one batched forward per site and prompt length; the checks that a complete cut gives NE 1 and a null patch NE 0. -->

```{.python .input #interpretability-patch}
%%tab pytorch
KINDS = ('resid_post', 'attn_out', 'mlp_out')
SITES = [(kind, l) for kind in KINDS for l in range(L)]
PATTERNS = [(c,) for c in range(8)]  # one per position class
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

def ne(site, patterns):  # (patterns, pairs)
    return ((m_clean[:, None] - patched_ld(site, patterns)) / gap[:, None]).T

NE = torch.empty(len(KINDS), L, len(PATTERNS), len(pairs))
def sweep():
    for i, site in enumerate(SITES):
        NE[i // L, i % L] = ne(site, PATTERNS)
sweep_time = d2l.Benchmark(sweep, warmup=0, repeats=1).time
NE = NE.numpy()
for site in SITES:  # the checks, outside the timed sweep
    if site[0] == 'resid_post':  # all positions of the stream: NE = 1
        assert (ne(site, ['all']) - 1).abs().max() < 1e-3
    assert ne(site, [()]).abs().max() < 1e-3  # a null patch: NE = 0
AXES = [('kind', KINDS), ('block', range(L)), ('position', CLASSES),
        ('landmark', [r['landmark'] for r in pairs])]
RESULTS['patching.csv'] = d2l.long_csv(NE, AXES, 'ne')
print(f'effect tensor {NE.shape} (kinds, blocks, positions, pairs)')
print(f'{NE.size:,} patched forwards, {NE[..., 0].size} per pair')
```

<!-- Placeholder (prose, ### The Patching Map): additivity at the handover block, NE(subject last) + NE(final) against NE(both). The two-site toy shows that effects need not add; on this pair at the handover block they nearly do (review 18.6: 0.11 + 0.55 = 0.66 against 0.68): never title it "effects do not add". -->

```{.python .input #interpretability-patch-additivity}
%%tab pytorch
SUBJ_LAST, FINAL = 1, 7  # position classes: subject's last token, final one

def handover_of(mean):  # first block where final overtakes subject last
    after = mean[0, :, FINAL] > mean[0, :, SUBJ_LAST]
    return int(np.argmax(after)) if after.any() else np.nan

mean = NE.mean(-1)  # (kinds, blocks, positions)
handover = handover_of(mean)
both = (m_clean - patched_ld(('resid_post', handover),
                             [(SUBJ_LAST, FINAL)])[:, 0]) / gap
print(f'block {handover}: NE(subject last) {mean[0, handover, SUBJ_LAST]:.2f}'
      f' + NE(final) {mean[0, handover, FINAL]:.2f} = '
      f'{mean[0, handover, SUBJ_LAST] + mean[0, handover, FINAL]:.2f}; '
      f'NE(both) {float(both.mean()):.2f}')
```

<!-- Placeholder (prose, ### The Patching Map): the map below with a city-cluster bootstrap: cells above the declared thresholds, the handover block with its interval (block 20 in every replicate: say so; the percentile interval is narrow), attn_out at the final position at the handover block (one attention layer moves the fact), negative cells (the one below -0.05 is mlp_out at block 27, final position: consistent with final-layer suppression, mention it); caption from spec fig_lm_patching_map (the colour is mean NE). Reading (verify 18.6, review 18.6): the subject-last column stays near 0.4 through blocks 0-19 and only the 0.5 threshold hides it: never write that there is no subject-last band; write that the subject's effect splits across its tokens (both subject classes together NE about 1) and neither class alone passes 0.5. Per pair, before the handover the subject's effect stays on the token or tokens that name the landmark: on the earlier tokens when the last word is generic (Bolshoi Theatre, Shibuya Crossing), on the last token when it is the distinctive one (Big Ben, Pantheon). The information moves to the final position at the handover. Do not write that the earlier tokens carry "the other half" of a per-prompt split; the class means average two kinds of pairs. Name the pitfall: a position class pools tokens with different roles. Single-layer noising shows no mid-depth MLP band (mlp_out passes 0.1 only at blocks 0-2 and 21-22): explain it by single-layer noising, redundancy and a different corruption, not as a refutation of causal tracing. -->

```{.python .input #interpretability-patch-map}
%%tab pytorch
reps = np.stack([NE[..., idx].mean(-1)
                 for idx in d2l.cluster_resample(cities)])
lower = d2l.percentile_ci(reps)[0]
SUMMARY['patching'] = {'handover': [handover, *d2l.percentile_ci(
    [handover_of(r) for r in reps], names=['handover']).tolist()]}
for i, (kind, t) in enumerate(zip(KINDS, (0.5, 0.5, 0.1))):
    cells = [(int(l), CLASSES[c]) for l, c in zip(*np.nonzero(lower[i] > t))]
    SUMMARY['patching'][kind] = cells
    print(f'{kind}, lower bound above {t}: {d2l.block_runs(cells, CLASSES)}')
print('handover block {}, 95% interval [{:.0f}, {:.0f}]'.format(
    *SUMMARY['patching']['handover']))
attn = mean[1, :, FINAL]  # attn_out at the final position, per block
lo, hi = d2l.percentile_ci(reps[:, 1, handover, FINAL])
others = [l for l in np.argsort(-attn) if l != handover][:2]
SUMMARY['patching']['attn_out, final, handover'] = [
    float(attn[handover]), lo, hi]
print(f'attn_out at the final position, block {handover}: '
      f'{d2l.fmt_ci(attn[handover], (lo, hi), 2)}')
print('  next highest: ' + ', '.join(f'block {l} {attn[l]:.2f}'
                                     for l in others))
early, last = NE[0, :handover, :SUBJ_LAST + 1].mean(0)  # per pair
share = last / (early + last)
order = np.argsort(share)
q = np.percentile(early + last, [25, 50, 75])
print(f'resid_post, blocks 0-{handover - 1}, per pair: the subject classes '
      f'together NE {q[1]:.2f} (quartiles {q[0]:.2f}, {q[2]:.2f})')
for where, idx in (('earlier tokens', order[share[order] < 1 / 3]),
                   ('last token', order[share[order] > 2 / 3][::-1])):
    print(f'  over 2/3 on the {where}: {len(idx)} pairs, e.g. '
          + ', '.join(pairs[i]['landmark'] for i in idx[:3]))
neg = np.argwhere(mean < -0.05)
print(f'cells with mean NE below -0.05: {len(neg)}' + ''.join(
    f'; {KINDS[k]}, block {l}, {CLASSES[c]!r}: {mean[k, l, c]:.2f}'
    for k, l, c in neg))
if FULL is not None:
    print('full run: handover', FULL['patching']['handover'])
    for kind in KINDS:
        print(f'  {kind}: {d2l.block_runs(FULL["patching"][kind], CLASSES)}')
d2l.show_heatmaps(torch.tensor(mean).clamp(min=0)[None], 'position', 'block',
                  titles=list(KINDS), figsize=(7, 4.5), xticklabels=CLASSES)
for ax in d2l.plt.gcf().axes[:len(KINDS)]:  # the names read vertically
    ax.tick_params('x', labelrotation=90)
d2l.plt.gcf().axes[-1].set_ylabel('mean NE');  # the colour bar
```

<!-- Placeholder (prose, ### Attribution Patching): eq_lm_atp at every site from one clean forward and backward and one corrupted forward, timed against the sweep (quote "about 20 times faster": the pass-count ratio is 224, and batching the patches closes most of that gap). -->

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
        M = masks(g['cls'], PATTERNS).float().to(dev)  # (8, T)
        for i, s in enumerate(SITES):
            delta = corrupt.cache[s][0] - clean.cache[s][0].detach()
            pe = -(delta * clean.grad_cache[s][0]).sum(-1)  # (pairs, T)
            out[i // L, i % L, :, g['idx']] = (M @ pe.T).cpu() / gap[g['idx']]
    return out.numpy()

atp_time = d2l.Benchmark(atp, warmup=1, repeats=3).time
NE_atp = atp()
RESULTS['atp.csv'] = d2l.long_csv(NE_atp, AXES, 'atp_ne')
print(f'AtP tensor {NE_atp.shape}')
print(f'patching: {sweep_time:.2f} s for {3 * L * 8} forwards per pair; '
      f'AtP: {atp_time:.2f} s for 3 passes; ratio {sweep_time / atp_time:.0f}')
```

<!-- Placeholder (prose, ### Attribution Patching): rank agreement of AtP with exact patching outside the near-zero cells (the attn_out interval rests on few cells), the error by block band, and the plot below (caption from spec fig_lm_atp_vs_exact). Reading (verify 18.6): AtP misses the subject positions at every block before the handover, not only early (the subject classes together: exact NE about 1, AtP about 0.1); cite Kramar et al. 2024 (AtP*) on attention saturation. The effect passes through attention there and the patched path is flat at the clean end (the effect rises in an S-curve from clean to corrupted; the slope at the clean end, which AtP computes exactly, is small). After the handover AtP is close but low at the final position (about 40% low at the handover block, 11-18% low from block 22 on). Without the early subject cells resid_post's Spearman is about 0.88, but that cut was not declared: word it by block band only. -->

```{.python .input #interpretability-atp-agreement}
%%tab pytorch
def spearman(E, A):  # per kind and pooled, cells with |exact NE| > 0.05
    keep = np.abs(E) > 0.05
    pooled = spearmanr(E[keep], A[keep]).statistic
    return [spearmanr(E[k][keep[k]], A[k][keep[k]]).statistic
            for k in range(len(KINDS))] + [pooled]

exact, approx = NE.mean(-1), NE_atp.mean(-1)
big = np.abs(exact) > 0.05  # the cells the rank agreement uses
rho = spearman(exact, approx)
rho_ci = d2l.percentile_ci(
    [spearman(NE[..., idx].mean(-1), NE_atp[..., idx].mean(-1))
     for idx in d2l.cluster_resample(cities)], names=[*KINDS, 'pooled'])
SUMMARY['atp spearman'] = {}
print('Spearman over the cells with |exact NE| > 0.05:')
for j, (name, n) in enumerate(zip([*KINDS, 'pooled'],
                                  [*big.sum((1, 2)), big.sum()])):
    SUMMARY['atp spearman'][name] = d2l.fmt_ci(rho[j], rho_ci[:, j], 2)
    print(f'  {name} ({n} cells): {d2l.fmt_ci(rho[j], rho_ci[:, j], 2)}')
print('mean absolute error on those cells, by blocks: ' + ', '.join(
    f'{b[0]}-{b[-1]} {np.abs(approx - exact)[:, b][big[:, b]].mean():.2f}'
    for b in np.array_split(np.arange(L), 4)))
for what, cols, bs in (('both subject classes', [0, SUBJ_LAST],
                        range(handover)),
                       ('final position', [FINAL], range(handover, L))):
    e, a = (x[0, bs][:, cols].sum(-1) for x in (exact, approx))
    print(f'resid_post, {what}, blocks {bs[0]}-{bs[-1]}: '
          f'NE exact {e.mean():.2f}, AtP {a.mean():.2f} (block means)')
if FULL is not None:
    print('full run:', json.dumps(FULL['atp spearman'], indent=1))
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
        """Squared error normalized by the batch variance; the auxiliary
        loss, the same for the residual x - x_hat reconstructed from the
        k_aux largest pre-activations of the dead features (a mask); and
        the codes."""
        x = x.to(self.W_enc.dtype).reshape(-1, self.d_model)
        pre = self.pre_acts(x)
        z = self.topk_relu(pre, self.k)
        x_hat = self.decode(z)
        nmse = lambda u, v: ((u - v) ** 2).sum(-1).mean() / (
            (v - v.mean(0)) ** 2).sum(-1).mean()
        aux = x_hat.new_zeros(())
        if dead is not None and dead.any():
            e = (x - x_hat).detach()
            z_dead = self.topk_relu(pre.masked_fill(~dead, -torch.inf),
                                    min(k_aux, int(dead.sum())))
            aux = nmse(F.linear(z_dead, self.W_dec), e)
        return nmse(x_hat, x), aux, z

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
    """Train a TopKSAE with Adam on random rows of activations (N, d), with
    unit decoder columns and the auxiliary loss of the features idle for
    dead_after steps; seed fixes the batches. Returns the losses."""
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
        mse, aux, z = sae.loss(x, idle >= dead_after)
        loss = mse + aux_coef * aux
        with torch.no_grad():  # steps since each feature last fired
            idle = torch.where((z > 0).any(0), 0, idle + 1)
        optimizer.zero_grad()
        loss.backward()
        sae.remove_parallel_grad_()
        optimizer.step()
        sae.normalize_decoder_()
        losses.append(loss.item())
    return losses
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): caching the block-14 stream and training an SAE on it under D2L_RERUN; the CPU path loads the one trained for Qwen3-0.6B. -->

```{.python .input #interpretability-sae-r1}
%%tab pytorch
def train_ours(acts, name):  # TopKSAE(d, 8192, k=50), seed 0, fp16 file
    sae = d2l.TopKSAE(model.num_hiddens, 8192, 50, center=True,
                      seed=0).to(dev)
    losses = d2l.train_sae(sae, acts, steps=SAE_STEPS)
    print(f'loss at steps 0, {SAE_STEPS // 2}, {SAE_STEPS}: ' + ', '.join(
        f'{losses[i]:.3f}' for i in (0, SAE_STEPS // 2, -1)))
    d2l.lm_save_sae(sae, len(acts), name)
    return sae, len(acts)

def load_ours(key):
    sae, tokens = d2l.lm_load_sae(key, dev)
    print(f'{key}: sha1 {d2l.DATA_HUB[key][1]}')
    return sae, tokens

if 'sae-06b' in RERUN:
    sae_06b, tokens_06b = train_ours(
        d2l.cache_stream(model, train_windows, 14), 'lm-sae-r1-l14.pt')
elif SMALL:
    sae_06b, tokens_06b = load_ours(ARTIFACTS['sae-06b'])
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): Qwen3-1.7B-Base replaces Qwen3-0.6B, because the released dictionaries exist for it. -->

```{.python .input #interpretability-load-17b}
%%tab pytorch
if SMALL:
    print('CPU path: Qwen3-0.6B with its own SAE; the release is skipped')
else:
    BASE_MODEL = 'Qwen/Qwen3-1.7B-Base'
    del model, lens, ctx  # a context keeps the model it instruments
    del E, gamma, target, lp  # Qwen3-0.6B's embedding, held-out logits
    for g in groups:  # the corrupted activations of the patching runs
        del g['acts']
    d2l.lm_free()
    assert torch.cuda.memory_allocated() < 2**29, 'Qwen3-0.6B is still held'
    model = d2l.load_qwen3(BASE_MODEL).requires_grad_(False)  # bf16
    # the same vocabulary, so the token ids of the data cell stay valid
    assert AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=model.revision).get_vocab() == tok.get_vocab()
    print(f'{BASE_MODEL} at {model.revision}: '
          f'{sum(p.numel() for p in model.parameters()):,} parameters')
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): under D2L_RERUN, the block-14 stream of Qwen3-1.7B-Base on the training windows. -->

```{.python .input #interpretability-act-cache}
%%tab pytorch
if 'sae-17b' in RERUN:
    acts = d2l.cache_stream(model, train_windows, 14)
    print(f'{len(acts):,} positions cached, {acts.numel() * 2 / 1e9:.1f} GB '
          'of host memory')
```

<!-- Placeholder (prose, ### A Top-k Autoencoder Trained in the Notebook): our SAE for Qwen3-1.7B-Base, trained under D2L_RERUN and loaded otherwise (about 7 M tokens, not 8 M: the outliers and position 0 are left out). -->

```{.python .input #interpretability-sae-train-or-load}
%%tab pytorch
if SMALL:
    sae, sae_tokens = sae_06b, tokens_06b
elif 'sae-17b' in RERUN:
    sae, sae_tokens = train_ours(acts, 'lm-sae-q17b-l14.pt')
    del acts
else:
    sae, sae_tokens = load_ours(ARTIFACTS['sae-17b'])
```

<!-- Placeholder (prose, ### A Released Dictionary): Qwen-Scope's Caution paragraph, quoted from the README at the pinned revision, and the size of the release. -->

```{.python .input #interpretability-caution}
%%tab pytorch
SAE_REPO = 'Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'
revision = d2l.HF_REVISIONS[SAE_REPO]
with open(hf_hub_download(SAE_REPO, 'README.md', revision=revision)) as f:
    d2l.print_wrapped(re.search(r'## Caution\n(.*?)\n## ', f.read(),
                                re.S)[1].strip())
sizes = [s.size for s in HfApi().model_info(
    SAE_REPO, revision=revision, files_metadata=True).siblings
         if s.rfilename.endswith('.sae.pt')]
mb = ', '.join(str(m) for m in sorted({round(b / 1e6) for b in sizes}))
print(f'{len(sizes)} files of {mb} MB; this notebook downloads one')
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
    """L0, FVU, loss recovered and firing density of sae on the stream
    after block layer of model over token_batches ((B, T) ids), at the
    positions >= 1 whose norm is at most 10 times the batch median."""
    if sae.W_dec.shape[0] != model.num_hiddens:
        raise ValueError(f'SAE width {sae.W_dec.shape[0]} != model width '
                         f'{model.num_hiddens}')
    site, dev = ('resid_post', layer), model.token_emb.weight.device
    token_batches = list(token_batches)  # read twice

    def ce(logits, X, keep):  # per row: summed loss and count, kept t
        loss = torch.stack([F.cross_entropy(row[:-1].float(), x[1:],
                                            reduction='none')
                            for row, x in zip(logits, X)])  # a row at a time
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
    sse, sst = [], []  # per row, over its kept positions
    fired = torch.zeros(sae.d_sae, device=sae.W_dec.device)
    nnz, ce_splice, ce_mean = 0, [], []
    for X, keep in zip(token_batches, keeps):  # pass 2: the two splices
        X, k = X.to(dev), keep[..., None]

        def splice(x):
            nonlocal nnz, fired
            x_hat, z = sae(x)
            xf, x_hat = x.float(), x_hat.float()
            sse.append((((x_hat - xf) ** 2).sum(-1) * keep).sum(1))
            sst.append((((xf - mean) ** 2).sum(-1) * keep).sum(1))
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
    pooled = lambda pairs: (sum(float(s.sum()) for s, _ in pairs)
                            / sum(int(c.sum()) for _, c in pairs))
    ce_o, ce_s, ce_m = pooled(ce_orig), pooled(ce_splice), pooled(ce_mean)
    density = (fired / n).cpu().numpy()
    return {'l0': nnz / n, 'fvu': float(cat(sse).sum() / cat(sst).sum()),
            'ce_orig': ce_o, 'ce_splice': ce_s, 'ce_mean_ablate': ce_m,
            'loss_recovered': (ce_m - ce_s) / (ce_m - ce_o),
            'unfired_frac': float((density == 0).mean()), 'density': density,
            'n_tokens': n}
```

<!-- Placeholder (prose, ### A Released Dictionary): the instrument check, both encoder conventions of the release by reconstruction on held-out text, and the declared rule that decides whether the release is used. The release reconstructs only with the raw residual (center=False): a good teaching point for the instrument check. -->

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

<!-- Placeholder (prose, ### A Released Dictionary): the table above (spec tab_lm_sae_compare), ours beside the release as context, not a contest; FVU and loss recovered are different instruments. Loss recovered compresses because mean ablation is a weak baseline (its cross-entropy is about 9.2); the cross-entropy the splice adds is the more telling number. "Unfired" counts the features that never fire on these held-out positions: never call them dead. -->

```{.python .input #interpretability-sae-compare}
%%tab pytorch
m_ours = d2l.sae_metrics(sae, model, eval_batches, 14)
rows = {'ours': (sae.d_sae, sae_tokens, m_ours)}
if not SMALL:  # the release states no token count
    rows['Qwen-Scope'] = (release.d_sae, None, m_release)
COLUMNS = {'FVU': 'fvu', 'CE orig': 'ce_orig', 'CE splice': 'ce_splice',
           'CE mean-ablate': 'ce_mean_ablate',
           'loss recovered': 'loss_recovered'}
sae_table = pd.DataFrame({name: {
    'width': f'{width:,}',
    'training tokens': 'not stated' if t is None else f'{t:,}',
    'L0': f'{m["l0"]:.0f}',
    **{col: f'{m[key]:.3f}' for col, key in COLUMNS.items()},
    'unfired features': f'{int((m["density"] == 0).sum()):,}'}
                          for name, (width, t, m) in rows.items()}).T
print(f'{m_ours["n_tokens"]:,} positions of {len(heldout)} documents')
print(sae_table.T.to_string())  # one column per SAE
SUMMARY['sae'] = sae_table.to_dict('index')
RESULTS['sae_metrics.json'] = json.dumps({
    name: {k: v for k, v in m.items() if k != 'density'}
    for name, (_, _, m) in rows.items()}, indent=1).encode()
if FULL is not None:
    print('full run:', json.dumps(FULL['sae'], indent=1))
```

<!-- Placeholder (prose, ### Features and Their Labels): a feature chosen by contrast (Chinese against English Wikipedia) and two by a density rule, their top contexts; never-firing and dense features (feature splitting and absorption named). -->

```{.python .input #interpretability-features}
%%tab pytorch
feature_sae = (release if USE_RELEASE else sae).requires_grad_(False)
density = (m_release if USE_RELEASE else m_ours)['density']  # held-out
SITE = ('resid_post', 14)
H = {lang: {part: d2l.lm_stream(model, docs, SITE)
            for part, docs in wiki[lang].items()} for lang in wiki}
rate = {lang: d2l.sae_fire_rate(feature_sae, H[lang]['contrast'])
        for lang in H}
contrast_top = (rate['zh'] - rate['en']).topk(5).indices
print(pd.DataFrame({'feature': contrast_top.tolist(),
                    'rate zh': rate['zh'][contrast_top].tolist(),
                    'rate en': rate['en'][contrast_top].tolist()}).round(
    3).to_string(index=False))
feature = int(contrast_top[0])
d_mean = (H['zh']['contrast'].flatten(0, 1).float().mean(0)
          - H['en']['contrast'].flatten(0, 1).float().mean(0))
print(f'features never firing on the held-out text: {(density == 0).sum():,}'
      f'; dense (above 0.1): {(density > 0.1).sum():,}')
alive = np.flatnonzero((density >= 1e-3) & (density <= 1e-2))  # by index
alive = alive[alive != feature]
FEATS = [feature, *alive[np.random.default_rng(0).permutation(len(alive))[:2]]]
context_docs = torch.cat([wiki['zh']['contrast'], wiki['en']['contrast'],
                          heldout[:, :128]])
A = torch.cat([d2l.sae_acts(feature_sae, H[lang]['contrast'], FEATS)
               for lang in H] + [d2l.sae_acts(feature_sae, d2l.lm_stream(
                   model, heldout[:, :128], SITE), FEATS)])
for j, f in enumerate(FEATS):  # top-8 contexts: 11 tokens, then [token]
    print(f'feature {f}:')
    for i, t in (divmod(int(k), A.shape[1]) for k in A[..., j].ravel().topk(8)
                 .indices):
        before, at = tok.decode(context_docs[i, max(0, t - 10):t + 1]), (
            tok.decode(context_docs[i, t + 1:t + 2]))
        print(f'  {A[i, t, j]:6.1f}  {before}[{at}]'.replace('\n', '\\n'))
```

<!-- Placeholder (prose, ### Features and Their Labels): the contrast feature's label "Chinese text" tested as a token rule on held-out documents, beside the difference-of-means probe and the five contrast features together (feature splitting). The rule counts a token as Chinese if it holds a CJK character, punctuation included, and leaves out byte-fragment tokens; an earlier rule ("holds a Han character") counted the fullwidth punctuation and the fragments of Chinese text as not Chinese: say that the rule corrects the instrument. -->

```{.python .input #interpretability-label-test}
%%tab pytorch
# the rule: a token is Chinese if it holds a CJK ideograph, CJK punctuation
# or a fullwidth form; a token that decodes alone to U+FFFD (part of a
# character's bytes) cannot be labelled alone and is left out
CHINESE = re.compile('[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff'
                     '\uf900-\ufaff\uff00-\uffef]')
zh_tokens = np.array([CHINESE.search(p) is not None for p in pieces])
fragments = np.array(['\ufffd' in p for p in pieces])
label_ids = torch.cat([wiki[lang]['label'][:, 1:] for lang in H]).numpy()
keep = ~fragments[label_ids].ravel()
y = zh_tokens[label_ids].ravel()[keep]
label_doc = np.repeat(np.arange(len(label_ids)), label_ids.shape[1])[keep]
doc_lang = np.repeat(list(H), [len(wiki[lang]['label']) for lang in H])
H_test = torch.cat([H[lang]['label'] for lang in H])
acts5 = d2l.sae_acts(feature_sae, H_test, contrast_top.tolist())
score = {'feature': acts5[..., 0], 'probe': H_test.float() @ d_mean,
         'any of the top 5': acts5.max(-1).values}
score = {name: s.cpu().numpy().ravel()[keep] for name, s in score.items()}
threshold = {'feature': 0.0, 'any of the top 5': 0.0, 'probe': float(np.mean(
    [(H[lang]['contrast'].float() @ d_mean).mean().item() for lang in H]))}
print(f'{len(y):,} tokens labelled, {y.sum():,} of them Chinese; '
      f'{(~keep).sum():,} byte fragments left out')

def label_stats(idx):  # per score: precision, recall, AUC
    return [v for name, s in score.items() for v in d2l.detection_stats(
        y[idx], s[idx], threshold[name])]

STATS = [f'{stat}, {name}' for name in score
         for stat in ('precision', 'recall', 'AUC')]
with np.errstate(invalid='ignore', divide='ignore'):  # no active token
    label_est = label_stats(np.arange(len(y)))
    label_reps = np.array([label_stats(idx) for idx in
                           d2l.cluster_resample(label_doc)])
label_ci = d2l.percentile_ci(label_reps, names=STATS)
results = {name: [d2l.fmt_ci(label_est[3 * j + i], label_ci[:, 3 * j + i])
                  for i in range(3)] for j, name in enumerate(score)}
print(pd.DataFrame(results, index=['precision', 'recall', 'AUC']).T.to_string())
for name, s in score.items():
    active = doc_lang[np.unique(label_doc[s > threshold[name]])]
    print(f'{name}: active in ' + ', '.join(
        f'{(active == lang).sum()} of {(doc_lang == lang).sum()} {lang}'
        for lang in H) + ' documents')
d_auc = d2l.fmt_ci(label_est[2] - label_est[5], d2l.percentile_ci(
    label_reps[:, 2] - label_reps[:, 5]))
print(f'AUC, feature minus probe: {d_auc}')
SUMMARY['label test'] = {**results, 'AUC difference': d_auc}
if FULL is not None:
    print('full run:', json.dumps(FULL['label test'], indent=1))
```

<!-- Placeholder (prose, ### Steering Directions): three directions at block 14 (the feature's decoder column, the difference of means, random ones), and the median stream norm that makes their strengths comparable. The two informed directions have cosine only about 0.28, yet both steer. -->

```{.python .input #interpretability-directions}
%%tab pytorch
directions = {
    'SAE feature': F.normalize(feature_sae.W_dec[:, feature].float(), dim=0),
    'difference of means': F.normalize(d_mean, dim=0)}
for seed in range(SIZES['random']):
    g = torch.Generator().manual_seed(seed)
    directions[f'random {seed}'] = F.normalize(
        torch.randn(model.num_hiddens, generator=g), dim=0).to(dev)
INFORMED, RANDOM = list(directions)[:2], list(directions)[2:]
n_bar = float(d2l.lm_stream(model, cost_docs, SITE).float().norm(
    dim=-1).median())
cos = lambda u, v: float(u @ v)
print(f'median stream norm on English text: {n_bar:.1f}')
print(f'cosine of the feature and the difference of means: '
      f'{cos(*(directions[n] for n in INFORMED)):.2f}')
worst = max(abs(cos(directions[r], directions[i]))
            for r in RANDOM for i in INFORMED)
print(f'largest |cosine| of a random direction with either: {worst:.3f}')
if not SMALL:  # its decoder column is all we keep of the release
    del release, feature_sae
    d2l.lm_free()
```

<!-- Placeholder (prose, ### The Steering Frontier): eq_lm_steer on a grid of strengths for each direction (half-octave from 1/16 to 2; for the two informed directions also every twelfth of an octave from 1/16 to 1/4: this fine grid was added after a first run showed the switch happening within a third of an octave; say so), and a fixed Chinese prefix that changes no activation: behaviour (switched prompts) and cost (cross-entropy on English text). The CPU path (Qwen3-0.6B with its own SAE, another model) uses half-octave steps up to 1/2 plus two quarter-octave points between 1/8 and 1/4, where its switch happens; its frontier numbers are not comparable with the full run's. Past the cliff the outputs degenerate ("（（（（" loops, repeated phrases); the switched fraction falls partly because fullwidth punctuation is not Han. Call it breakdown, never a return to English: the cost falls there only because a broken model predicts less confidently. -->

```{.python .input #interpretability-steer}
%%tab pytorch
COARSE = [2 ** (j / 2) for j in range(-8, 3)]  # 1/16 to 2, half-octave
FINE = [2 ** (j / 12) for j in range(-48, -23)]  # 1/16 to 1/4, 1/12 octave
GRID = {}
for name in directions:  # each grid starts at alpha = 0
    if SMALL:  # the CPU path: half-octave steps up to 1/2, and quarter-
        # octave steps between 1/8 and 1/4 for the informed directions
        alphas = ([a for a in COARSE if a <= 0.5] + [2 ** -2.75, 2 ** -2.25]
                  if name in INFORMED else [0.125, 0.5])
    elif name in INFORMED:
        alphas = sorted(set(COARSE) | set(FINE))
    else:
        alphas = COARSE
    GRID[name] = [0.0] + sorted(alphas)
PREFIX = '下面的文字将用中文继续。\n\n'  # declared before any run, not tuned
SWITCH = 0.3  # a continuation switched: at least 30% Han characters
CAP = 0.5  # the cost cap: nats per token on English text
lengths = [{len(ids_of(pre + t)) for t in steer_texts} for pre in ('', PREFIX)]
assert all(len(n) == 1 for n in lengths)  # one bucket of prompts each
print(f'{len(steer_texts)} prompts of {lengths[0].pop()} tokens, '
      f'{lengths[1].pop()} with the prefix')

def steer(v):  # adds v at every position but the first
    def fn(x):
        if x.shape[1] == 1:  # a decode step
            return x + v.to(x.dtype)
        return torch.cat([x[:, :1], x[:, 1:] + v.to(x.dtype)], 1)
    return fn

def steering(fn):
    return contextlib.nullcontext() if fn is None else d2l.intervene(
        model, {SITE: fn})

def run(fn=None, prefix=''):  # Han fractions per prompt, CE per document
    with steering(fn):
        gens = d2l.generate(model, tok, [prefix + t for t in steer_texts],
                            max_new_tokens=48, batch_size=len(steer_texts))
        return (np.array([d2l.char_fraction(g.text, HAN) for g in gens]),
                d2l.lm_doc_ce(model, cost_docs, ids_of(prefix), han_tokens))

base, runs = run(), {}
for name, d in directions.items():
    runs[name, 0.0] = base  # alpha = 0 is one shared run
    for alpha in GRID[name][1:]:
        runs[name, alpha] = run(steer(alpha * n_bar * d))
runs['prefix', 0.0] = run(prefix=PREFIX)
switched = lambda n, a: (runs[n, a][0] >= SWITCH).mean()
cost_of = lambda n, a: (runs[n, a][1] - base[1])[:, 0].mean()
below = {n: max(i for i, a in enumerate(GRID[n]) if cost_of(n, a) < CAP)
         for n in directions}  # the largest alpha below the cap
print('switched at the largest alpha below the cap (cost in nats), then at '
      'the next alpha:')
for n in INFORMED:
    print(f'  {n}: ' + '; '.join(
        f'{switched(n, a):.2f} at alpha {a:.3g} ({cost_of(n, a):.2f})'
        for a in GRID[n][below[n]:below[n] + 2]))
i = max(below[n] for n in RANDOM)  # the random draws share their grid
print(f'  random ({len(RANDOM)} draws): at most '
      f'{max(switched(n, a) for n in RANDOM for a in GRID[n]):.2f} '
      'switched at any alpha;\n    cost ' + ', '.join(
          f'{min(cost_of(n, a) for n in RANDOM):.2f}-'
          f'{max(cost_of(n, a) for n in RANDOM):.2f} at alpha {a:.3g}'
          for a in GRID[RANDOM[0]][i:i + 2]))
print(f'prefix: switched {switched("prefix", 0.0):.2f}')
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

<!-- Placeholder (prose, ### The Steering Frontier): the frontier plot below (caption from spec fig_lm_steering_frontier: the line is the frontier, the most switched at each cost or less; grey points are strengths past each direction's peak, where the output breaks down; every random draw switches nothing, so their mean is the floor), areas up to the cost cap with paired intervals under both costs (plain cross-entropy on English, which partly measures the intended shift to Chinese, and the same without Han tokens, the collateral damage), the random-direction floor, and the prefix against each direction at its own cost; the bootstrap resamples prompts (seed 0) and cost documents (seed 1) independently. Cost of half switched is read along the strengths, between the first strength that switches half and the one before it. Wording (verify 18.6, review 18.6): the area difference is marginal (lower bound near 0, about 2% of replicates at or below 0): the feature is "slightly cheaper at this sample size", not clearly cheaper; a step envelope without interpolation gives the same sign. The prefix is an instruction on a base model, which continues the document: at its cost every method switches only a few percent, so "worse than a direction" carries little; AxBench's comparison is about instruction-tuned models (author ruling pending, review 18.6 finding 5). Samples are taken at the largest strength below the cap, where only a minority of prompts switch: that is why most continuations stay English. -->

```{.python .input #interpretability-frontier}
%%tab pytorch
SW = {n: np.array([runs[n, a][0] >= SWITCH for a in GRID[n]])
      for n in directions}  # (alphas, prompts)
CE = {n: np.array([runs[n, a][1] - base[1] for a in GRID[n]])
      for n in directions}  # (alphas, documents, plain or without Han)
sw_p = runs['prefix', 0.0][0] >= SWITCH
ce_p = runs['prefix', 0.0][1] - base[1]
COSTS = ('plain', 'without Han')  # CE[n][..., 0] and CE[n][..., 1]
ROWS = [*(f'area, {n}' for n in INFORMED), 'area difference',
        *(f'switched at the cap, {n}' for n in INFORMED),
        *(f'cost of half switched, {n}' for n in INFORMED)]

def frontier_stats(ip, idoc):  # the declared statistics, resampled
    out = []
    for m in range(len(COSTS)):  # per alpha: (mean cost, switched fraction)
        curve = {n: (CE[n][:, idoc, m].mean(1), SW[n][:, ip].mean(1))
                 for n in directions}
        a = [d2l.frontier_area(*curve[n], CAP) for n in INFORMED]
        out += [*a, a[0] - a[1], *(np.interp(CAP, *d2l.frontier_envelope(
            *curve[n])) for n in INFORMED), *(d2l.cost_of_half(*curve[n])
                                               for n in INFORMED)]
        if m == 0:
            plain = curve
    excess = max(plain[n][1][np.flatnonzero(plain[n][0] < CAP).max()]
                 for n in RANDOM) - plain[RANDOM[0]][1][0]
    cp, sp = ce_p[idoc, 0].mean(), sw_p[ip].mean()
    return out + [excess, *(sp - np.interp(cp, *d2l.frontier_envelope(
        *plain[n])) for n in INFORMED)]

n_p, n_d = len(steer_texts), len(cost_docs)
NAMES = [f'{r}, cost {c}' for c in COSTS for r in ROWS] + [
    'random excess', *(f'prefix minus {n}' for n in INFORMED)]
est = frontier_stats(np.arange(n_p), np.arange(n_d))
# bootstrap seeds: 0 for the prompts, 1 for the documents (independent)
est_reps = np.array([frontier_stats(ip, idoc) for ip, idoc in zip(
    d2l.cluster_resample(range(n_p)),
    d2l.cluster_resample(range(n_d), seed=1))])
est_ci = d2l.percentile_ci(est_reps, names=NAMES)
cells = [d2l.fmt_ci(v, est_ci[:, j]) for j, v in enumerate(est)]
SUMMARY['steering'] = dict(zip(NAMES, cells))
excess_bound = float(np.quantile(est_reps[:, NAMES.index('random excess')],
                                 0.95))
below_0 = float(np.mean(est_reps[:, 2] <= 0))
SUMMARY['steering']['random excess, upper 95% bound'] = excess_bound
SUMMARY['steering']['area difference, replicates at most 0'] = below_0
print(pd.DataFrame({f'cost {c}': cells[len(ROWS) * m:len(ROWS) * (m + 1)]
                    for m, c in enumerate(COSTS)}, index=ROWS).to_string())
print(f'area difference, plain cost: {below_0:.1%} of replicates at most 0')
print(f'random directions, switched beyond the unsteered rate below the cap: '
      f'{est[NAMES.index("random excess")]:.3f} (upper 95% bound '
      f'{excess_bound:.3f})')
for n in INFORMED:
    print(f'prefix minus {n}, at the prefix\'s cost: '
          f'{SUMMARY["steering"][f"prefix minus {n}"]}')
if FULL is not None:
    print('full run:', json.dumps(FULL['steering'], indent=1))
cost = {n: CE[n][..., 0].mean(1) for n in directions}  # plain, per alpha
frac = {n: SW[n].mean(1) for n in directions}
peak = {n: len(frac[n]) - 1 - int(np.argmax(frac[n][::-1]))
        for n in INFORMED}  # the last alpha of the most switched
env = {n: d2l.frontier_envelope(cost[n], frac[n]) for n in INFORMED}
past = [np.concatenate([v[n][peak[n] + 1:] for n in INFORMED])
        for v in (cost, frac)]  # past the peak: the output breaks down
d2l.plot([env[n][0] for n in INFORMED]
         + [np.mean([cost[n] for n in RANDOM], 0), [ce_p[:, 0].mean()],
            past[0]] + [cost[n][:peak[n] + 1] for n in INFORMED] + [[CAP, CAP]],
         [env[n][1] for n in INFORMED]
         + [np.mean([frac[n] for n in RANDOM], 0), [sw_p.mean()], past[1]]
         + [frac[n][:peak[n] + 1] for n in INFORMED] + [[0, 1]],
         'cross-entropy increase on English (nats per token)',
         'fraction switched to Chinese',
         legend=[*INFORMED, f'random (mean of {len(RANDOM)})', 'prefix',
                 'past the peak'],
         fmts=('C0-', 'C1-', 'C2x', 'C3*', 'C7o', 'C0o', 'C1s', 'k--'),
         figsize=(5, 3.5), legend_loc='upper left')
ax = d2l.plt.gca()
ax.set_xscale('symlog', linthresh=0.01)  # linear up to 0.01 nats, then log
ax.set_xlim(min(0, *(c.min() for c in cost.values())),
            2 * max(c.max() for c in cost.values()))
RESULTS['summary.json'] = json.dumps(SUMMARY, indent=1, default=float,
                                     ensure_ascii=False).encode()
if 'results' in RERUN:  # the CPU path's numbers are never published
    d2l.lm_save_rerun('lm-interp-results.zip',
                      lambda p: d2l.lm_write_zip(p, RESULTS))
```

<!-- Placeholder (prose, ### The Steering Frontier): continuations at each informed direction's operating point, beside the unsteered model and the prefix, including the Eiffel prompt. -->

```{.python .input #interpretability-steer-samples}
%%tab pytorch
def print_folded(text, width=76, indent='    '):  # Han takes two columns
    cols = lambda s: sum(1 + (unicodedata.east_asian_width(c) in 'WF')
                         for c in s)
    lines, line = [], ''
    for word in text.split(' '):
        sep = ' ' if line.strip() else ''
        if cols(line + sep + word) <= width:
            line += sep + word
        elif cols(word) == len(word) and len(indent + word) <= width:
            lines.append(line)  # a word without wide characters: next line
            line = indent + word
        else:  # a run of wide characters: fill the line, then go on
            line += sep
            for c in word:
                if cols(line + c) > width:
                    lines.append(line)
                    line = indent
                line += c
    print('\n'.join(l.rstrip() for l in lines + [line]))

samples = [*steer_texts[:3], eiffel]
conditions = {'unsteered': (None, '', 0.0),
              'prefix': (None, PREFIX, ce_p[:, 1].mean())}
for n in INFORMED:  # the smallest alpha that switches half below the cap
    under = cost[n] < CAP
    hit = np.flatnonzero(under & (SW[n].mean(1) >= 0.5))
    i = hit[0] if len(hit) else np.flatnonzero(under).max()
    print(f'{n}: alpha {GRID[n][i]:.3g}'
          + ('' if len(hit) else ' (the largest below the cap; no half)'))
    conditions[n] = (steer(GRID[n][i] * n_bar * directions[n]), '',
                     CE[n][i, :, 1].mean())
for name, (fn, prefix, renorm) in conditions.items():
    with steering(fn):
        gens = d2l.generate(model, tok, [prefix + s for s in samples],
                            max_new_tokens=48)
    print(f'--- {name} (cost without Han: {renorm:+.3f} nats per token)')
    for s, g in zip(samples, gens):
        print_folded(f'...{s[-30:]}|{g.text}'.replace('\n', ' '))
```
