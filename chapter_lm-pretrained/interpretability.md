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
import json
import os
import re
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoTokenizer
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
