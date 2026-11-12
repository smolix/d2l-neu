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

<!-- Placeholder prose (stage 2): the section text is written after the runs. -->
The released sparse autoencoder is pinned like the models of
:numref:`sec_lm_hf_models`, by adding its repository to `d2l.HF_REVISIONS`.

```{.python .input #interpretability-revisions}
%%tab pytorch
#@save
d2l.HF_REVISIONS['Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'] = (
    'ce1a79d9c5163932d65c417380e53230e1086370')
```
