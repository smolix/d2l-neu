# Deep-Scan Triage: Remaining Items

Items from `step1.md` that were **still pending** after the work landed
in commits `1afa0fb` (Critical) through `f0702b2` (deep-scan log update
on 2026-04-26). The fully-addressed Critical/Warning/Math/Notation items
were removed before this file was checked in. The third pass against
this list landed in commits `20e94c2`, `ad76f3c`, and `4f9cc17` (also
2026-04-26); each line below is now annotated with its disposition
(`DONE <commit>`, `SKIPPED <reason>`, `KEPT`, or `IGNORED`).

Counts (after the 2026-04-26 third pass):
- Critical: **0** outstanding; **8** intentionally not changed (all `KEEP`/`IGNORE` below).
- Warning: **0** outstanding; the actionable items closed, additive/by-design ones marked.
- Math / Notation: **0** outstanding.
- Code Issues: **0** outstanding.
- Cross-Framework Drift: **0** outstanding actionable items; deep architectural divergences left as `IGNORE`.

See `docs/deep-scan.md` "Fixes Applied (2026-04-26 — third pass)" for
the per-commit breakdown.

---

## Critical — none outstanding, but 7 items deliberately not changed

These were on the Critical list in `step1.md`; each is left alone with a
documented reason.

- `recurrent-neural-networks/rnn-concise.md`: JAX `RNN.__call__` is a `NotImplementedError` stub. Flax has no built-in vanilla RNN; the class is defined but never instantiated, so the stub never fires. Implementing a from-scratch JAX vanilla RNN is a feature addition.
**KEPT**
- `builders-guide/read-write.md`: `from flax.training import checkpoints` still works in `flax==0.10.6` (deprecated but not removed); leaving until Flax actually removes it.
**KEPT**
- `builders-guide/init-param.md`: `dtype=jnp.float_` still resolves in current JAX; leaving until JAX actually removes it.
**KEPT**
- `hyperparameter-optimization/sh-intro.md` JAX tab: `import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')` works because the JAX venv has TensorFlow installed for `tensorflow_datasets` data loading.
**SKIPPED** — removing the TF shim risks GPU-memory conflicts during the multi-trial HPO loop; the d2l/jax.py preamble's `set_memory_growth` alone may not be sufficient to keep TF off the GPU. Cosmetic gain not worth the regression risk.
- `recommender-systems/autorec.md` (PT): the masking via `pred * sign(input)` is applied in the model's `forward` before loss, so unobserved entries already contribute zero MSE. Mathematically equivalent to MX behavior.
**IGNORED**
- `convolutional-neural-networks/lenet.md`, `pooling.md`: triage flagged TF cells as having no output, but the latest TF runs produce outputs in those cells; the claim is stale.
**IGNORED**
- `multilayer-perceptrons/mlp.md` activation plots: same — current TF run produces all the plots.
**IGNORED**

---

## Warning — code-correctness items still pending

### Code paths

- `appendix-tools-for-deep-learning/utils.md`: claim of duplicate `download_extract` was a false alarm — there are two definitions but in different `%%tab` blocks (one in JAX-tab, one in PT/MX/TF-tab), so they don't shadow each other in the generated `d2l/<fw>.py`. **Skip unless we want to deduplicate the source for readability.**
**IGNORED**
- `linear-regression/linear-regression-concise.md` MX: `return 2 * fn(y_hat, y).mean()` — already commented in source ("Gluon's L2Loss includes 1/2; multiply by 2 to get plain MSE"). **Skip.**
**IGNORED**
- `convolutional-neural-networks/conv-layer.md` C2 / C4 / etc.: any remaining "Critical" items in the per-chapter report we already triaged.
**IGNORED**

### TensorFlow gaps (additive — large changes needed)

- `attention-mechanisms-and-transformers/vision-transformer.md` (TF): `tf.keras.layers.MultiHeadAttention` doesn't accept `valid_lens`; PT/JAX silently differ in the masking behavior. Either subclass MHA or document the gap.
**DONE `20e94c2`** — added a `:begin_tab:tensorflow:` block before the `ViTBlock` TF definition explaining that ViT patches need no masking, that Keras's MHA expects an `attention_mask` (shape `(B, Q, K)`) rather than `valid_lens`, and that sequence-data reuse must build the mask explicitly.
- `attention-mechanisms-and-transformers/vision-transformer.md`: MXNet tab is entirely absent. (TF Coverage Plan addresses this kind of gap — but the inverse, MX gap, doesn't have a similar plan.)
**IGNORED** — no new MXNet notebooks.
- `attention-mechanisms-and-transformers/attention-pooling.md`: TF tabs are absent for several output blocks (the helpers / plotting code work generically across frameworks via `d2l.numpy(...)`, but readers may want a TF-specific story).
**IGNORED**
- `attention-mechanisms-and-transformers/bahdanau-attention.md`: JAX is missing the `AttentionDecoder` base-class stub (PT/TF/MX have it).
**DONE `20e94c2`** — added a JAX `AttentionDecoder(d2l.Decoder)` `#@save` block alongside the existing PT/MX/TF one. Verified against `d2l.AttentionDecoder` after `make lib`; JAX bahdanau-attention notebook re-runs cleanly.
- `recurrent-modern/bi-rnn.md` (TF): concise BiGRU implementation block is absent (PT and MX have it).
**DONE `20e94c2`** — added a TF `BiGRU` block using `tf.keras.layers.Bidirectional(GRU(..., return_sequences=True, return_state=True))`. TF bi-rnn notebook executes cleanly (11.2s).
- `builders-guide/use-gpu.md` (TF): the `Trainer` GPU-extension cells are missing in the TF tab; TF training silently runs on CPU even when `try_gpu()` returns a GPU device. Adding a `Trainer.gpu()` etc. for TF is a non-trivial refactor.
**DONE `ad76f3c`** — added a TF `%%tab tensorflow` block with `Trainer.__init__` and `Trainer.prepare_batch` extensions. `prepare_batch` re-wraps each `tf.data.Dataset` batch via `tf.identity` inside `with self.gpus[0]:` so subsequent ops keep their inputs on-device, mirroring the PT/MX/JAX pattern. Added a `:begin_tab:tensorflow:` note explaining why no `prepare_model` override is needed (Keras layers materialize variables on whichever device they're first called with). TF use-gpu notebook executes cleanly (15.7s).
- `builders-guide/lazy-init.md`: JAX has no pre-init parameter access demonstration. Flax doesn't have lazy init in the same way as PT/MX, so this is somewhat unavoidable, but a "Why JAX is different" note would help.
**DONE `20e94c2`** — extended the existing `:begin_tab:jax:` paragraph with a "Why Flax is different" note: shape inference happens at `net.init(rng, dummy_input)` time (mandatory before use), so there is nothing to inspect *before* initialization; the imperative-framework narrative below is therefore framed for PT/MX/TF only.

### Pedagogical drift (architectural changes — needs author input)

- `computer-vision/fine-tuning.md`: PT uses ResNet-18; JAX/TF use ResNet-50. Pick one model throughout (ResNet-18 is faster; ResNet-50 is closer to what fine-tuning is typically applied to in practice).
**PARTIALLY DONE `20e94c2`** — split the single shared paragraph into two `:begin_tab:` blocks: `mxnet,pytorch` keeps the ResNet-18 narrative; `jax,tensorflow` documents that those tabs use ResNet-50 because `keras.applications` does not ship a pretrained ResNet-18 (and `keras_hub`/`keras_cv`/`tensorflow_hub` aren't in the venv). PT/MX code already used ResNet-18; a true cross-framework switch would require porting weights, out of scope.
- `computer-vision/fcn.md`: JAX/TF use a from-scratch ResNet instead of pretrained weights, defeating the section's transfer-learning point.
**PARTIALLY DONE `20e94c2`** — TF tab already used `keras.applications.ResNet50(weights='imagenet')`. Added a `:begin_tab:jax:` note explaining that JAX uses from-scratch ResNet because no pretrained Flax model is available in the venv (no `flaxmodels`, `transformers`, or PyTorch-to-JAX bridge installed). Closing this gap requires adding a pretrained-weights source.
- `multilayer-perceptrons/mlp.md:299–300`: kernel-method overstatement "way more effective". (Opinion-bearing — leave to authors.)
**IGNORED**

### Recommender-systems / sequence-aware

- `recommender-systems/recsys-intro.md`: triage flagged `:numref:`subsec_recommender_systems`` as broken, but the cross-ref actually resolves at HTML build time (Quarto renders it as "Section 1.3.1.5"). **Skip.**
**IGNORED**
- `recommender-systems/seqrec.md`: training cells in MX and PT are commented out. Both now carry the "takes >1h" rationale. The TF/JAX tabs don't have training cells at all. Leave to authors to decide whether to add a training run.
**DONE `4f9cc17`** — uncommented training in PT and MX with `eval_step=num_epochs`, deferring the costly per-user `evaluate_ranking` to the final epoch only. PT runs in 53s, MX in 255s (4.3 min). PT also moved `net.to(devices[0])` since the original cell trained on CPU. Fixed a latent bug in `evaluate_ranking` PT: the list-of-lists flatten step was iterating over scalar `numpy.float32` items because Caser returns shape `(B,)`; now uses `.ravel().tolist()` which handles both `(B,)` and `(B, 1)` outputs uniformly. Added a `:begin_tab:tensorflow,jax:` note documenting that the chapter is PT/MX-only because the Caser-specific helpers (`BPRLoss`, `SeqDataset`, `train_ranking`, `evaluate_ranking`) are not yet ported to those frameworks; TF/JAX implementations would be additive new content.

---

## Math / Notation — pending

- `appendix-mathematics-for-deep-learning/multivariable-calculus.md`: introduction uses `\frac{d}{dw_1} L(w_1, w_2, \ldots)` for a multivariate function (line 9) before introducing partial-derivative notation at line 12. The Hessian section was switched to `∂²f/∂x_i ∂x_j` but the earlier Taylor-expansion derivation still mixes `d` and `∂`. Cleanup would homogenize the whole section.
**DONE `ad76f3c`** — converted ~25 occurrences of `\frac{d}{dw_i}` / `\frac{df}{dx_i}` / `\frac{dx_i}{dx_k}` etc. to `\partial` form throughout the chain-rule, Jacobian, and matrix-factorization derivations. Single-variable analogues (`\frac{df}{dx} = 12x^3 - ...`, the contrast `\frac{d}{dx}(bx) = b`, the 1×1 sanity checks) intentionally left as ordinary `d`.
- `appendix-mathematics-for-deep-learning/random-variables.md`: triage flagged a `sign(a)` factor in `ρ(aX+b, Y)` and `1/(π(1+x²))` for the Cauchy pdf — both were already correct in the current source. **Skip.**
**SKIPPED** — already correct.
- `attention-mechanisms-and-transformers/queries-keys-values.md` line 32: circular softmax — fixed (commit `babce7a`). Remaining: re-check that the rest of the chapter's normalization equations are now self-consistent.
**CHECKED** — full sweep of `queries-keys-values.md`, `attention-pooling.md`, `attention-scoring-functions.md`, `bahdanau-attention.md`, `multihead-attention.md`, `self-attention-and-positional-encoding.md`, `transformer.md`. No inconsistencies found; the `a / Σ a` un-normalized form (line 32) appears only as a stepping stone to the explicit softmax (line 36). Downstream files reference either `eq_softmax_attention`, `eq_attn-scoring-alpha`, or `eq_softmax_QK_V`; all consistent. `attention-pooling.md` overloads `α` for both kernel and weight, but this is internally consistent and pre-existing.
- `linear-regression/linear-regression.md` line 206: `l = (1/2)(ŷ - y)²`. PT/TF MSE losses don't have the 1/2, so the effective LR is doubled when readers swap in `nn.MSELoss`. The book acknowledges "the constant 1/2 makes no real difference but proves notationally convenient" so this isn't strictly wrong, but a single-line warning that "if you use `nn.MSELoss`, set `lr` to half of what we use here" would help.
**DONE `20e94c2`** — added a one-paragraph note that frameworks' built-in MSE losses (e.g., `nn.MSELoss`, `tf.keras.losses.MeanSquaredError`) omit the `1/2`, so swapping in a built-in doubles the effective gradient and the learning rate should be halved to compensate.
- `multilayer-perceptrons/mlp.md` / `backprop.md`: row-vector vs column-vector convention switch between batch-first and per-example. Large rewrite to homogenize; needs author judgment.
**IGNORED**
- `multilayer-perceptrons/numerical-stability-and-init.md:86–92, 466`: Jacobian product for tensor layers (e.g., Conv) not flagged. Adding a sentence is easy; deciding what to flag and how is the work.
**IGNORED**
- `natural-language-processing-pretraining/glove.md`: the ratio expression `p_{ji}/p_{ki}` matches Pennington 2014 if we read indices as "first = center, second = context". The text is internally consistent. **Skip unless we want to add a clarifying sentence about the index convention.**
**DONE `20e94c2`** — added a brief sentence right after the definition of `p_{ij} ≝ P(w_j | w_i)` clarifying that *first index = conditioning center word, second index = generated context word*, so `p_{ji}` later means "context $w_i$ given center $w_j$".
- `recommender-systems/mf.md, neumf.md, ranking.md, seqrec.md`: argmin scope, bold/non-bold `\mathbf{h}` inconsistency, extra `)` somewhere, non-standard transpose-on-row-concat. Need a careful per-equation pass.
**DONE `20e94c2`** — surgical fixes:
  - `mf.md` line 25: `\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}}` → `b_*` (objective contains `b_u` and `b_i`, not a single `b`).
  - `neumf.md` line 14: `h` → `\mathbf{h}` for consistency with surrounding lines (11/24/32).
  - `neumf.md` line 23: removed an extra trailing `)` in `\alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})`.
  - `ranking.md` line 23: `\prod_{(u, i, j \in D)}` → `\prod_{(u, i, j) \in D}` (matches lines 24-25).
  - `seqrec.md` line 12: row-concat-then-transpose `[\mathbf{q}_{S_{t-L}^u}, ..., \mathbf{q}_{S_{t-1}^u}]^\top` rewritten as an explicit `\begin{bmatrix} ... \\ \vdots \\ ... \end{bmatrix}` so the result actually has the stated $L \times k$ shape.
- `recurrent-modern/seq2seq.md` BLEU footnote — DONE.
**IGNORED** (already done in earlier pass).
- `attention-mechanisms-and-transformers/attention-scoring-functions.md` — DONE.
**IGNORED** (already done in earlier pass).

---

## Cross-Framework Drift — pending

These are mostly architectural differences between frameworks. They aren't bugs per se; the question is whether to homogenize, document, or leave alone.

### API shape differences (intentional but undocumented)

- `linear-regression/linear-regression-scratch.md`: JAX `fit_epoch` is ~40 lines vs PT's ~15. JAX's functional state-passing pattern is genuinely more verbose. A two-sentence "Why JAX looks longer" callout would help readers.
**DONE `20e94c2`** — extended the existing `:begin_tab:jax:` block immediately above `fit_epoch` with a paragraph explaining that JAX is purely functional: there is no implicit `self`-attached state, so each step must explicitly take in and return optimizer state, dropout RNG, and (optionally) batch statistics; the explicit plumbing is what the extra lines are doing.
- `appendix-tools-for-deep-learning/utils.md`: `sgd` signature differs (PT/MX takes `params`, TF takes `params, grads`). Functional vs imperative — by design.
**SKIPPED**
- `appendix-tools-for-deep-learning/utils.md`: `evaluate_accuracy` only present for MX/TF. JAX/PT use the Trainer's built-in eval. Document this.
**DONE `20e94c2`** — see next entry; covered by the same end-of-file note.
- `appendix-tools-for-deep-learning/utils.md`: JAX is missing `train_ch6`, `train_seq2seq`, `predict_seq2seq`, `MaskedSoftmaxCELoss`. Either port them or note that JAX uses Trainer-only flow.
**DONE `20e94c2`** — added a short "A Note on Framework Coverage" section at the end of `utils.md` documenting that the legacy helpers are kept for parity with the original d2l-en, that JAX deliberately omits them in favour of the unified `Trainer` flow, and that PT only ships the subset useful outside the Trainer; MX/TF retain `evaluate_accuracy` because earlier-chapter snippets call it directly.
- `recurrent-modern/gru.md`: PT `GRU.__init__` takes `num_inputs`; MX/TF do not (they infer). PT could be made to also infer (`nn.LazyRNN`).
**SKIPPED** — `nn.LazyRNN` does not exist in PyTorch (only `nn.LazyLinear` / `nn.LazyConv*` / `nn.LazyBatchNorm*` / `nn.LazyInstanceNorm*`). Making PT GRU truly lazy would require a custom subclass that defers `nn.GRU` construction to first call — too invasive for stylistic alignment.
- `recurrent-neural-networks/rnn-scratch.md`: `clip_gradients` API differs PT/MX (in-place modify) vs TF/JAX (return new). By design — functional vs imperative.
**SKIPPED**
- `multilayer-perceptrons/dropout.md`: TF tab fuses `activation='relu'` into `Dense`; PT separates `nn.LazyLinear + nn.ReLU`. Both produce the same network, but the layer-count differs.
**SKIPPED**
- `builders-guide/init-param.md`: TF requires rebuilding `Sequential` for new init; PT/MX use `.apply(init_fn)`; JAX passes initializers to constructors. Three different patterns, no comparative section.
**SKIPPED**
- `builders-guide/read-write.md`: TF saves `.h5`, PT saves `.params`, JAX uses a checkpoint dir. No cross-framework comparison.
**SKIPPED**
- `recurrent-modern/seq2seq.md`: PT encoder uses `int64`, decoder used to use `int32` — DONE (decoder switched to int64 in commit `1afa0fb`). The MX/TF/JAX side may still be mixed.
**DONE `ad76f3c`** — JAX encoder + decoder now `d2l.astype(..., d2l.int64)` (was `d2l.int32`), matching PT. MX and TF do not call `astype` at all in seq2seq (they pass the raw transposed tensor to `nn.Embedding` / `tf.keras.layers.Embedding`, which accept either dtype); flagged for awareness, no change made.
- `multilayer-perceptrons/dropout.md`: dropout-key default — DONE (commit `e16b3da`).
**SKIPPED**

### Data-pipeline drift

- `recommender-systems/movielens.md`: PT `drop_last=False` (commit `6310018`) — partial fix; MX still uses `last_batch='rollover'` which has slightly different semantics. The PT change reduces the gap but doesn't close it.
**DONE `ad76f3c`** — MX `split_and_load_ml100k` switched from `last_batch='rollover'` to `last_batch='keep'` (matches PT `drop_last=False`: keep the partial last batch as-is each epoch, no roll-over to the next epoch). Updated the surrounding prose accordingly. Affects `d2l.split_and_load_ml100k` for MX after `make lib`.
- `recurrent-neural-networks/*`: shared discussion-thread IDs across `text-sequence.md` / `language-model.md` / `sequence.md` (117, 118, 1049 appear in multiple chapters). Need correct IDs from discuss.d2l.ai maintainers.
**IGNORED** for now.
- `generative-adversarial-networks/gan.md` JAX: discussion-thread URL points to PT URL (1082). Need correct JAX ID.
**IGNORED** for now.

### Implementation specifics

- `generative-adversarial-networks/dcgan.md` JAX: `use_running_average=False` is hardcoded as a class attribute and never overridden, so the BN-equipped generator is always in "training mode". DCGAN-specific convention says you can leave the generator in training mode at sampling time, so this is arguably correct, but readers will be confused.
**DONE `ad76f3c`** — added a clarifying comment in the JAX `G_block` explaining that DCGAN's convention is to keep the generator in training-mode BatchNorm even at sampling time, with a hint that users can override via `Generator(use_running_average=True)` for population-statistics inference.
- `natural-language-processing-applications/sentiment-analysis-rnn.md` TF BiRNN: `outputs[:, 0, :]` and `outputs[:, -1, :]`. PT/MX use `outputs[0]` and `outputs[-1]` (time-major). Semantically equivalent — different layout convention.
**IGNORED**
- `computational-performance/multiple-gpus-concise.md` PT: `nn.DataParallel` deprecation note added (commit `2642674`). The actual migration to `DistributedDataParallel` would be a large rewrite.
**IGNORED**
- `natural-language-processing-pretraining/bert.md` (TF): `BERTEncoder` calls `TransformerEncoderBlock` with 9 positional args while other tabs pass 5. The TF `TransformerEncoderBlock` signature has `key_size, query_size, value_size, num_hiddens, norm_shape, ...`; aligning across frameworks is a non-trivial refactor.
**IGNORED**

### Notebook execution quirks

- `recurrent-neural-networks` chapter: triage flagged TF/JAX/MX notebooks with `execution_count: null` in places. This is a build artifact — the notebooks ARE executed in the regular pipeline; the null counts come from `make notebooks` regenerating from source between runs. Not a source bug.
**IGNORED**
- `convolutional-neural-networks/lenet.md`, `pooling.md`: triage said TF cells 3-5 / 5-7 lack output. Latest TF runs (Apr 26) show all those cells have outputs. Stale claim.
**IGNORED**

### Recommender-systems

- `recommender-systems/mf.md`: PT evaluator was using mean-of-batch-means RMSE; switched to proper sum-of-squares accumulator (commit `018f4ea`).
**DONE `20e94c2`** — addressed as part of the recsys notation pass: `argmin` scope clarified to `b_*` since the objective sums over `b_u` and `b_i`. No other text ambiguities found in the prose.
- `recommender-systems/neumf.md`: PT `NeuMF.__init__` no longer takes `**kwargs` (commit `018f4ea`). MX still does (intentional — `nn.Block` accepts kwargs).
**IGNORED**
- `recommender-systems/ranking.md`: MX-only prose now wrapped in `:begin_tab:` (commit `018f4ea`).
**IGNORED**
- `recommender-systems/seqrec.md`: PT training cell now has the same "takes >1h" comment as MX (commit `6d6c62b`).
**DONE `4f9cc17`** — see the seqrec entry under "Recommender-systems / sequence-aware" above. Training is now actually running (PT 53s, MX 4.3min) with the per-epoch evaluation deferred via `eval_step=num_epochs`.

---

## Items I'd recommend tackling next

After the 2026-04-26 third pass, the principal remaining items are
genuinely architectural / additive content. None block correctness:

1. **bert.md TF arg-count alignment** (cross-framework drift). The 9-arg TF signature was intentional but creates a long-term maintenance burden. A focused refactor of `TransformerEncoderBlock` TF to match the 5-arg MX/PT/JAX signature would close one of the bigger architectural gaps in the book.
2. **Discussion-thread IDs** for JAX-newly-added chapters. Needs the discuss.d2l.ai maintainer to allocate IDs.
3. **JAX `clip_gradients` / `SGD` callouts**: short prose blocks explaining why the JAX functional pattern looks more verbose. (`fit_epoch` callout *is* now in `linear-regression-scratch.md` per `20e94c2`.)
4. **fine-tuning.md / fcn.md** truly cross-framework pretrained ResNet: requires either porting PyTorch ResNet-18 weights to JAX/TF or pulling in an external library (`flaxmodels`, `keras_hub`, `tensorflow_hub`). The current fix documents the gap inline; closing it adds a dependency.
5. **`mlp.md` / `backprop.md` notation homogenization**: row-vector vs column-vector, batch-first vs per-example. Large rewrite, needs author judgment.
6. **TF/JAX seqrec implementations**: requires porting `Caser`, `BPRLoss`, `SeqDataset`, `train_ranking`, `evaluate_ranking` to TF and JAX. The PT implementation (now actually executing) makes a clean reference.

---

*Items not in this file are addressed in commits `1afa0fb` through
`f0702b2`. The 2026-04-26 third pass landed in `20e94c2`, `ad76f3c`,
and `4f9cc17`. See `docs/deep-scan.md` "Fixes Applied (2026-04-26 —
third pass)" for the full per-commit breakdown.*
