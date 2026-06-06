# Code-quality deepdive: d2l-neu notebooks

A critical review of the Python code across all chapters and the shared `d2l`
library, focused on **poor teaching**, **poor efficiency**, **poor coding
practices**, and **per-framework idiom** (with extra weight on PyTorch and JAX,
as requested). The goal of this document is triage: decide what's worth fixing.

## How this was produced

- Reviewed the Python in all 208 source `.md` files (`chapter_*/*.md`) plus the
  generated library (`d2l/torch.py`, `d2l/jax.py`, `d2l/tensorflow.py`,
  `d2l/mxnet.py`). Library fixes belong in the `#@save` source blocks, not the
  generated `.py`.
- Findings are tagged `[severity] [category] [framework(s)]` with a
  `file:line`, a concrete fix, and — where relevant — the teaching↔efficiency↔
  practice tradeoff. The book deliberately trades idiom for clarity in places;
  where that's the right call it's flagged as "keep, but…", not as a defect.
- **Verification:** the headline JAX-jit finding and the four highest-stakes
  correctness claims were checked by hand against the source and the committed
  output store. One automated finding was **refuted** on inspection — see
  *Verification notes* at the end. Findings I did **not** independently confirm
  are marked *(reported, unverified)* so you can weight them accordingly.

Severity = impact × blast-radius. A `[high]` in the library affects every
notebook in a framework; a `[high]` in one chapter is a real bug or a
genuinely-misleading lesson confined to that page.

---

## TL;DR — the five things worth deciding on first

1. **The JAX `Trainer` never JITs the forward/backward.** Every JAX training
   notebook in the book runs the model forward, loss, and autodiff *eagerly,
   op-by-op*, because `Module.training_step` calls `self.plot(...)` (a host
   side-effect) inside the step. Only the optimizer *apply* is jitted. This is
   the single biggest, highest-leverage issue — one library fix speeds up the
   entire JAX track. **Confirmed by hand.**
2. **Legacy PyTorch `.data` surgery is pervasive** — in the library and ~10
   chapters — for init, in-place SGD, gradient zeroing, and even just reading
   `.device`. It's mostly benign but it's the exact footgun the autograd chapter
   steers readers away from, taught as normal practice (and in one slide,
   recommended as best practice).
3. **JAX PRNG-key hygiene contradicts what the book teaches.** Fixed
   `PRNGKey(0)` defaults, at-import-time mutable default args (`key=d2l.get_key()`
   evaluated once), and a from-scratch "dropout" that applies the **same mask
   every step**. The `ndarray`/`oo-design` chapters preach key-splitting; several
   later cells model the anti-pattern.
4. **A short list of genuine correctness/teaching bugs** (not style): GP prior
   band uses variance instead of its square root; the JAX FCN "pretrained"
   backbone is randomly initialized; a JAX BiRNN concatenates the wrong endpoint;
   the two-GPU slide claims a speedup the chapter body spends a page explaining is
   a *slowdown*; Adam's ε is stated three different ways.
5. **Cross-framework asymmetry that's invisible to a single-framework reader.**
   The clearest case: in the *same* from-scratch RNN/GRU/LSTM cells, the JAX tab
   uses `lax.scan` while PyTorch/TF/MXNet keep slow Python timestep loops, with no
   note that the difference is intentional.

The good news: no correctness bugs were found in the core math (conv/pool/BN/
residual, attention masking logic, the MDL closed-form-vs-numerical cross-checks),
the PyTorch `fit_epoch` `no_grad`/`train`/`eval` discipline is solid, and many
cells have clearly been modernized already (`weights_only=True`, `nn.init.*` in
`init-param`, out-of-place masking).

---

## Cross-cutting themes (ranked by leverage)

### Theme A — JAX training loop is not JIT-compiled  `[high] [efficiency] [JAX]`
**Where:** `d2l/jax.py:262-266` (`Module.training_step`), `367-394`
(`Trainer.fit_epoch`); only `_trainer_update` (line ~490) is `@jax.jit`.
**Why it matters:** the forward pass, loss, and `value_and_grad` run eagerly
every minibatch across the whole JAX track — the biggest single JAX perf win is
left on the table. The blocker is the `self.plot("loss", l, ...)` host call
*inside* the step, which can't be traced.
**Fix:** factor a pure, jitted `train_step(state, batch) -> (state, loss)` (jit
the `value_and_grad` + `apply_gradients` together — the chapter-13 helpers
`train_batch_ch13`/`train_concise_ch11` already do exactly this), and move
`self.plot(...)` out to the Python loop, plotting the returned scalar.
**Tradeoff:** the shown `training_step` becomes slightly less "one method does
everything," but the jitted pattern is already taught later in the book.

Two structural enablers sit right next to it:
- `d2l/jax.py:474-477` `SGD.init` returns the **class** `optax.EmptyState`, not
  an instance `optax.EmptyState()`. A real `GradientTransformation.init` must
  return a state pytree. **Confirmed.** This is what forces…
- `d2l/jax.py:371` `jit_ok = isinstance(self.state.opt_state, tuple)` — the
  Trainer decides whether to take the fast path by *sniffing the optimizer
  state's runtime type*. Fragile, and confusing plumbing in a `#@save` block.
  Fixing `SGD.init` lets both the heuristic and the duplicated non-jit branches
  (377-383, 389-393) collapse to a single `_trainer_update` call.

Also JAX-Trainer-specific: `clip_gradients` (`jax.py:414-419`) is **defined but
never called** — `Trainer(gradient_clip_val=...)` is a silent no-op on JAX,
which the RNN chapters rely on (torch/TF/MXNet all wire it in).

### Theme B — Legacy PyTorch `.data` surgery  `[med] [practice] [PyTorch]`
The classic footgun (bypasses autograd version-tracking; distinct from MXNet's
legitimate `weight.data()` accessor). Modern idiom is `with torch.no_grad():` +
`nn.init.*`, or `.detach()` for reads. Locations (representative, not
exhaustive):
- **Library:** `d2l/torch.py:377-378,391` (`LinearRegression` init + `get_w_b`)
  — the *first* concise model students see.
- `chapter_linear-regression/linear-regression-concise.md:154-155,370-371,378`
- `chapter_multilayer-perceptrons/mlp.md:504,579` (`x.grad.data.zero_()`)
- `chapter_builders-guide/init-param.md:328,388-390` and the recap **slide:551**
  (which *recommends* `layer.weight.data[...] =` as the surgery idiom)
- `chapter_builders-guide/use-gpu.md:564,569` (`.data.device` just to read device)
- `chapter_convolutional-neural-networks/conv-layer.md:375`;
  `chapter_convolutional-modern/batch-norm.md:334`
- `chapter_optimization/{momentum,adagrad,rmsprop,adam,adadelta,minibatch-sgd}.md`
  — from-scratch loops use `p.grad.data.zero_()`/`p.data.sub_()` *inside* an
  existing `with torch.no_grad():`, where `.data` is redundant.
- `chapter_computer-vision/fcn.md:488` (`conv_trans.weight.data.copy_(...)`)
**Fix:** mechanical sweep — drop `.data`; inside `no_grad` use the plain tensor;
for reads use `.detach()`/`.device`. Reframe slide 551 as "use `torch.no_grad()`
for one-off surgery."
**Tradeoff:** `.data` is one token shorter; that's the only thing it has going.

### Theme C — JAX PRNG-key hygiene  `[med→high] [idiom/practice] [JAX]`
The book teaches key-splitting in `ndarray.md`/`oo-design.md`, then several cells
do the opposite:
- **`chapter_multilayer-perceptrons/dropout.md:218`** — `dropout_layer(X, dropout,
  key=d2l.get_key())` binds **one fixed key at def-time**, so the from-scratch
  `DropoutMLPScratch` applies the **identical dropout mask on every step**. It
  isn't stochastic dropout at all — it undercuts the technique the section
  teaches. **Confirmed** (the comment openly acknowledges the fixed key, citing
  JIT-traceability). `[high]` on teaching grounds. *Fix:* thread a per-step key
  (split `state.dropout_rng`, as the concise tab already does) or fold in the
  step index; if kept, say plainly "this mask is fixed; real dropout needs a
  fresh key — see `nn.Dropout`."
- `chapter_preliminaries/probability.md:261-414` — the JAX coin-toss tab
  abandons JAX entirely and falls back to the **unseeded global**
  `np.random.multinomial` (comment: "jax.random does not have multinomial").
  `jax.random.categorical` exists. This is the chapter *about sampling*. `[high]`.
- **Library mutable defaults:** `d2l/jax.py:426`
  (`SyntheticRegressionData(..., key=jax.random.PRNGKey(0))` — every
  default-built dataset is identical) and `jax.py:594`
  (`layer_summary(..., key=d2l.get_key())` — evaluated once at class-def time,
  advancing `_master_key` as a side effect). Also
  `chapter_linear-regression/synthetic-regression-data.md:115`.
- Hardcoded `PRNGKey(0)` in recurrent `initialize_carry`: `d2l/jax.py:943-944,
  985-986` and `rnn-concise.md:136`, `gru.md:431`, `lstm.md:562`. Harmless for
  zero-carries but copied widely; `GRU` also builds `[carry]*num_layers` (aliased
  same object — latent bug if a carry is ever mutated).
**Fix:** default `key=None` then `key = d2l.get_key() if key is None else key`;
thread keys through dropout/carry; never call `np.random` global in a JAX tab.

### Theme D — Per-step host syncs and eager element loops  `[med] [efficiency]`
Device→host transfers or Python loops in hot paths:
- `d2l/torch.py:206` (`Module.plot`) — `d2l.numpy(d2l.to(value, d2l.cpu()))`
  forces a **GPU→CPU sync of the loss every batch**, before the `every_n`
  filter. The JAX version (`jax.py:254-257`) was already fixed to skip the sync
  unless the point will be plotted; torch was not. *Fix:* mirror the JAX guard.
- `chapter_computational-performance/multiple-gpus.md:681` — `.item()` inside the
  per-batch eval loop (the chapter literally teaches against this).
- `chapter_computer-vision/anchor.md` — `assign_anchor_to_bbox` (566-603) and
  `nms` (1054-1120) are per-box Python `while`/`for` loops with in-loop `argmax`
  (a CPU↔GPU sync each iteration on PyTorch), wrapped in a per-image batch loop.
  Presented as the canonical implementation. *Fix:* at minimum point to
  `torchvision.ops.batched_nms`; note the cost. *Tradeoff:* the loop mirrors the
  algorithm — keep it, but flag it and don't let readers cargo-cult it.
- `chapter_recommender-systems/neumf.md:223-250` — `evaluate_ranking` rebuilds
  sets, a fresh `DataLoader`, and Python `sorted` of score tuples per user; it's
  why eval is deferred to the final epoch. *Fix:* batch users, `torch.topk`.
- `chapter_linear-regression/linear-regression.md:446-474` — the
  "loops are slow" demo runs 10,000 scalar iterations; the **JAX** tab's
  `c = c.at[i].set(...)` rebuilds a 10k array every iteration (O(n²) alloc) — the
  point survives at n=1000. *Tradeoff:* deliberately-bad code; just bound it.

### Theme E — Missing `no_grad`/`eval` at inference  `[med] [idiom] [PyTorch]`
Inference cells build autograd graphs / run BN in train mode:
- `chapter_linear-classification/softmax-regression-scratch.md:357-382` (predict
  without `no_grad`).
- `chapter_computer-vision/fcn.md:763-766` — `predict` never calls `net.eval()`,
  so a ResNet/BN net runs inference in train-mode BN (the SSD tab *does* call
  `eval()` at `ssd.md:1382` — inconsistent).
- `d2l/torch.py:439-446` (`Classifier.accuracy`) and `751-762`
  (`RNNLMScratch.predict`) lack `@torch.no_grad()`; they're safe only because
  callers happen to wrap them. A reused metric helper should self-protect.
**Fix:** `@torch.no_grad()` on `accuracy`/`predict`; `net.eval()` in `fcn` predict.

### Theme F — Cross-framework asymmetry presented without comment  `[med] [teaching]`
A single-framework reader can't see these, but a comparer is confused:
- **RNN/GRU/LSTM from-scratch:** JAX uses `jax.lax.scan`, PyTorch/TF/MXNet keep
  Python timestep loops in the *same* cells (`gru.md:267-276`, `lstm.md:359-371`,
  `rnn-scratch.md:142-146`). *Fix:* one prose line that the explicit loop is the
  readable teaching form and JAX needs `scan` for compile time.
- TF cross-entropy uses `tf.boolean_mask(y_hat, tf.one_hot(y))` while the other
  three use direct integer gather (`softmax-regression-scratch.md:242,292`) — the
  prose describes the *indexing* version. *Fix:* `tf.gather(y_hat, y, batch_dims=1)`.
- TF from-scratch `fit_epoch`/`BatchNorm` carry `@tf.function`/`reduce_retracing`
  machinery the other three don't (`linear-regression-scratch.md:488-534`,
  `batch-norm.md:502`), making the "same loop, four frameworks" pedagogy lopsided.

### Theme G — Inline figure-drawing in notebooks  `[low→med] [teaching]`
The project's own rule: illustrative figures are pre-generated SVGs, not drawn
inline ("a wall of matplotlib that teaches nothing"). Offenders:
- `chapter_preliminaries/calculus.md:244-270` — ~40 lines of `plot`/`set_axes`/
  `use_svg_display` plumbing inline (they're `#@save` — move to the library, keep
  only the cell that *uses* `plot`).
- `chapter_mdl-calculus/mdl-integral-calculus.md:431-482` — a 3-D wireframe
  surface drawn inline; only the 2-line Riemann volume computes anything.
- `chapter_mdl-single-variable-calculus.md:209-261,391-447` — tangent/parabola
  overlays drawn inline while the same chapter pre-generates other figures (mixed
  style within one chapter).
- `chapter_multilayer-perceptrons/mlp.md:323-595` — activation derivatives
  computed via full autograd backward passes purely to *plot a known curve*
  (JAX already does the elegant `vmap(grad(...))`; the others don't).

### Theme H — Deprecated / fragile APIs  `[low] [practice]`
- `nn.DataParallel` shipped as the "concise" multi-GPU reference
  (`multiple-gpus-concise.md:330`) — effectively deprecated in favor of DDP.
- Flax `flax.training.checkpoints` + defensive `flax.core.freeze` on restore
  (`read-write.md:320-321`) — deprecated in favor of Orbax.
- Keras legacy H5 weights (`read-write.md:283,314`).
- `type(module) == nn.Linear` dispatch (`init-param.md`) works *only* because
  `LazyLinear` rewrites its `__class__` after the first forward — silent no-op on
  a fresh net. *Fix:* `isinstance`.
- Legacy global NumPy RNG (`np.random.seed` + `np.random.*`) mixed with the
  modern `default_rng` Generator within the MDL probability chapter
  (`mdl-distributions.md:111,586,752`, `mdl-statistics.md:430`).
- Unclosed `multiprocessing.Pool(4)` in `natural-language-inference-bert.md`
  (419-420, 473-474, 583-584) — leaks workers; use `with`.

---

## Confirmed correctness / actively-misleading bugs (the short list)

These are *wrong*, not stylistic. Worth fixing regardless of the idiom debate.

| Bug | Location | Status |
|---|---|---|
| GP prior credible band uses variance, not its √ — `mean ± 2*np.diag(cov)` should be `± 2*np.sqrt(np.diag(cov))`. Correct *only* because `rbfkernel` has unit amplitude (`d2l/torch.py:2556` has no amplitude term, so `diag==1`); the prose says "two times the square root of our variance." Re-check the posterior band too. | `chapter_gaussian-processes/gp-inference.md:138` | **Confirmed** |
| JAX FCN "pretrained" backbone (`ResNetFeatures`) is **randomly initialized** — the entire transfer-learning premise is silently false for the JAX tab; output masks are noise. | `chapter_computer-vision/fcn.md:101-181,322-342` | *(reported)* |
| JAX `BiRNN` takes `backward_out[:, 0, :]`, but `nn.RNN(reverse=True)` re-reverses its output to input order, so index 0 is the backward pass at the *first* token, not its final state — wrong endpoint vs. PyTorch/MXNet. | `chapter_natural-language-processing-applications/sentiment-analysis-rnn.md:185-207` | *(reported — needs a length-checked toy test to confirm Flax's reverse semantics)* |
| Two-GPU slide claims per-epoch time "roughly halves," but the chapter body spends a page correctly explaining the run is a **~30% regression** (tiny model, naive allreduce). The slide teaches the opposite. | `chapter_computational-performance/multiple-gpus.md:916-921` vs `764-770` | *(reported)* |
| Adam ε stated three ways: prose `1e-6` (`adam.md:31`), code `eps=1e-6` (`:56`), slides `1e-8` (`:326,363`). Framework default is `1e-8`. | `chapter_optimization/adam.md` | *(reported)* |
| Sentiment-RNN freezes the embedding (`requires_grad=False`) but the slide says "we fine-tune." | `sentiment-analysis-rnn.md:307-331` vs slide `:558` | *(reported)* |
| Flax NiN model: `num_classes = 10` written without a type annotation → it's a class constant, not a dataclass field, so `NiN(num_classes=...)` can't override it (every other model annotates it). | `chapter_convolutional-modern/nin.md:205` | *(reported — high-confidence Flax semantics)* |

---

## Detailed findings by area

The per-area sections below are the raw reviewer output, lightly edited. Use the
themes above for triage; use these for the specifics when you fix.

### Preliminaries + Linear Regression
**Overall:** Mature, didactically careful, mostly idiomatic. Substantive issues
concentrate in JAX (probability.md global-RNG fallback; PRNG defaults). PyTorch
scratch/concise pair is in good shape; the hand-rolled SGD is correctly framed as
pedagogy.
- `[high][idiom][JAX]` `probability.md:261-414` — JAX coin-toss falls back to
  unseeded global `np.random.multinomial`; contradicts the key discipline taught
  two chapters earlier. *Fix:* `jax.random.categorical` + `bincount`, threaded key.
- `[med][idiom][JAX]` `synthetic-regression-data.md:115` — `key=jax.random.PRNGKey(0)`
  default bakes a fixed key into the signature; every default dataset is identical.
- `[med][teaching][JAX]` `linear-regression-scratch.md:536-592` — JAX `fit_epoch`
  is ~55 lines with two JIT helpers + a `jit_ok` probe + BN/dropout branches, in
  the *from-scratch* chapter whose stated goal is "understand this code fully."
  *Fix:* show a minimal JAX `fit_epoch` here; defer BN/JIT-fallback to batch-norm.
- `[med][practice][PyTorch]` `linear-regression-concise.md:154-155,370-371,378` —
  `.data` init + `get_w_b` returns `.data`. *Fix:* `no_grad` + `nn.init.*`; `.detach()`.
- `[med][efficiency][all]` `linear-regression.md:446-474` — 10k-iter scalar-index
  loop; JAX `.at[i].set` is O(n²). *Fix:* bound the loop length.
- `[med][teaching][all]` `calculus.md:244-270` — inline matplotlib plumbing
  (`#@save`); move to library.
- `[low][practice][JAX]` `ndarray.md:803-804` — `X.at[:].set(X+Y)` shown as the
  "saving memory" analogue, but it's a full copy whose `id` always differs;
  demonstrates the opposite. *Fix:* reframe as "JAX has no in-place write," or show
  `jit(..., donate_argnums=0)`.
- `[low][idiom][JAX]` `autograd.md:479-482` — recomputes `y(x)` twice; bind
  `u = stop_gradient(y(x))` once.

### Linear Classification + MLP
**Overall:** Clean and compact; framework parity high. JAX tab has the most
issues (fixed-key dropout, eager fancy-indexing) plus a couple legacy PyTorch idioms.
- `[high][idiom/practice][JAX]` `dropout.md:218` — fixed-key dropout (see Theme C).
- `[high][idiom/efficiency][PyTorch,JAX]` `softmax-regression-scratch.md:357-382` —
  prediction runs without `no_grad`/un-jitted.
- `[med][idiom][JAX]` `softmax-regression-scratch.md:283,315` —
  `y_hat[list(range(len(y_hat))), y]` inside a jitted `loss`. *Fix:*
  `jnp.take_along_axis(y_hat, y[:,None], 1)`.
- `[med][practice][PyTorch]` `mlp.md:504,579` — `x.grad.data.zero_()` + reliance
  on `retain_graph=True`.
- `[med][teaching][PyTorch,all]` `mlp.md:323-595` — activation derivatives via
  autograd backward just to plot a known curve.
- `[med][efficiency/idiom][PyTorch]` `softmax-regression-scratch.md:360` /
  `classification.md:165` — argmax recomputed at predict; no `no_grad`.
- `[med][teaching][TF]` `softmax-regression-scratch.md:242,292` — `boolean_mask`+
  `one_hot` where `tf.gather(..., batch_dims=1)` matches the other tabs.
- `[low]` items: `mlp-implementation.md:162` `torch.clamp` vs `maximum` parity;
  `kaggle-house-price.md:304-309` rebuilds tensors per `get_dataloader` call + a
  silent `None` return; `classification.md:213-219` `dir(self)` scan should be
  `vars(self)`; `image-classification-dataset.md:236-239` `NotImplementedError`
  stub shown as the `#@save` function.

### Builders' Guide
**Overall:** Largely modernized and idiomatic. Weak spots are JAX/Flax cells and
a few PyTorch idiom slips. **Note:** the "tied parameters" finding was
**refuted** on inspection — see Verification notes.
- `[high][practice][JAX]` `model-construction.md:461-467` — `MySequential` stores
  `modules: List` and iterates them as raw callables; depending on Flax version
  this doesn't register children the way the prose promises. *Fix:* assign in
  `setup()` / use `@nn.compact`.
- `[med][idiom][PyTorch]` `init-param.md:328,388-390` + slide `551` — `.data`
  surgery taught as the recommended idiom (Theme B).
- `[med][practice][PyTorch]` `init-param.md` — `type(m)==nn.Linear` dispatch is
  fragile w/ LazyLinear; use `isinstance`.
- `[med][teaching][PyTorch]` `custom-layer.md:204-206` — comment says
  "(Xavier-ish)" but `randn/in_units**0.5` is LeCun/Kaiming-fan_in, not Xavier.
- `[med][idiom][PyTorch]` `model-construction.md:597-602` — `FixedHiddenMLP`
  stores a fixed weight as a bare attribute; should be `register_buffer` (else
  `.to('cuda')` leaves it on CPU → device-mismatch; the chapter's own use-gpu
  slide warns about exactly this).
- `[med][idiom][JAX]` `model-construction.md:651` — `FixedHiddenMLP.setup` draws
  a freshly-keyed "constant" each trace; not in the param tree (not reproducible).
- `[med][teaching][JAX]` `init-param.md:151-208` — `bias_init` defined in one cell
  and relied on in later cells → `NameError` out of order; other frameworks are
  self-contained.
- `[low]` items: `use-gpu.md:392,459` `.cuda(1)` vs the `.to()` idiom used
  everywhere else; `use-gpu.md:564,569` `.data.device`; `parameters.md:296-299`
  `.data` in an `assert` + bare assert renders no visible output; `read-write.md`
  H5 weights + Flax `checkpoints`/`freeze` deprecation; `model-construction.md:441`
  iterate registered modules not `children()` + trailing whitespace.
- *No MXNet findings* — `weight.data()` etc. is the genuine Gluon API, and the
  `_layers` strong-ref handling is a correct, well-commented Gluon-2.0 fix.

### CNNs (classic + modern)
**Overall:** High-quality and pedagogically careful. Issues concentrate in JAX
(un-jitted teaching loops, the NiN field bug, per-call pool-window recompute) and
a couple of legacy PyTorch `.data` idioms. No correctness bugs in the core math.
- `[med][idiom][JAX]` `nin.md:205` — `num_classes = 10` unannotated → not a field.
- `[med][practice/idiom][PyTorch]` `conv-layer.md:375` — `.data[:] -=` in-place SGD.
  *Fix:* keep the explicit loop, use `no_grad`.
- `[med][efficiency/idiom][JAX]` `conv-layer.md:137-144`, `pooling.md:154-163` —
  `corr2d`/`pool2d` double Python loop with `.at[i,j].set` (un-jitted, O(H·W) ops);
  note it's illustrative, optionally show the vectorized form.
- `[low][efficiency][JAX]` global-avg-pool lambdas recompute `window_shape` from
  input every call (`nin.md:218`, `resnet.md:476`, `densenet.md:491`,
  `googlenet.md:464`, `cnn-design.md:252`). *Fix:* `lambda x: x.mean(axis=(1,2))`.
- `[low]` items: `conv-layer.md` init-sensitivity note; `batch-norm.md:502`
  `@tf.function` on a stateful scratch layer; `padding-and-strides.md` three
  padding spellings in one section; `batch-norm.md:334` `.data` returns;
  `lenet.md:69` `FunctionType` annotation (use `Callable`); channels.md op-count
  wording mismatch.

### Sequence models (RNN, modern RNN, attention/transformers)
**Overall:** Clean and consistent; most loops are intentional teaching artifacts.
Substantive issues: the JAX-only `lax.scan` asymmetry, attention masking via
`-1e6`, the O(T²) decode loop, and JAX PRNG/scan handling.
- `[high][correctness][JAX]` recurrent `initialize_carry` hardcodes `PRNGKey(0)`
  (`rnn-concise.md:136`, `gru.md:431`, `lstm.md:562`).
- `[high][efficiency][JAX]` `deep-rnn.md:306-308` — `[init_carry(...)] *
  num_layers` aliases one object across layers.
- `[high][efficiency][PyTorch/MXNet/TF]` `gru.md:267-276`, `lstm.md:359-371`,
  `rnn-scratch.md:142-146` — Python timestep loop where the JAX tab uses
  `lax.scan`; flag the asymmetry (Theme F).
- `[high][efficiency][all]` `transformer.md:772-800,1022-1041` — autoregressive
  decode re-runs attention over the whole prefix every step (O(T²), no KV cache);
  add one sentence that real decoders cache K/V.
- `[med][practice][PyTorch/JAX/TF]` `attention-scoring-functions.md:125-188` —
  masked softmax fills with literal `-1e6`; overflows/under-masks in fp16. *Fix:*
  `masked_fill(~mask, float('-inf'))` / `torch.finfo(dtype).min`.
- `[med][efficiency][all]` `seq2seq.md:472-574` — context broadcast via
  `tile`/`repeat` materializes a full copy across time; rely on broadcasting.
- `[med][correctness][PyTorch]` `attention-scoring-functions.md:512-513` — legacy
  `super(Class,self).__init__(**kwargs)` + dead `**kwargs`.
- `[med][practice][TF]` `sequence.md:546-550` — `tf.Variable` mutated with
  per-step `.assign` and retrace; use `tf.TensorArray`.
- `[low]` items: gradient clipping points to `clip_grad_norm_`; per-char host
  sync in greedy decode (unavoidable); `sow` of a constant `P` every forward
  (`self-attention-and-positional-encoding.md:342`); scattered unused MXNet
  imports; `deep-rnn.md:179-181` list→tensor re-stack wart.

### Optimization + Computational Performance
**Overall:** From-scratch optimizer loops are sound; cross-framework parity is
careful. Real problems are in the *perf* chapter (the two-GPU slide) and a few
JAX idioms.
- `[high][teaching][all]` `multiple-gpus.md:916-921` — slide contradicts body
  (Theme D / correctness list).
- `[med][practice][all]` `adam.md` — ε stated three ways.
- `[med][idiom][JAX]` `lr-scheduler.md:277` — in-place mutation of an optax state
  (`opt_state.hyperparams['learning_rate'] = ...`); works today, fragile.
- `[med][efficiency][JAX]` `multiple-gpus.md:681` — `.item()` per eval batch.
- `[med][idiom][JAX]` `async-computation.md:111-112` — timed loop reuses one key +
  drops device placement (inconsistent with warm-up; XLA may CSE identical data).
- `[low]` items: JAX from-scratch optimizers use `enumerate`+list-write (un-
  idiomatic but harmless); `sgd.md` noise-RNG differs per tab without a note;
  `.data` in all torch from-scratch loops (Theme B); `DataParallel` as the concise
  reference; `multiple-gpus-concise.md:420` epoch barrier blocks on an unrelated
  scalar, not `state`; `gd.md` hand-coded analytic gradients can drift from `f_2d`;
  `minibatch-sgd.md` double on-device residency.

### Computer Vision + NLP (pretraining + applications)
**Overall:** PyTorch/MXNet/TF faithful to upstream and idiomatic. Problems: per-
image Python loops in the anchor/NMS path presented as canonical; inference cells
missing `no_grad`/`eval`; and JAX ports that are teaching liabilities.
- `[high][practice][JAX]` `natural-language-inference-bert.md:103-185` — 80-line
  hand-written PyTorch-pickle reader shipped as notebook content. *Fix:* move to
  `d2l/`, or load via `safetensors`.
- `[high][teaching/practice][JAX]` `fcn.md:101-181,322-342` — random "pretrained"
  backbone (correctness list).
- `[high][idiom][JAX]` `sentiment-analysis-rnn.md:185-207` — BiRNN wrong endpoint
  (correctness list).
- `[high][efficiency][PyTorch/all]` `anchor.md:566-603,1054-1120` — per-box loops
  with in-loop `argmax` syncs (Theme D).
- `[med]` items: `natural-language-inference-attention.md` instantiates fresh
  `MLP` in `@nn.compact` (brittle positional names); `fine-tuning.md` scratch/
  fine-tune comparison not hyperparameter-matched across frameworks; `fcn.md`
  predict missing `eval()`/`no_grad`; `anchor.md:587/594-595` device-placement
  mismatch; unclosed `multiprocessing.Pool`; `anchor.md:650-666` TF label fns are
  eager-only (worth a caveat); slide/code freeze-vs-fine-tune contradiction.
- `[low]` items: unseeded global RNG in `bert-dataset.md` (irreproducible
  pretraining data); `word2vec-pretraining.md` JAX 4-arg `skip_gram` divergence;
  GloVe/BERT weight injection by string-keyed dict surgery (brittle); per-sample
  PIL round-trips in `image-augmentation.md` display grid; JAX SSD/NMS drop to
  numpy (can't jit — expected divergence, worth noting).

### Advanced topics (GP, RL, RecSys, GANs, HPO)
**Overall:** Generally clean and faithful; recent edits show care. Main issues: a
real GP plotting bug, several O(states/actions) Python loops in RL the chapter
could vectorize to teach the Bellman backup as linear algebra, and a few opaque
dense-numerics cells.
- `[high][teaching][PyTorch]` `gp-inference.md:138` — variance vs √variance band
  (correctness list).
- `[med][efficiency/teaching][PyTorch]` `value-iter.md:140-152` — 4-deep Python
  loop for the Bellman backup; vectorize to `Q = R + γ·(P@V)` to match the slide.
- `[med][efficiency][PyTorch/NumPy]` `neumf.md:223-250` — per-user eval rebuild
  (Theme D).
- `[med][idiom][PyTorch]` `gp-inference.md:255-259,320` — `torch.tensor()` on
  existing arrays (warning) + silent float64. *Fix:* `torch.from_numpy(...).float()`.
- `[med][practice][PyTorch]` `rs-async.md:103`, `sh-async.md:129` —
  `HPOTrainer(max_epochs=1)` then hand-cranking `fit_epoch()`; established idiom
  but fragile/confusing.
- `[med][teaching][MXNet/PyTorch]` `mf.md:148-224` — `train_recsys_rating` reuses
  `l` as both batch loss and accumulator; rename + drop the dead `train_l` path.
- `[low]` items: per-sample `random.randint` negative sampling (slow/irreproducible);
  `qlearning.md:104-153` mixed return types + per-episode all-states recompute "for
  visualization"; GAN `update_D/G` dead `loss_fn` param + `fake_X` recomputed 2–3×/
  step; `gp-priors.md` Python sample loop (vectorizable); `seqrec.md` shape-
  dependent bare `squeeze()`; `ranking.md` `log(sigmoid)` vs `F.logsigmoid` +
  non-scalar loss; `ctr.md` calls TSV data "csv"; `dcgan.md` runs D mutable in the
  G step then discards batch_stats.

### Mathematics for Deep Learning (MDL part)
**Overall:** The four *written* chapters (LA, Calculus, Probability & Statistics,
the main Information Theory file) have strong, vectorized, idiomatic teaching code
that computes rather than draws. Optimization, the two depth Information-Theory
files, and all of Dynamics are **ToC/outline stubs with zero Python** — nothing to
review there yet. No correctness bugs in the reviewed code (the closed-form-vs-
numerical cross-checks are all set up correctly and use stable primitives).
- `[med][idiom][JAX]` `mdl-eigendecomposition.md:895-896` — power iteration seeds
  `A` and `v` with two different integer keys; use one `split`.
- `[med][teaching][all]` `mdl-integral-calculus.md:431-482` — inline 3-D wireframe
  (Theme G).
- `[low]` items: tangent/parabola overlays drawn inline (`mdl-single-variable-
  calculus.md`); legacy global RNG vs `default_rng` (Theme H); `torch.tensor([...])`
  from tensor elements (`mdl-multivariable-calculus.md:126-134` — use `stack`);
  `nansum` masking in entropy/KL hides the `p>0,q=0→inf` case the prose discusses;
  naive-Bayes JAX tab imports TF, no jax; per-class Python loops where `bincount`/
  one-hot `@` is idiomatic; bare f-string expressions instead of `print`.

### Shared d2l library (training infrastructure)
**Overall:** Mostly clean; PyTorch side is hardened (correct `no_grad`/`train`/
`eval`, out-of-place masking). **The JAX jit verdict is the headline (Theme A) and
it's bad.** Secondary: `SGD.init` returns a class not an instance; the
`isinstance(opt_state, tuple)` jit heuristic; mutable/global `PRNGKey(0)` defaults.
- `[high][efficiency][JAX]` `jax.py:262-266,367-394` — forward/backward not jitted.
- `[high][idiom/practice][JAX]` `jax.py:474-477` — `SGD.init` returns class.
- `[high][teaching/idiom][JAX]` `jax.py:371` — `jit_ok` type-sniffing heuristic.
- `[med][practice][JAX]` `jax.py:426,594` — at-import mutable default keys.
- `[med][idiom][JAX]` `jax.py:943-986` — fixed-key + aliased-list carries.
- `[med][efficiency][JAX]` `jax.py:581-612` — jitting bound methods with `self`
  static → re-trace per model instance.
- `[med][efficiency][PyTorch]` `torch.py:206` — `Module.plot` GPU→CPU sync every
  batch (JAX was fixed, torch wasn't).
- `[med][idiom][PyTorch]` `torch.py:309-314` — hand-rolled grad clipping; note
  `clip_grad_norm_`. `torch.py:377-391` — `.data` init in `LinearRegression`.
  `torch.py:439-446` — `accuracy` lacks `@torch.no_grad()`.
- `[low]` items: `torch.py:1450` / `jax.py:8` / `jax.py:352-360` import-time
  side-effects (`try_all_gpus()` default arg, `jax.devices()` at import, `TrainState`
  redefined per `fit()`); `jax.py:414-419` `clip_gradients` defined but never called
  (JAX clipping is a silent no-op); TF `.numpy()` host syncs in clip/eval helpers;
  `torch.py:751-762` `RNNLMScratch.predict` no `no_grad`/`eval`.

---

## Verification notes (what I checked, and one correction)

- **JAX no-jit (Theme A):** confirmed by reading `d2l/jax.py:262-394` — only
  `_trainer_update` carries `@jax.jit`; the `value_and_grad` runs eagerly.
- **Fixed-key dropout (`dropout.md:218`):** confirmed — the default `key` is bound
  once at def-time; the comment openly documents the fixed key.
- **GP prior band (`gp-inference.md:138`):** confirmed — `rbfkernel`
  (`d2l/torch.py:2554-2556`) has no amplitude term, so `diag(cov)==1` and the plot
  is right only by coincidence; the formula omits the required √.
- **CORRECTION — "tied parameters" (`parameters.md:320-334`):** a reviewer flagged
  the JAX cell as "not demonstrating tying at all." **This is wrong.** The committed
  output (`outputs/jax/chapter_builders-guide/parameters.json`) shows the cell
  prints `True`, i.e. `len(params['params']) == 3` for 4 Dense applications —
  Flax *did* collapse the two `shared` call-sites into one param entry, which is
  exactly tying. The only fair criticism is pedagogical: the JAX demo proves tying
  *indirectly* (a count) where PyTorch shows it *directly* (`is`-identity, then
  mutate-and-observe). Downgraded to `[low][teaching]`: optionally strengthen the
  JAX demo to mutate the shared weight and show both sites change.
- Findings marked *(reported, unverified)* are factual code reads or doc/slide
  consistency claims I trust but did not separately re-run; the **BiRNN endpoint**
  one specifically depends on Flax `nn.RNN(reverse=True)` output ordering — confirm
  with a length-checked toy sequence before changing the model.

---

## Suggested triage order

**Tier 1 — high leverage, do first (library `#@save` blocks):**
1. JAX `Trainer`: jit the train/val step + move `plot` out (Theme A). Speeds up
   the entire JAX track. Also wire in `clip_gradients` (currently a no-op).
2. `SGD.init` → return `optax.EmptyState()`; delete the `jit_ok` heuristic and the
   duplicated non-jit branches.
3. `torch.py` `Module.plot`: gate the GPU→CPU sync behind `every_n` (mirror JAX).
4. Library mutable/global default keys → `key=None` pattern.

**Tier 2 — mechanical sweeps (high count, low risk each):**
5. Drop legacy `.data` surgery across library + chapters; use `no_grad`/`nn.init.*`/
   `.detach()` (Theme B).
6. `isinstance` instead of `type()==` for module dispatch.
7. Add `@torch.no_grad()`/`net.eval()` to inference helpers and predict cells.

**Tier 3 — confirmed correctness/teaching bugs (small, high-value):**
8. GP prior band √; Adam ε consistency; two-GPU slide vs body; sentiment-RNN
   freeze-vs-fine-tune slide; NiN `num_classes` annotation.
9. Confirm-then-fix: JAX FCN random backbone; JAX BiRNN endpoint.

**Tier 4 — teaching polish (judgment calls, respect the clarity tradeoff):**
10. Fixed-key dropout honesty note (or per-step key); cross-framework `lax.scan`
    asymmetry note; attention `-1e6`→`-inf`; KV-cache note in transformer decode;
    move inline figure-drawing to generators (Theme G); JAX PRNG-key hygiene in
    `probability.md` and recurrent carries.

Nothing here blocks the build — these are quality, not breakage. The Tier 1 JAX
jit fix is the one with outsized impact (correctness of the "JAX is fast" story
across the book) for a single, contained change.
