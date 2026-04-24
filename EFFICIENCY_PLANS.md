# Slow-Notebook Efficiency Plans

Plans to speed up notebooks with runtimes >200s. **Nothing is implemented yet** — each plan is for user review. Principle: simplicity ≥ speed (textbook context).

## Status summary from `logs/make-all.log`

| Framework | Passed | Failed | Notes |
|-----------|--------|--------|-------|
| PyTorch | 139/139 | 0 | — |
| TensorFlow | 91/91 | 0* | `sentiment-analysis-cnn.ipynb` — my `add5fc9` commit mislabeled a cell `#@tab mxnet, pytorch, tensorflow`; reverted in this session. TF notebook was never intended for this file. |
| JAX | 122/123 | 1 flake | `fine-tuning.ipynb` — cuDNN autotune failure; the "JAX" version actually uses `tf.keras` internally (pre-existing, not my bug). |
| MXNet | 128/128 | 0 | — |

JAX slide `channels.qmd` had one flaky assertion failure; re-run passed. JAX slide `fine-tuning.qmd` failed for the same reason as the notebook.

Bugs from my earlier changes: the TF tab regression above. Already fixed in this session. No other regressions found.

---

## Part A — PyTorch slow notebooks (8 total)

### Justified (no change)

| Notebook | Time | Reason |
|---|---|---|
| `chapter_hyperparameter-optimization/sh-intro.md` | 460s | 30 trials × LeNet; pedagogical HPO demo |
| `chapter_hyperparameter-optimization/hyperopt-intro.md` | 353s | 5 trials × softmax reg × 8 epochs |
| `chapter_hyperparameter-optimization/hyperopt-api.md` | 251s | 5 trials × LeNet × 10 epochs — API demo |
| `chapter_optimization/lr-scheduler.md` | 375s | 5 schedulers × 30 epochs LeNet; needed to show scheduler effects |
| `chapter_convolutional-modern/vgg.md` | 240s | Smallest VGG the author thought was reasonable |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.md` | 315s | 5 epochs BERT-small × ~1100 steps — BERT is inherently slow |

### Proposed changes

#### 1. `chapter_hyperparameter-optimization/sh-async.md` (311s → ~120s)

**Current**: `max_wallclock_time = 5 * 60` (300 seconds). Notebook literally states "will run for about 12 minutes".

**Plan**: Change line 139: `max_wallclock_time = 2 * 60`. Async scheduling demo doesn't need 5 minutes to be pedagogical.

**Why**: The notebook is a wall-clock-bounded demo, not a results-bounded one. Cutting wallclock in half still demonstrates the algorithm.

#### 2. `chapter_hyperparameter-optimization/rs-async.md` (310s → ~120s)

**Plan**: Same. Change line 99: `max_wallclock_time = 2 * 60`.

---

## Part B — JAX slow notebooks (25 total)

### B.1 Systemic issue: `d2l.Trainer` for JAX

~15 JAX CNN/Transformer notebooks (alexnet, vgg, nin, googlenet, resnet, densenet, batch-norm, cnn-design, lenet, vision-transformer, nli-attention, and others) share a single bottleneck in the JAX `Trainer.fit_epoch`, defined in `chapter_linear-regression/linear-regression-scratch.md` (`#@tab jax` block around lines 503-545).

**Current** (reconstructed):
```python
for batch in self.train_dataloader:
    (_, mutated_vars), grads = self.model.training_step(
        self.state.params, self.prepare_batch(batch), self.state)
    self.state = self.state.apply_gradients(grads=grads)
    self.state = self.state.replace(
        dropout_rng=jax.random.split(self.state.dropout_rng)[0])
    self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
    self.train_batch_idx += 1
```

Each step runs:
- `training_step` (Python orchestrator; inner `self.loss` is JIT'd but `value_and_grad`-wrapping is unJITted)
- `apply_gradients` — many small optax ops outside JIT
- Two `state.replace()` calls (Python)
- `jax.random.split(...)[0]` returns a (2,) key and extracts — cheap but not fused

Plus `self.plot("loss", l, train=True)` inside `training_step` calls `d2l.to(value, d2l.cpu())` **every batch** (`every_n` only throttles display, not the host transfer). Every batch pays a device→host sync.

**Proposed change** (in `chapter_linear-regression/linear-regression-scratch.md`, `#@tab jax` block for `fit_epoch`):

```python
#@tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    @jax.jit
    def step_with_bn(state, batch):
        (l, mutated), grads = jax.value_and_grad(
            self.model.loss, has_aux=True)(
                state.params, batch[:-1], batch[-1], state)
        new_key, _ = jax.random.split(state.dropout_rng)
        return (state.apply_gradients(grads=grads)
                     .replace(dropout_rng=new_key,
                              batch_stats=mutated['batch_stats']),
                l)

    @jax.jit
    def step_no_bn(state, batch):
        l, grads = jax.value_and_grad(self.model.loss)(
            state.params, batch[:-1], batch[-1], state)
        new_key, _ = jax.random.split(state.dropout_rng)
        return (state.apply_gradients(grads=grads)
                     .replace(dropout_rng=new_key), l)

    step = step_with_bn if self.state.batch_stats else step_no_bn
    self.model.training = True
    for batch in self.train_dataloader:
        self.state, l = step(self.state, self.prepare_batch(batch))
        self.model.plot("loss", l, train=True)
        self.train_batch_idx += 1
    # ...validation unchanged
```

Also remove `self.plot(...)` calls from `Classifier.training_step` / `validation_step` — plot from `fit_epoch` instead (already done by `_report_train` in PT/TF versions; JAX version should mirror that).

**Concerns**:
- Two step variants (BN vs no-BN) is verbose. Alternative: always use the BN variant with a trivial `batch_stats={}` fallback. Simpler, fewer branches.
- Extending `Classifier` to not call `plot` inside `training_step` touches `chapter_linear-classification/classification.md`. Small change.
- The `@jax.jit` defined inside `fit_epoch` recompiles each epoch because it closes over `self`. Better: define once as methods and pass `self.model.loss` explicitly, or use `jax.jit(step, static_argnames=...)` at class level. Trade-off: class-level is cleaner code but a bit more plumbing.

**Estimated speedup**: 1.5–2.5× on JAX CNN notebooks. Concretely, `googlenet` 469s→~200s, `densenet` 378s→~160s, `resnet` 305s→~140s, `vision-transformer` 207s→~100s.

**Pedagogical preservation**: Keep `Classifier.loss` and `training_step` readable; only the `Trainer.fit_epoch` gets a JITted inner step. Add a comment explaining why: "We compile the gradient+state update together so JAX can fuse it."

**Decision needed from user**:
- (a) Apply as-is,
- (b) Simplify to a single `step` variant (always use BN path),
- (c) Leave as-is and focus only on the biggest outliers (ssd, image-aug, sentiment-rnn).

---

### B.2 Per-notebook plans (not using `d2l.Trainer`)

#### 3. `chapter_computer-vision/ssd.md` (2985s — 33× slower than PT)

Current (lines ~985-1027): training step is JIT'd, but the outer loop converts `features → jnp.array(features)` each iteration *inside the loop body*. Also, `train_step` returns large tuples of auxiliary values (`cls_preds`, `cls_labels`, `bbox_preds`, `bbox_labels`, `bbox_masks`) that are extracted every step for metric accumulation.

**Plan**:
1. Inside the loop, call `train_step` but only unpack `(state, loss)` — move `cls_err`/`bbox_mae` computation into the JIT'd step (return them as scalars).
2. Drop the auxiliary tensor returns from `train_step`. Only return scalar loss + metrics.
3. The data pipeline already yields NumPy arrays; `jnp.array(...)` inside the loop forces a host→device transfer every batch — unavoidable, but ensure it's not doing dtype conversion.

**Estimated speedup**: 3–5× (ssd 2985s → ~800s, still slower than PT but plausible).

**Pedagogical preservation**: The anchor/bbox logic stays unchanged. Only the metric accumulation pattern changes.

**Decision needed**: OK to proceed?

---

#### 4. `chapter_natural-language-processing-applications/sentiment-analysis-rnn.md` (1611s — 16× slower than PT)

Current training loop extracts `params_p = params['params']` each step and re-wraps. Each `state.apply_gradients` + `state.replace` happens outside JIT.

**Plan**: Wrap the whole step in `@jax.jit`, extract `params['params']` once outside the loop, re-wrap only at epoch end for prediction. Remove any `float()`/`int()` on the loss inside the step.

**Estimated speedup**: 2–3× (1611s → ~600s). Still slower than PT because BiRNN LSTM is inherently slow in Flax.

---

#### 5. `chapter_computer-vision/image-augmentation.md` (1478s — 12× slower than PT)

Current `train_ch13` iterates `tfds.as_numpy(train_iter)` and calls `train_batch_ch13(state, X, y, net, loss_fn)` without JIT. Inside `train_batch_ch13` there is `compute_loss` which is JIT-friendly but not JITted.

**Plan**:
- JIT `train_batch_ch13` with `static_argnums=(3, 4)` for `net` and `loss_fn`. Or better: hoist `loss_fn` out so the JIT only takes `state, X, y`.
- Data pipeline yields `(numpy X, numpy y)` — ensure it's in a shape ready for JIT (no per-batch dtype conversion).

**Estimated speedup**: 4–6× (1478s → ~250s).

---

#### 6. `chapter_natural-language-processing-pretraining/word2vec-pretraining.md` (1307s)

Current `train_step` is **not** decorated with `@jax.jit`. Every batch retraces the Python function.

**Plan**: Add `@jax.jit` to `train_step`. This is a ~1-line change.

**Estimated speedup**: 5–10× (1307s → ~200s). Single biggest win per line of code.

---

#### 7. `chapter_generative-adversarial-networks/dcgan.md` (1654s — 65% slower than TF)

Current: `loss_D_fn` and `loss_G_fn` are closures without JIT, and `net_G` is called twice per step (once outside, once inside the D loss).

**Plan**:
- Introduce two JIT'd step functions: `step_D(state_D, state_G, X, Z) → (state_D, loss_D)` and `step_G(state_D, state_G, Z) → (state_G, loss_G)`.
- Compute `fake_X` once inside the JIT'd function (don't pre-compute and pass as input).

**Estimated speedup**: 2–4× (1654s → ~500s).

**Pedagogical concern**: The current DCGAN code is heavily commented and deliberately shows the D/G updates sequentially. Refactoring to JIT'd steps is still readable if we keep the comments. Net change: ~+10 lines, same flow.

---

#### 8. `chapter_natural-language-processing-applications/sentiment-analysis-cnn.md` (262s — 5× slower than PT)

Does not use `d2l.Trainer`. Custom loop calls `net.apply` with `mutable=['batch_stats']` every step but doesn't JIT the value_and_grad step.

**Plan**: Wrap the step in `@jax.jit`. Same pattern as #6.

**Estimated speedup**: 3–5× (262s → ~80s).

---

### B.3 Notebooks to leave alone

These JAX notebooks are slow but either the workload is legitimate or the cost of a JAX-specific optimization would hurt clarity:

- `chapter_natural-language-processing-applications/natural-language-inference-bert.md` (413s) — BERT fine-tune, 5 epochs. PT is 315s; the 30% gap is acceptable.
- `chapter_hyperparameter-optimization/sh-intro.md` (882s) — HPO, wall-clock-bound.
- `chapter_hyperparameter-optimization/hyperopt-intro.md` (641s) — HPO.
- `chapter_hyperparameter-optimization/hyperopt-api.md` (389s) — HPO.
- `chapter_computational-performance/multiple-gpus-concise.md` (214s) — multi-GPU setup dominates.
- `chapter_computational-performance/minibatch-sgd.md` (252s) — no known waste.
- `chapter_recurrent-modern/deep-rnn.md` (245s) — similar to sentiment-rnn; fixed by B.1.
- `chapter_optimization/lr-scheduler.md` (291s) — justified workload.
- `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md` (206s) — dataset loading dominates.
- `chapter_generative-adversarial-networks/gan.md` (297s) — same pattern as dcgan; a smaller version of plan #7 applies.

---

## Part C — Overall code quality / clarity observations

Beyond performance, these patterns show up across JAX notebooks and may be worth flagging:

1. **`jax.random.PRNGKey(d2l.get_seed())` pattern**: several notebooks split/re-seed inside loops. `d2l.get_seed()` returns a fresh Python `random.randint` each call, so this is non-deterministic. For reproducibility, most notebooks should use a fixed seed (or be explicit that the demo is stochastic).

2. **`chapter_computer-vision/fine-tuning.md` JAX version uses `tf.keras` for training**. This is a pre-existing choice (not mine) that's awkward: a "JAX" notebook trains with Keras. Two options:
   - (a) Rewrite properly in Flax (substantial work — maybe 100 lines).
   - (b) Add a short caveat at the top of the JAX section explaining the choice.

3. **`d2l.get_seed`** is defined as `lambda: random.randint(0, 1e6)`. Using `1e6` (a float) as the upper bound for `random.randint` works by accident in Python (it coerces). Better: `10**6`. Cosmetic.

4. **JAX notebooks mixing `numpy as np` and `jax.numpy as jnp`**: The convention is mostly consistent but a few notebooks use `np.array(...)` where `jnp.array(...)` would be clearer. Not a bug, just style.

5. **`d2l.Trainer.plot` helper syncs GPU→CPU every batch** in PT too (not just JAX). In PT it's masked by CUDA's async copy; in JAX it's a hard sync. Could be improved for both by aggregating losses on the device and transferring at epoch end, but that's a bigger refactor.

None of C is a bug — just rough edges.

---

## Summary of what I'd actually do

If you want me to proceed, my ordered recommendation is:

1. **Quick wins** (minutes, low risk):
   - `sh-async.md` & `rs-async.md` wallclock 5→2 min (already passing; just cheaper)
   - `word2vec-pretraining.md` add `@jax.jit` (1-line win, 5-10×)
   - `sentiment-analysis-cnn.md` add `@jax.jit` (same, 3-5×)

2. **Medium effort** (tens of lines, modest risk):
   - The `d2l.Trainer` JAX `fit_epoch` JIT refactor (B.1) — helps ~15 notebooks at once.
   - `sentiment-analysis-rnn.md` JIT step
   - `image-augmentation.md` JIT `train_batch_ch13`
   - `ssd.md` metric-return cleanup

3. **Bigger effort**:
   - `dcgan.md` step function split + JIT
   - `fine-tuning.md` rewrite in Flax (if desired)

Tell me which of these to go ahead with and I'll implement.
