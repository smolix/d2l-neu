# D2L Code Guide

Scope: all code cells in Basics (ch. 1–8), Advanced (ch. 9–17), and Mathematics
for Deep Learning (ch. 24–29), and the `d2l` library they build. Prose around
code follows `writing-guide.md` (§9 there). The review that motivated this guide
is `reviews/code-review-2026-10.md`.

## 1. What Code Is For

Code in D2L does one of three things:

1. **Makes an idea concrete.** An implementation from scratch shows the
   mechanism the prose describes, line by line.
2. **Shows practice.** The concise version uses each framework the way an
   experienced engineer would, so the reader learns the idiom as well as the
   idea.
3. **Measures.** An experiment computes a number or a curve that the prose then
   interprets.

Everything else is scaffolding: plotting, logging, timing harnesses, data
plumbing, framework workarounds. Scaffolding is written once, in the library,
or not at all. It never fills a teaching cell.

A section whose point needs no framework may be plain NumPy. That is
intended, not a defect.

## 2. Cells

- **One purpose per cell.** The sentence before the cell says what it computes;
  the sentence after its output says what the output shows. Definitions and
  runs go in separate cells.
- **Short.** Most cells are under about 20 lines. A cell over 30 lines needs a
  reason, usually a complete model shown for the first time. In a good cell,
  most lines carry the idea. When fewer than half do, move the rest to the
  library or an earlier cell.
- **Shared when identical.** Code that is the same in every tab is one cell
  (untagged in Basics, `%%tab pytorch, jax` in Advanced). Framework tabs exist
  only where the frameworks differ.
- **One imports cell** per framework, at the top. No imports further down. In
  every tab `np` means NumPy; MXNet's array module gets its own name.
- **Names follow the math.** If the text writes $\mathbf{W}_q$, the code
  writes `W_q`. Comment shapes where shapes are the point
  (`# (batch, heads, steps, d_head)`).
- **No magic numbers.** `2 * 12 * 12 * 64 * n * 4` becomes
  `kv_cache_bytes(model, n)`; `0.693` becomes `math.log(2)`.
- **No hidden inputs.** A function reads its arguments, not globals defined
  cells or sections earlier. A function does one job: pass the loss or the
  update as a function instead of switching on a string flag
  (`loss_type='rp'`).
- **Explicit randomness.** Seeds are visible. In JAX, keys are split and never
  reused, and the book uses one key style (`jax.random.key`).
- **Nothing dead.** No unused imports, parameters, or return values; no
  commented-out code; no section or chapter numbers in comments or docstrings.

## 3. The Library

The `#@save` mechanism lets the book build a library as it goes. The rules
below trade off two costs: re-typing code that the reader has already seen, and
sending the reader elsewhere to look up what a call does.

### 3.1 First Sight

Code enters the library in the section where the reader sees it in full and
the prose explains it. Later sections call it from `d2l`. Two consequences:

- Never save code that the text does not explain. Display utilities in the
  Tools appendix are the one exception; the reader can skip them.
- Never re-type a saved function, and never save the same thing twice under
  two names. The review found `ResidualBlock` defined in six sections, ResNet-18
  in four, `rope` in five, `smooth` in five, and EMA in about twelve copies.

### 3.2 Mechanism in Full

In the section that teaches a mechanism, the mechanism is written out, even if
the library or the framework has it. The attention section writes the masked
softmax; later sections call the fused kernel. The optimizer sections write
each update rule; later sections call `torch.optim` and `optax`.

### 3.3 Lookup Cost

A cell should rarely call more than two or three library functions that the
reader has not met in the current chapter. When a later chapter calls a
library function, the prose says in one clause what it does
(`writing-guide.md` §7.1). Every saved function has a one-line docstring;
`build_lib.py` adds the "Defined in" reference.

### 3.4 Library Design

- **Small functions with clear contracts.** Options are keyword-only and few.
  `GPT` and `TransformerBlock` with nine to eleven positional flags are the
  pattern to avoid; pass components (`norm=nn.RMSNorm`, `attn_factory=...`)
  instead.
- **Return values, not side effects.** Training returns a history; evaluation
  returns a dict. No code reads results back from a plotting object
  (`model.board.data['val_loss'][-1].y` appears about 15 times today).
- **One contract for all tabs.** A saved function has the same signature and
  the same return type in every framework. Today the attention modules return a
  tensor in PyTorch and a tuple in JAX, and `update_D` takes a loss object in
  PyTorch but hardcodes it in JAX.
- **One home per symbol,** in the section that teaches it.
- **Tags.** Helpers for Advanced chapters are tagged `%%tab pytorch, jax`. An
  untagged `#@save` in an Advanced chapter leaks into `d2l/tensorflow.py` and
  `d2l/mxnet.py`.

## 4. Shared Infrastructure

These pieces are needed across the book. Each is taught once, where noted, and
reused everywhere else. `reviews/code-review-2026-10.md` lists the current
copies that each one replaces.

| Piece | Taught in | Replaces |
|---|---|---|
| `Trainer` whose steps return values; `Trainer.history`, `Trainer.evaluate`, a scheduler from `configure_optimizers`, lazy initialization | ch. 2 `oo-design`, extended in ch. 3 and ch. 7 | board scraping; two `RecipeTrainer`s; five validation loops; dead `init_cnn` calls |
| `synchronize`, `benchmark(f, *args)`, `peak_memory(f, *args)` | ch. 5, asynchronous execution in `gpus-devices-memory` | about ten timers with conflicting semantics in ch. 10–13 |
| `num_params`, `param_bytes` | ch. 5 `parameters-state-memory` | about twenty inline counts |
| `EMA` (PyTorch `AveragedModel`, JAX `optax.ema`) | ch. 7 `training-recipes` | three designs, about twelve copies |
| `lm_perplexity` | ch. 8 `rnn-scratch` | five `val_ppl`s that disagree across tabs |
| Shared test models (`TinyLM`, `SmallCNN`) with identical initialization in both tabs, plus a step-based training loop `train_steps(model, data, optimizer, num_steps, loss_fn)` | ch. 9, a section of its own before Adam | about eleven training loops and eleven evaluators per framework in ch. 9–17 |
| One 4-D multi-head attention, `causal_mask`, `padding_mask`, `rope` | ch. 10 | two attention APIs, five `rope`s, five decoder variants |

The Advanced chapters use the `Trainer` where epochs and a `DataModule` fit
the experiment, and `train_steps` where training is counted in steps. They do
not write their own loops.

## 5. Frameworks

### 5.1 Agreement and Difference

Across tabs, the algorithm, the hyperparameters, the initialization, and the
conclusion agree. The structure of the code may differ wherever a framework
has a clearer idiom, and such differences are worth showing. Good examples
already in the book:

- Hessian–vector products: forward-over-reverse in one line in JAX and
  `torch.func`, nested tapes in TensorFlow (`mdl-matrix-calculus-autodiff.md:992`).
- A doubling scan in PyTorch against `lax.associative_scan` in JAX
  (`ssm.md:222`).
- Muon as a `torch.optim.Optimizer` subclass and as an
  `optax.GradientTransformation` (`muon.md:314`, `:333`).
- A straight-through estimator as `autograd.Function` and as `custom_vjp`
  (`custom-layers.md:864`, `:877`).
- A preallocated KV cache with `dynamic_update_slice` under `jit`, against a
  growing `torch.cat` (`kv-cache.md:219`).

Differences that change results are bugs. Examples from the review: `TinyLM`
initialized differently in the two tabs (so each tab needs its own learning
rate), learning-rate schedules per epoch in one tab and per step in the other,
max pooling padded with zeros in JAX and TensorFlow but with $-\infty$ in
PyTorch.

### 5.2 PyTorch

Use `nn.Parameter` for every trainable tensor, from the first scratch model
on. Other current idioms:

- `torch.optim` and `torch.optim.lr_scheduler` (`LambdaLR`, `SequentialLR`);
- `torch.optim.swa_utils.AveragedModel` for EMA;
- `torch.func` (`grad`, `vmap`, `jvp`, `vjp`, `hessian`) for functional
  derivatives;
- `F.scaled_dot_product_attention`;
- `torch.inference_mode` for evaluation and sampling;
- `clip_grad_norm_`, once the scratch version has been taught;
- `torch.distributions.Categorical`;
- `torch.linalg.vector_norm` rather than the deprecated `torch.norm`.

### 5.3 JAX

- Pure functions; `nnx.jit` at module level, never inside a function that is
  called repeatedly (today this retraces on every call in `train_lm`, `gpt`,
  `kv-cache`, `moe`, `deltanet`, `hybrids`).
- `lax.scan` and `lax.fori_loop` for loops over time steps and sampler steps;
  `lax.associative_scan` for parallel scans; `vmap` for batches of anything.
- `optax` for optimizers, schedules (pass the schedule as `learning_rate`),
  clipping (`optax.chain`), EMA (`optax.ema`), and Polyak averaging
  (`optax.incremental_update`).
- `nnx.Sequential` for plain stacks; `rngs` as a required argument; NNX
  filters for freezing; `nnx.remat` for checkpointing; `jax.debug_nans` for
  NaN hunting; `block_until_ready` before stopping a timer.
- No TensorFlow dependency for loading in-memory arrays.

### 5.4 TensorFlow and MXNet (Basics and Appendix)

TensorFlow: Keras layers, `add_weight` for scratch parameters, `tf.function`
for anything that is timed, the `*_with_logits` losses. MXNet: Gluon,
`hybridize`, one way to collect parameters, GPU work on the main thread.

## 6. Performance Habits

The code teaches habits as much as algorithms. These habits are always kept:

- No host synchronization in a hot loop: `.item()`, `float()`, `np.asarray`,
  or `if norm > c` on a device value. Accumulate on the device and convert
  once.
- Vectorize loops over elements, seeds, or time steps unless the loop is the
  lesson.
- Time with a warm-up, device barriers, and `time.perf_counter`
  (`d2l.benchmark`).
- No recompilation inside a loop.
- Numerically stable forms: `logsumexp`, `logaddexp`, `log_softmax`, and a
  clip before `arccos`.

## 7. What Code Never Contains

- Drawing code for illustrative figures (`CLAUDE.md`), or raw matplotlib in a
  teaching cell. Use `d2l.plot`, or move the figure to a generator.
- Framework workarounds and environment settings in the main flow (for
  example, setting `XLA_PYTHON_CLIENT_MEM_FRACTION` in an imports cell).
- Draft history, hardcoded section numbers, or narration of what one
  framework's stored run did (`writing-guide.md` §4.5).

## 8. Checklist

For each cell:

1. Does it compute, demonstrate, or measure something the prose discusses?
2. Is its purpose stated before it and its output read after it?
3. Is it short, and do most of its lines carry the idea?
4. Is every library call one the reader has met, or explained in a clause?
5. Has this code appeared before? If so, call the saved version.
6. Do the tabs agree on algorithm, hyperparameters, initialization, and result?
7. Does each tab use its framework's idiom?
8. Are there host syncs, Python loops over data, or recompilation in a hot path?
9. Are names, shapes, and constants readable without the surrounding cells?
