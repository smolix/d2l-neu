# Review — chapter_linear-classification/classification.md  (§4.3 "The Base Classification Model")

**Role in the chapter:** Pure infrastructure file. Defines the `Classifier` base class (subclasses `d2l.Module`), the `accuracy` metric, and a default `configure_optimizers` hook (SGD), so every downstream classification model in the book can inherit them rather than re-implementing them.

**Verdict:** The file does its narrow job correctly. The accuracy code is logically sound across all four frameworks, and the placement of `configure_optimizers` here makes the subsequent scratch/concise sections clean. The chief weaknesses are: (1) the *idea* behind this section — why accuracy and loss are both needed, why a base class exists, and what the `configure_optimizers` hook does — is explained so sparsely that a reader coming to this cold gets only boilerplate, not insight; (2) the introductory paragraph and Summary are perfunctory and should be combined into one tight motivating paragraph + a richer summary; (3) the MXNet-specific `get_scratch_params` / `parameters` helper block is pure plumbing and not explained; (4) no figure despite a genuine concept worth drawing (score → loss → accuracy pipeline); (5) exercises 1–2 address only the validation-averaging bias (minor), while exercise 3 is the right caliber but is the only substantive problem.

**Grade:** B− — correct and usable, but a reader assigning it at a top program would want the conceptual core (loss ≠ accuracy; why we train with one and report the other; what `configure_optimizers` does) explained rather than implied.

---

**Top priorities (ranked):**
1. [P1] CLS-1 — Add a motivating "Scores, Loss, and Accuracy" paragraph that explains *why* the base class exists and the conceptual distinction between the differentiable loss and the discrete accuracy metric.
2. [P1] CLS-2 — Rewrite the intro paragraph and Summary into a single coherent narrative arc.
3. [P1] CLS-3 — Explain the `configure_optimizers` hook: one sentence saying what it does, what it returns, and how `Trainer` calls it — this is the first time the book installs a default SGD, and it is currently presented without explanation.
4. [P1] CLS-4 — Add one pre-generated schematic figure illustrating the forward-pass → logits → softmax/argmax → loss vs. accuracy pipeline.
5. [P2] CLS-5 — Add an exercise on calibration: a model can have high accuracy but be badly miscalibrated; ask the reader to construct a two-class example.
6. [P2] CLS-6 — MXNet retirement note: add a one-line `:begin_tab:` note that MXNet is archived (2023) so readers know its tab is legacy.
7. [P2] CLS-7 — Explain the MXNet `get_scratch_params` block (currently zero prose).

---

## 1. Coverage

### Add

**Why the base class exists (currently implicit).** The opening sentence says "it is worth adding functionalities to support this setting specifically" without saying *what* problem it solves. A top-program reader deserves one sentence: the `Module` base class's `validation_step` only plots loss; by subclassing to `Classifier` we add accuracy tracking without duplicating it in every model. This is a legitimate OO-design point and belongs here.

**The loss–accuracy distinction.** The Accuracy section (lines 134–148) describes *how* accuracy is computed but never explains *why* we need it alongside the loss. The conceptual point — that cross-entropy is a smooth surrogate optimized by gradient descent, while accuracy is the discrete metric benchmarks and practitioners actually care about, but its gradient is zero almost everywhere — deserves two sentences. The slides already contain this ("Why report both loss and accuracy?"), but it is absent from the text. This belongs as a short paragraph before the code.

**The `configure_optimizers` hook.** Lines 100–130 silently patch a default SGD onto `d2l.Module`. There is no prose explaining what `configure_optimizers` returns, how `Trainer` calls it (it is called once at the start of training, not per-step), or why this is the right place to install a default (so individual subclasses don't have to). This hook is used by every downstream model in the book; one explanatory sentence here would pay dividends.

### Remove / trim

**MXNet `get_scratch_params` prose vs. code imbalance.** Lines 196–225: The `:begin_tab:mxnet` note explains *that* Gluon misses bare `np.ndarray` attributes but the block itself (two methods, ~15 lines) gets no inline comments and no prose saying when a reader would hit this. Either trim the explanation to one sentence and add inline comments, or move the block to a sidebar. It is a framework-specific workaround, not a conceptual contribution.

**Intro paragraph (lines 6–9).** "You may have noticed that the implementations from scratch and the concise implementation were quite similar in the case of regression." This is filler that names neither what changes nor why. It should be replaced with a one-sentence statement of the section's purpose (see CLS-2).

### Reorder / restructure

The current flat two-section structure (`## The Classifier Class` → `## Accuracy`) is acceptable given the file's brevity. The only structural suggestion: move the conceptual explanation of loss-vs-accuracy to *before* the code blocks in the Accuracy section (currently it appears after). The idea should precede the mechanism.

---

## 2. Teaching quality

### Structure & flow

Two `##` sections plus Summary + Exercises. For a file this short (≈230 lines of source), the structure is reasonable. The Summary (lines 228–231) is two sentences and adds nothing the reader did not already know. It should either be cut and folded into the intro, or expanded to connect forward to how `Classifier` is used in every subsequent file.

### Figures

**No figures at all.** This is the most impactful gap. The conceptual pipeline

```
input x  →  [model]  →  logits o  →  softmax → probabilities p̂
                                              ↘  argmax → ŷ → accuracy (discrete)
                  cross-entropy loss ←  p̂, y  (differentiable, used for training)
```

is the central organizing idea of the entire chapter, and it is never drawn. A schematic figure here would serve readers for the entire rest of the chapter. Per book conventions it must be pre-generated (committed SVG generator), not inline matplotlib.

### Prose & clarity

- **Lines 6–9** (intro): weak, should be replaced (see CLS-2).
- **Lines 134–148** (Accuracy intro): good concrete motivating examples (Gmail), but the transition "Accuracy is computed as follows" skips the conceptual point about *why* we care about accuracy in addition to loss.
- **Lines 151–157** (accuracy description): clear step-by-step walk of the code. Keep as is.
- **Lines 228–231** (Summary): two sentences that could be cut without loss. CLS-2 proposes a replacement.

### Exercises

Exercise 1 (lines 235–236): asks to derive an exact formula for the validation loss from the averaged estimate. Fine but mechanical — worth keeping because it sharpens thinking about batch averaging.

Exercise 2 (lines 236–237): prove the quick-and-dirty estimate is unbiased. Correct and good, though the answer is trivial (linearity of expectation).

Exercise 3 (lines 237–238): optimal classifier under a general loss matrix. This is the right caliber for a top course — it asks for the Bayes-optimal decision rule. Keep. (The answer: minimize expected loss $\sum_{y} l(y, y') p(y|x)$ over $y'$.)

**Missing:** A calibration exercise (CLS-5). Also missing: an exercise asking the student to implement a top-$k$ accuracy (a natural extension of argmax → top-1).

---

## 3. Code & examples

### Does the code teach?

The code blocks are short and do teach: `validation_step` shows what gets logged, and the accuracy implementation walks through the argmax-compare-mean pipeline in four lines. **No cell draws an illustrative figure** (good). The MXNet `get_scratch_params` block is the one cell that is pure plumbing with no pedagogical value in the context of this section; it belongs in an appendix or behind a fold.

### PyTorch

```python
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

Correct and idiomatic for PyTorch 2.x. No issues.

```python
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

Correct. The `Y.dtype` cast avoids a silent comparison mismatch when `Y` is `int64` and `argmax` returns `int64`; explicit. The cast to `float32` before `mean` is correct. One mild concern: `d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))` silently handles the case where `Y_hat` is a flat vector (1D), but the code does not explain this — a comment would help.

### JAX

The JAX `Classifier` overrides `training_step` to handle `has_aux=True` (needed for BatchNorm auxiliary state). This is documented in the `:begin_tab:jax` prose, which is adequate. The `accuracy` method re-runs the forward pass internally (via `state.apply_fn`) rather than taking a precomputed `Y_hat`. This is correct and necessary given Flax's stateless convention, and it is explained.

Minor: the `@partial(jax.jit, static_argnums=(0, 5))` decorator marks `self` and `averaged` as static. This is correct (both are compile-time constants in practice), but a one-line comment saying why `self` must be static here would help readers who are new to JAX JIT tracing.

### TensorFlow

```python
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(float(self.lr))
```

Correct. `float(self.lr)` defensive cast is appropriate. The `_report_val` method on `Classifier` (lines 73–75) is TF-specific and documented. No issues.

**TF accuracy:** uses `d2l.argmax = tf.argmax` with `axis=1` explicitly passed — correct (TF's `argmax` signature requires an explicit `axis`, unlike NumPy, so this is safe).

### MXNet

MXNet was **archived by the Apache Software Foundation in 2023**. It is still present in this book and the wheel still builds (the recent custom 2.0+cu13 wheel), but it should carry a note that it is a legacy/archived framework. The `configure_optimizers` MXNet variant correctly dispatches between Gluon Trainer and the scratch SGD fallback. The `get_scratch_params` / `parameters` methods are a workaround for the fact that Gluon's `collect_params` does not find bare `np.ndarray` attributes. This is described in the `:begin_tab:mxnet` note but the code itself has no comments, making it opaque.

### Cross-framework consistency & d2l conventions

- **One imports cell per framework** at the top: ✓ (cell `classification-the-base-classification-model`).
- **Stable cell IDs:** ✓ all four frameworks share a single stable ID per logical block.
- **`#@save` hygiene:** ✓ on `Classifier`, both accuracy variants, both `configure_optimizers` variants, and both MXNet helpers.
- **No re-imports later in the file:** ✓.
- **Cross-framework divergence:** Unavoidable and minimal. The JAX variant is meaningfully different (has-aux, stateless forward), which is correctly explained. The MXNet variant has an extra cell (`classification-accuracy-2`) absent from the other frameworks; this is correct because only MXNet needs the scratch-params workaround.
- **Correctness across frameworks:** The accuracy logic — `argmax(axis=1) → cast to Y.dtype → == → cast to float32 → mean` — is identical in PyTorch, JAX, TF, and MXNet (via their respective shim aliases). Confirmed by inspection of `d2l/{torch,jax,tensorflow,mxnet}.py`. No bugs found.

---

## 4. Implementation spec

### CLS-1 — Add "Loss vs. Accuracy" motivating paragraph  ·  P1 · S · authored

- **Type:** teaching / coverage
- **Where:** `chapter_linear-classification/classification.md` — immediately before the `## Accuracy` heading (after the `configure_optimizers` code blocks, before line 133 `## Accuracy`).
- **Change:** Insert the following paragraph between the last `configure_optimizers` code block and the `## Accuracy` heading:

```
Why do we need accuracy if we already have a loss?
The cross-entropy loss is a smooth function that gradient descent can minimize —
it rewards the model for assigning higher probability to the correct class,
even when the argmax decision is already correct.
Accuracy, by contrast, counts only whether the argmax matches the label:
it is the metric practitioners and benchmarks ultimately care about,
but its gradient is zero almost everywhere and useless for training.
Reporting both during validation gives a complete picture:
loss tracks optimization progress and calibration;
accuracy tracks deployment-quality decision performance.
```

- **Touches:** none.
- **Done when:** the paragraph appears in the rendered HTML between the last code block of §4.3.1 and the `## Accuracy` (§4.3.2) heading; `make html` clean.
- **Depends on:** none.

---

### CLS-2 — Replace intro paragraph and expand Summary  ·  P1 · S · authored

- **Type:** prose / teaching
- **Where:** `chapter_linear-classification/classification.md`
  - Verbatim anchor for intro: `"You may have noticed that the implementations from scratch and the concise implementation were quite similar in the case of regression."`  (lines 6–9)
  - Verbatim anchor for Summary: `"Classification is a sufficiently common problem that it warrants its own convenience functions."` (line 230)

- **Change (intro replacement):**

Replace lines 6–9 with:

```
Every classification model in this book — from the linear softmax regressor we
build next to deep convolutional networks chapters later — shares two common
needs: a **validation step** that reports both loss and accuracy, and a default
optimizer. Rather than re-implementing these in every subclass, we collect them
here in a `Classifier` base class that extends `d2l.Module`
(introduced in :numref:`sec_oo_design`).
```

- **Change (Summary replacement):**

Replace the Summary section (lines 228–231) with:

```
## Summary

The `Classifier` class adds two things to `d2l.Module`: an overridden
`validation_step` that logs *both* loss and accuracy, and a default
`configure_optimizers` that returns a minibatch SGD optimizer.
All classification models in the rest of the book subclass `Classifier`
and need only supply `forward` and `loss` — the training and
evaluation loop comes for free.
Accuracy is defined as the fraction of examples whose predicted class
(the argmax of the score vector) matches the true label; it is a discrete
metric and cannot be used as a training objective, but it is almost always
the number reported in benchmarks and the one the reader should watch.
```

- **Touches:** none.
- **Done when:** rendered §4.3 opens with the new intro; the Summary section reads as the replacement text; `make html` clean.
- **Depends on:** none.

---

### CLS-3 — Explain the `configure_optimizers` hook in prose  ·  P1 · S · authored

- **Type:** teaching / prose
- **Where:** `chapter_linear-classification/classification.md` — verbatim anchor: `"By default we use a stochastic gradient descent optimizer, operating on minibatches, just as we did in the context of linear regression."` (line 100)
- **Change:** Replace that single sentence with:

```
By default we use a stochastic gradient descent optimizer operating on
minibatches, just as we did in the context of linear regression.
`configure_optimizers` is a hook called once by `Trainer` at the start of
training (see :numref:`sec_oo_design`); it returns an optimizer object that
`Trainer` uses to update parameters after each backward pass.
Subclasses may override this method to switch to a different optimizer —
later chapters do exactly that — but SGD is the right default for the
models in this chapter.
```

- **Touches:** none.
- **Done when:** the three-sentence explanation appears in HTML immediately before the `configure_optimizers` code tabs; `make html` clean.
- **Depends on:** none.

---

### CLS-4 — Add pre-generated schematic figure: scores → loss vs. accuracy pipeline  ·  P1 · L · authored

- **Type:** figure
- **Where:** `chapter_linear-classification/classification.md` — insert immediately after the new motivating paragraph added by CLS-1 (i.e., just before the `## Accuracy` heading).
- **Change:** Add a committed SVG figure via the `mdl-figure` skill / `tools/gen_mdl_classification_figures.py`. The figure should depict:

  - Left column: input $\mathbf{x}$.
  - Centre box: model → score vector $\mathbf{o} \in \mathbb{R}^q$ (logits).
  - Two branches from $\mathbf{o}$:
    - **Top branch (training):** softmax → probability vector $\hat{\mathbf{y}}$ → cross-entropy loss $\ell$ (arrow labelled "differentiable; used for gradient descent").
    - **Bottom branch (evaluation):** argmax → predicted class $\hat{y}$ → compare with true label $y$ → accuracy (arrow labelled "discrete; used for benchmarks").
  - The style should follow `tools/gen_mdl_figures.py` house conventions (neutral palette, clean sans-serif labels, no gradients).

  The figure inclusion in the `.md`:

```markdown
![From model scores to training loss and evaluation accuracy. The two branches share the same logits but serve different purposes: the differentiable cross-entropy loss drives gradient descent; the discrete argmax decision is what accuracy counts.](../img/cls-scores-loss-accuracy.svg)
:label:`fig_cls_scores_loss_accuracy`
```

- **Touches:** `tools/gen_mdl_classification_figures.py` (create or extend), `img/cls-scores-loss-accuracy.svg` (generate via `make figures`), commit SVG.
- **Done when:** figure renders in HTML with caption and `:numref:` label resolving; SVG committed; `make html` clean.
- **Depends on:** CLS-1 (placement).

---

### CLS-5 — Add calibration exercise  ·  P2 · S · authored

- **Type:** teaching / exercises
- **Where:** `chapter_linear-classification/classification.md` — verbatim anchor: `"1. Given a multiclass classification loss, denoting by..."` (exercise 3, line 237). Insert as exercise 4 after exercise 3.
- **Change:** Add:

```
4. Suppose two classifiers $A$ and $B$ both achieve 90% accuracy on a
   ten-class test set, but $A$ assigns probability 0.91 to the correct
   class on average while $B$ assigns only 0.51.
   (a) Compute the expected cross-entropy loss of each.
   (b) Which classifier would you prefer in a safety-critical application?
   Why does accuracy alone fail to distinguish them?
   (c) Can you construct a simple post-hoc recalibration (a monotone
   rescaling of the scores) that improves $B$'s calibration without
   changing its accuracy?
```

- **Touches:** none.
- **Done when:** exercise 4 appears in rendered HTML exercises list; `make html` clean.
- **Depends on:** none.

---

### CLS-6 — MXNet archived-framework note  ·  P2 · S · mechanical

- **Type:** currency
- **Where:** `chapter_linear-classification/classification.md` — verbatim anchor: `:begin_tab:\`mxnet\`` block immediately before the MXNet `Classifier` code cell (`classification-the-classifier-class-1`). Currently there is no `:begin_tab:mxnet` prose block before this cell (the only `:begin_tab:` before the first code cell is the joint `pytorch, mxnet, tensorflow` block). Add a standalone MXNet note.

  Insert immediately after the `:begin_tab:\`jax\`` ... `:end_tab:` block (after line 62) and before the code fence at line 64:

```
:begin_tab:`mxnet`
**Note:** Apache MXNet was archived by the Apache Software Foundation in
2023 and is no longer actively maintained upstream. The MXNet tab is
preserved here for readers with existing MXNet codebases; new projects
should use PyTorch or JAX.
:end_tab:
```

- **Touches:** none.
- **Done when:** the MXNet tab for this section shows the archived note in rendered HTML; `make html` clean.
- **Depends on:** none.

---

### CLS-7 — Add inline comments to MXNet `get_scratch_params` block  ·  P2 · S · authored

- **Type:** teaching / code
- **Where:** `chapter_linear-classification/classification.md` — verbatim anchor: `def get_scratch_params(self):` (cell `classification-accuracy-2`, line 211).
- **Change:** Replace the existing code block with an annotated version:

```python
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    # Gluon's collect_params() only finds Parameters declared via Gluon's
    # Parameter API. For from-scratch models that store weights as bare
    # np.ndarrays, we walk the object's attributes recursively instead.
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    # Return Gluon ParameterDict when the model uses Gluon layers;
    # fall back to the bare-array scan for from-scratch implementations.
    params = self.collect_params()
    return params if isinstance(params, dict) and len(
        params.keys()) else self.get_scratch_params()
```

- **Touches:** none (comment-only change, no logic change).
- **Done when:** the MXNet `classification-accuracy-2` cell shows the inline comments in rendered HTML; cell outputs unchanged (definition only); `make html` clean.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **Accuracy code correctness.** The four-line accuracy implementation (`reshape → argmax(axis=1) → cast to Y.dtype → == → cast float32 → mean`) is correct across all four frameworks and handles the dtype-mismatch pitfall explicitly. The prose walk-through of the steps (lines 150–157) is clear.
- **JAX explanation.** The `:begin_tab:jax` notes for both the `Classifier` class and the `accuracy` method correctly explain the `has_aux` pattern and the `@jax.jit / static_argnums` decorator. These are genuinely tricky JAX idioms and the explanations are accurate.
- **Separation of validation step from training step.** The design choice to override only `validation_step` in `Classifier` (not `training_step`) is architecturally clean and implicitly teaches that validation and training have different concerns. Worth preserving.
- **`averaged` flag on `accuracy`.** Returning the per-example comparison tensor when `averaged=False` is a useful extension point (used by batch-norm-aware models) and costs nothing. Keep.
- **Exercises 1–3.** All three are worth keeping. Exercise 3 (Bayes-optimal decision under a general loss matrix) is particularly strong for a top-course assignment.
- **Slide content.** The "Why report both loss and accuracy?" slide (lines 314–325) contains the conceptual explanation that the main text lacks — CLS-1 essentially promotes this into the textbook prose where it belongs.
