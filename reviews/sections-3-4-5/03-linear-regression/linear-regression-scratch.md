# Review — chapter_linear-regression/linear-regression-scratch.md  (§3.4 "Linear Regression Implementation from Scratch")

**Role in the chapter:** The pedagogical heart of the chapter — it implements
linear regression end-to-end with nothing but tensors and autodiff (model, loss,
hand-rolled minibatch SGD, training loop) so the reader sees exactly what the
framework `nn.LazyLinear`/`MSELoss`/`SGD` of the concise sibling
(`linear-regression-concise.md`) hides. The first place in the book the reader
runs a full training loop and watches a loss curve fall.

**Verdict:** Strong, correct, and demonstrably converging in all four frameworks
(every committed `error in estimating w` is within ±5e-4, b within 1e-3, no stray
warnings). It cleanly avoids duplicating SGD *theory* (owned by
`linear-regression.md` §3.1, which already derives the minibatch update
`eq_linreg_batch_update` and the non-determinism point) and the concise sibling.
Its weaknesses are pedagogical, not factual: (1) **no seeding**, so the "we
recover the ground truth" payoff is non-reproducible for torch/tf/mxnet (only JAX
seeds), undercutting the file's whole reason for existing; (2) the manual SGD
step — the single most important thing this file teaches — is **split across the
`SGD.step` cell and the `fit_epoch` glue** with the crucial `torch.no_grad()` /
grad-zeroing explained in code comments rather than prose, so a reader never sees
"zero → forward → backward → step" as one annotated unit; (3) no "Discussion" that
connects forward (to `weight-decay`, optimization, the concise version) the way
the gold-standard `calculus.md` does. The single highest-value change is **adding
a short, framework-annotated walk-through of one SGD update + a fixed seed** so the
demystification actually lands and reproduces.

**Grade:** B. Assignable today and largely excellent, but the from-scratch SGD
step (its core teaching object) is under-narrated and the headline result is not
reproducible — both are squarely in scope and fixable.

**Top priorities (ranked):**
1. [P0] `LRS-1` — Seed the run (torch/tf/mxnet) so "we recover w,b" reproduces; tie the printed errors to the noise level.
2. [P1] `LRS-2` — Add a prose "anatomy of one SGD step" tying together grad-zeroing, `no_grad`, and the in-place update (the file's core lesson).
3. [P1] `LRS-3` — Reference the loss curve with a `:numref:` and one interpreting sentence (it currently floats, unlabelled, uninterpreted).
4. [P1] `LRS-4` — Add a short "Discussion" connecting forward (concise / weight-decay / optimization) and forward-point the three-way-split aside to `generalization.md`.
5. [P1] `LRS-5` — MXNet currency: add the one-line archived-framework note (book-wide decision) and de-emphasize the MXNet `autograd.record()` variant.
6. [P2] `LRS-6` — Fix the abs-loss exercise's `.sum()` vs averaged-loss learning-rate inconsistency; sharpen two exercises.
7. [P2] `LRS-7` — Trim the long TF/JAX `:begin_tab:` compile essays (they pre-empt `oo-design`/optimization and bury the from-scratch idea).

---

## 1. Coverage

### Add

- **A fixed seed and a sentence relating the residual error to the noise.** This
  is the most important *content* gap. The file's thesis (lines 18–25, 618–623)
  is "because we made the data we can check we recovered `w`,`b`." But for
  pytorch/tensorflow/mxnet nothing is seeded (confirmed: no `manual_seed` /
  `set_seed` anywhere in this file, `oo-design.md`, or the torch/tf/mxnet tabs of
  `synthetic-regression-data.md`; only JAX threads `PRNGKey(0)`). So the printed
  errors change every run and "indeed they turn out to be very close" (line 623)
  is a claim the reader cannot reproduce. The Preliminaries chapter already
  *teaches* seeding (`ndarray.md` line 353–355, `probability.md` line 274–276),
  so this file should *use* it. Pair the seed with one sentence connecting the
  ~3e-4 residual to the σ=0.01 label noise and the finite-sample/SGD-noise floor —
  this is the honest version of the payoff and it forward-connects to
  `generalization.md`. See `LRS-1`.

- **A prose "anatomy of one SGD update."** The single idea the whole file exists
  to demystify is the manual gradient step, yet it is never shown as one narrated
  unit. The `SGD.step` cell (lines 263–277) shows `param -= lr * param.grad` and
  `zero_grad`; the *ordering* (zero → forward → backward → step) and the reason
  for `torch.no_grad()` live only in the `fit_epoch` cell's inline comments (lines
  461–463) — which a reader skims as plumbing. A best-in-class treatment (CS231n's
  "SGD in ~10 lines", Goodfellow §8) states the loop in prose right next to the
  code. The autograd mechanics themselves (`no_grad`, `detach`, grad accumulation)
  are already taught in `autograd.md`, so here you only need 4–5 sentences that
  *apply* them to the update. See `LRS-2`.

- **A short "Discussion"/forward-pointing close.** The gold-standard
  `calculus.md` ends with a "## Discussion" that connects outward; this file ends
  with a competent but inward-looking "Summary." Add (or rename to) a Discussion
  that points forward to: the concise version (`sec_linear_concise`), weight decay
  / regularization (`sec_weight_decay`), and the Optimization part for SGD
  variants/learning-rate schedules (so the curious reader knows where momentum/Adam
  live — *one line*, per the scope map; do **not** import them). See `LRS-4`.

- **Forward-point the three-way-split aside.** Lines 601–609 raise train/val/test
  splitting and hyperparameter selection then say "We elide these details for now
  but will revise them later" with no pointer. `generalization.md` (the sibling,
  §3.6 — "Model Selection", "Cross-Validation", validation/test discussion at its
  lines 366–432) owns exactly this. Replace the dangling "later" with
  `:numref:`sec_generalization_basics``. See `LRS-4`.

### Remove / trim

- **The TensorFlow and JAX `:begin_tab:` compile essays are too long and
  off-topic here.** The TF block (lines 391–419, ~29 lines) and JAX block (lines
  421–442, ~22 lines) explain `tf.function`/`jax.jit` graph compilation, tracing,
  `_compile_steps`, and async dispatch in depth. This is real and correct, but (a)
  it pre-empts material that `oo-design.md` already previews (its `ProgressBoard`
  discussion, lines 156–175, covers "keep the hot path pure and compiled") and
  that the Optimization/performance chapters own, and (b) at "the tiny scale of
  this example the wall-clock difference is modest" (the text *admits* this, line
  417), so a 29-line digression about performance buries the from-scratch SGD idea
  the section is about. Trim each to ~4–6 sentences: *what* the wrapper does and
  *why it's needed at trace time*, then forward-point. See `LRS-7`.

- **Nothing else to cut.** The file is already lean; the model/loss/optimizer/loop
  decomposition is the right content at the right depth. Resist adding optimizer
  theory, init theory, or a closed-form-vs-SGD benchmark here — all are owned
  upstream (`linear-regression.md` §3.1) or downstream (Optimization).

### Reorder / restructure

- The four-in-a-row `## Defining the {Model, Loss Function, Optimization
  Algorithm}` + `## Training` headings are the natural decomposition and match the
  intro's enumerated list (lines 14–17) — keep them. The only structural lift is
  the closing **Discussion** (`LRS-4`) and, optionally, promoting the loss-curve
  interpretation into the Training section narrative (`LRS-3`). No content belongs
  in a different file.

---

## 2. Teaching quality

### Structure & flow

The spine (model → loss → optimizer → training loop → run → check) is exactly
right and mirrors the intro promise. Six `##` sections, no nesting; that's
acceptable for an implementation walk-through (the gold-standard files nest more,
but they are concept chapters). The one real flow defect: the **payoff arc is
weak**. The reader builds four components, runs `trainer.fit`, sees a curve
(unreferenced), then reads "Indeed they turn out to be very close" — but the
"close to *what*, and why not closer?" beat is missing, and there's no forward
connection. `LRS-2`/`LRS-3`/`LRS-4` close this.

### Figures

- **Inventory:** exactly one visual — the training **loss curve** emitted by the
  `training-3` cell (`outputs/<fw>/.../linear-regression-scratch-training-3-1.svg`,
  confirmed present and ~19–20 KB in all four manifests; rendered page shows
  train+val loss vs epoch). This is a *computed data plot* (`d2l.plot` via
  `ProgressBoard`), so it is correctly a notebook output, **not** an illustrative
  figure — it does not violate the "no figure-drawing code in notebooks" rule.
- **Defect:** it has no `:label:`/`:numref:` and the prose never points to it or
  interprets it. A reader is shown a converging curve and told nothing about it.
  Add a `:numref:` reference + one interpreting sentence ("training and validation
  loss fall together toward the σ²/2 ≈ 5e-5 noise floor — no gap, because a 2-D
  linear model on 1000 examples cannot overfit"). See `LRS-3`. (Note: with the
  `ProgressBoard`/`d2l.plot` pipeline an inline-output plot may not accept a
  `:label:` directly the way a static `img/` SVG does — the implementer should
  reference it in prose as "the loss curve produced above" if a numbered label
  cannot attach; either way the interpreting sentence is the substance.)
- **Missing figure?** None required. The concepts here are procedural, and the
  upstream `linear-regression.md` already carries the geometric/fit figures
  (`fig_fit_linreg`, etc.). Do **not** add schematic figures to this file — it
  would dilute the "code teaches" purpose.

### Prose & clarity

- **Generally clear and well-paced.** The intro (9–33) is a model "why from
  scratch" motivation. The init paragraph (69–79) and the SGD motivation (197–220)
  are good.
- **Worst offender — buried core mechanic.** The reason for `torch.no_grad()` and
  grad zeroing is delivered as a code comment (lines 461–463) inside the densest
  cell. Promote to prose (`LRS-2`).
- **Dangling promise (601–609):** "We elide these details for now but will revise
  them later" — replace "later" with a `:numref:` (`LRS-4`).
- **`save_hyperparameters()` magic-number aside (74–76):** "The magic number 0.01
  often works well in practice" is fine but could forward-point to init discussion
  (He/Xavier live in MLP/Builders' chapters) in *one* clause — optional, P2.

### Exercises

The eight exercises are genuinely good and above the bar — the physics modeling
problems (Ohm's law Ex. 2, Planck's law Ex. 3) are exactly the "thought-provoking
extension" a top course wants, and the robust-loss progression (Ex. 7a–c, building
to "combine squared + absolute → Huber") and the malicious-shuffle question
(Ex. 8) build well. Two fixes:

- **Ex. 7 inconsistency (line 703):** the suggested absolute-value loss uses
  `.abs().sum()`, but the file's taught loss uses `reduce_mean`. A summed loss has
  a gradient ~`batch_size`× larger, so with the same `lr=0.03` it will diverge —
  an unintended trap that muddies the *intended* lesson (robustness to the
  `y_5=10000` outlier). Change to `.abs().mean()` (or add a parenthetical "note
  you must lower the learning rate if you sum instead of average — cf.
  :numref:`eq_mse` discussion"). See `LRS-6`.
- **Ex. 1 (zero-init, var-1000):** strong, but add an explicit sub-prompt "does
  the *symmetry* argument that breaks zero-init for deep nets apply to this linear
  model? Why or why not?" — it sharpens the link to why init matters later. P2,
  `LRS-6`.
- A worthy **addition** (P2): "Verify the SGD solution against the closed form
  `w* = (XᵀX)⁻¹Xᵀy` from :numref:`sec_linear_regression`; how close are they, and
  why isn't SGD's answer exactly the closed-form minimizer?" — ties the file back
  to the analytic solution and the non-determinism point. `LRS-6`.

---

## 3. Code & examples

### Does the code teach?

Mostly yes — every cell computes/defines something the prose discusses; there is
no wall of decorative matplotlib. The exceptions are the **TF `_compile_steps` /
`fit_epoch` cell (495–542)** and the **JAX `fit_epoch` cell (544–592)**, which are
genuinely heavy (graph-tracing scaffolding, `_trainer_update_with_bn`, batch_stats
plumbing that is *unused* in this example and only matters for batch-norm much
later). They teach framework performance plumbing, not "SGD from scratch." This is
the price of keeping all four `fit_epoch` variants in one file; see `LRS-7` for
trimming the *prose* around them. The core teaching cells (model 81–130, forward
148–151, loss 168–192, `SGD` 250–319, the pytorch `fit_epoch` 450–474) are compact
and elegant.

### PyTorch

- **Idiomatic and correct for torch 2.x.** `requires_grad=True` at construction
  (88–89), manual `zero_grad` (273–276), in-place `param -= lr*param.grad` under
  `with torch.no_grad()` (464–465). The inline comment correctly explains the
  `no_grad` is needed because the in-place update on a leaf tensor would otherwise
  raise — accurate. Final read-back is wrapped in `no_grad` (627). Converges:
  `error in estimating w: tensor([0.0003, -0.0003])`, `b: tensor([0.0009])`.
- **Only gap:** the mechanic is under-narrated (see `LRS-2`); the code itself is
  the modern, recommended way to write a from-scratch step. No deprecated idioms.

### JAX

- **Idiomatic modern Flax/Optax** and correctly functional: the loss takes
  `(params, X, y, state)` and runs `state.apply_fn` (186–192); the hand-rolled
  `SGD` is a real `optax.GradientTransformation` returning a proper
  `optax.EmptyState()` so it stays `jax.jit`-traceable (293–319) — this is a
  genuinely nice touch and the `:begin_tab:`jax`` note (176–184) explaining *why*
  the JAX loss signature differs is excellent and should be kept.
- **Watch-point — the untagged `forward` cell (148–151) also applies to JAX.**
  Unlike the concise sibling, which splits `forward` into a `pytorch,mxnet,
  tensorflow` tab and a separate `jax` tab (concise lines 194–206), here the single
  untagged `def forward(self,X): return d2l.matmul(X,self.w)+self.b` is shared by
  all four. For Flax this *does* work (during `apply`, `self.w`/`self.b` are bound
  from the param pytree, and `Module.__call__`→`forward` is what `apply_fn`
  invokes), and the JAX run converges (`error in estimating w:
  [0.00047696 -0.0004344]`), so it is **not a bug**. But it is a latent
  inconsistency: the matmul form reads as a pure-tensor op while JAX reaches it
  through Flax's `setup`/`apply` machinery. Worth a one-line clarifying comment or
  matching the concise sibling's split. Judgment, P2 — folded into `LRS-7`/notes,
  not a required change.
- JAX `0.10.0`, no warnings in the manifest. The `_trainer_update_with_bn`
  batch_stats path (555–574) is dead weight for *this* example (no BN), but it
  lives in the shared `Trainer.fit_epoch` — fine to keep; just don't over-explain
  it here (`LRS-7`).

### TensorFlow

- **Modern TF 2.21 / Keras 3** idioms: `tf.Variable(..., trainable=True)`
  (114–115), `tf.GradientTape`, `assign_sub` in the hand-rolled `apply_gradients`
  (286–288), `tf.function(..., reduce_retracing=True)` (521–522). Converges:
  `error in estimating w: [-0.00035381 -0.00042129]`. No warnings.
- **The `_compile_steps` machinery is the most divergent of the four** and the
  least "from scratch" — it exists for performance, not pedagogy. The code is
  correct (the `if not params: params = list(tape.watched_variables())` fallback
  at 510–511 is a reasonable guard), but the surrounding 29-line essay oversells a
  perf concern the text itself calls "modest" here. Trim the prose (`LRS-7`); leave
  the code.

### MXNet

- **Functionally correct and converges** (`error in estimating w:
  [0.00023437 0.00011992]`, mxnet 2.0.0, no warnings). Uses `attach_grad()`
  (101–102), `autograd.record()` (481), `loss.backward()`, manual
  `param -= lr*param.grad` (257–260).
- **Currency — this is the book-wide MXNet question.** Apache MXNet was
  **archived/retired by the ASF (2023)**; presenting its `autograd.record()` /
  `npx.set_np()` idioms as a co-equal live option in 2026 is dated. The repo
  already runs a custom MXNet wheel, so the tab executes, but the *text* should not
  imply MXNet is a current recommendation. This file shouldn't unilaterally drop
  the tab (that's a cross-file decision the overview must make), but it should
  carry the standard one-line archived-framework note and the MXNet variant should
  be visually de-emphasized (ordered last, which it already is in the tab list).
  See `LRS-5`. Flag for the overview: **decide once, book-wide, whether MXNet tabs
  are dropped or footnoted**; this file is a representative case (a hand-rolled SGD
  that is near-identical to PyTorch's, so the MXNet variant adds little).

### Cross-framework consistency & d2l conventions

- **Imports:** one per-framework imports cell near the top (35–65), no re-imports
  later — compliant. `#@save` hygiene is correct (model, loss, `SGD`,
  `configure_optimizers`, `fit_epoch`, `prepare_batch` all saved). Stable cell IDs
  present throughout.
- **Unnecessary divergence:** the `SGD.step`/optimizer cell diverges across
  frameworks *more than the math requires* — pytorch and mxnet are nearly
  identical (`param -= lr*param.grad`), tf uses `apply_gradients`, jax wraps an
  Optax transformation. This is mostly *forced* by each framework's autodiff model
  (and is itself a teaching point the per-framework `:begin_tab:` notes 222–248
  make well), so it's acceptable — but the TF/JAX `fit_epoch` divergence (perf
  scaffolding) is larger than the lesson needs (`LRS-7`).
- **`reduce_mean` vs the `1/2` factor:** the scratch loss keeps the `1/2`
  (172, 191), consistent with `eq_mse`; `linear-regression.md` (lines 212–215)
  already warns that built-in MSE omits it and you must halve the LR — so the
  scratch/concise LR difference is principled and documented upstream. Good. No
  change needed; the Ex. 7 `.sum()` issue (`LRS-6`) is the only loss-scaling slip.

---

## 4. Implementation spec (downstream agents act on THIS)

### LRS-1 — Seed the run so ground-truth recovery reproduces  ·  [P0] · [S] · [authored]
- **Type:** code / currency
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`. Two anchors:
  (i) the training-run cell `#linear-regression-scratch-training-3` (verbatim:
  `model = LinearRegressionScratch(2, lr=0.03)` … `trainer.fit(model, data)`,
  lines ~611–616, untagged → all frameworks); (ii) the prose after it that claims
  recovery ("Indeed they turn out to be very close to each other.", line ~623).
- **Change:** Seed before constructing data/model so the printed errors are
  reproducible for torch/tf/mxnet (JAX already seeds via `PRNGKey(0)` in
  `synthetic-regression-data.md`). Because the run cell is untagged, add a small
  **per-framework** seed cell *immediately before* it. Suggested authored content
  (new cell, framework-tagged; cell-id `linear-regression-scratch-training-seed`):
  - pytorch: `import torch` (already imported) then `torch.manual_seed(d2l.numpy if False else 1)` — concretely:
    ```
    %%tab pytorch
    torch.manual_seed(1)
    ```
  - tensorflow:
    ```
    %%tab tensorflow
    tf.random.set_seed(1)
    ```
  - mxnet:
    ```
    %%tab mxnet
    from mxnet import np as _np  # already have np via imports
    npx.random.seed(1)
    ```
    (use the seeding call available in the repo's mxnet build; if `npx.random.seed`
    is unavailable, use `mx.npx.random.seed(1)` / `mx.random.seed(1)` — the
    implementer should pick whichever the pinned `mxnet==2.0.0` wheel exposes.)
  - jax: no cell needed (already deterministic).
  Then append one interpreting sentence to the recovery paragraph (after line 623),
  authored: *"With the data seed fixed, every run prints the same small residual:
  the leftover error (here ~3e-4 per weight) reflects the σ=0.01 label noise and
  the finite training set, not a bug — a noiseless, full-rank linear problem with
  infinitely many samples would drive it to zero."*
- **Touches:** re-execute the four CPU notebooks and re-capture outputs
  (`make -B _notebooks/<fw>/chapter_linear-regression/linear-regression-scratch.executed`
  then `make capture-outputs FILES=chapter_linear-regression/linear-regression-scratch.md`)
  — the committed `error in estimating` numbers in all four manifests will change
  to the seeded values. Slides reference `training-4` outputs via `inject_outputs`;
  they will pick up the new numbers automatically.
- **Done when:** the run reproduces bit-for-bit across two executions per
  framework; all four `error in estimating w` outputs have |components| ≤ 1e-3 and
  b ≤ 2e-3; `make html` clean; the new interpreting sentence renders.
- **Depends on:** none.

### LRS-2 — Narrate the anatomy of one SGD step  ·  [P1] · [S] · [authored]
- **Type:** teaching / prose
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`, in
  `## Defining the Optimization Algorithm`, immediately after the `SGD` class
  cells and before "We next define the `configure_optimizers` method" (anchor:
  the line `We next define the `configure_optimizers` method`, ~line 322). (It may
  alternatively go at the top of `## Training` before the `fit_epoch` cells —
  implementer's choice; put it wherever the zero→forward→backward→step order is
  first visible.)
- **Change:** insert this authored paragraph (the autograd mechanics are already
  taught in :numref:`sec_autograd`, so this only *applies* them):
  > Each optimization step has four parts, and their order matters. **(1) Zero the
  > gradients.** Autograd *accumulates* gradients by default
  > (:numref:`sec_autograd`), so the gradient from the previous batch must be
  > cleared first — that is what `zero_grad` does. **(2) Forward + loss**, computed
  > inside the autograd graph. **(3) Backward**, `loss.backward()`, which fills
  > each parameter's `.grad` with $\partial L/\partial \theta$ averaged over the
  > minibatch. **(4) Update.** We subtract `lr * param.grad` *in place*. This last
  > step must run **outside** the autograd graph — under `torch.no_grad()` — for
  > two reasons: we do not want the parameter update itself to be differentiated,
  > and an in-place write to a leaf tensor that requires grad would otherwise
  > raise. Forget to zero, and gradients from past batches leak into the present
  > one; forget `no_grad`, and the update either errors or silently grows the
  > graph. The whole of "training a neural network" is this four-line loop, run
  > millions of times.
  Then add a one-clause per-framework footnote (mxnet records the forward in
  `autograd.record()`; tf uses a `GradientTape`; jax computes value-and-grad
  functionally) — or rely on the existing `:begin_tab:` notes at lines 222–248.
- **Touches:** none (prose only). Make sure the `:numref:`sec_autograd`` label
  exists (it does — `chapter_preliminaries/autograd.md`).
- **Done when:** paragraph renders in HTML and PDF; `make html` clean; the
  four-step order is stated in prose, not only in code comments.
- **Depends on:** none.

### LRS-3 — Reference and interpret the loss curve  ·  [P1] · [S] · [authored]
- **Type:** figure / teaching
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`, the prose
  between the training-run cell (`#...training-3`) and the recovery cell
  (`#...training-4`) — currently there is a paragraph "Because we synthesized the
  dataset ourselves… very close to each other." (lines ~618–623). Add the loss-curve
  interpretation just before it.
- **Change:** authored insertion:
  > The `fit` call above produces a live plot of the training and validation loss
  > against the epoch. Both curves fall together and flatten near the irreducible
  > noise floor (with $\sigma=0.01$ the per-example squared loss bottoms out around
  > $\sigma^2/2 \approx 5\times10^{-5}$). Crucially, the validation curve tracks
  > the training curve with **no gap** — a two-dimensional linear model fit on
  > 1000 examples has no capacity to overfit. We will return to the train/validation
  > gap, and what to do when it opens, in :numref:`sec_generalization_basics`.
  If the `ProgressBoard` output can carry a numbered label, also add a
  `:numref:` and refer to it as ":numref:`fig_linreg_scratch_loss`"; if a label
  cannot attach to an inline `d2l.plot` output (it generally cannot the way a
  static `img/*.svg` does), keep the reference as "the loss curve produced above"
  — the interpreting sentence is the required substance, the numbered label is
  optional.
- **Touches:** none (the SVG already exists in all four manifests). Verify the
  `:numref:`sec_generalization_basics`` label exists in
  `chapter_linear-regression/generalization.md` (its `:label:` — confirm the exact
  key; the file is titled "Generalization").
- **Done when:** renders in HTML and PDF; the loss curve is now referenced and
  interpreted in prose; `make html` clean.
- **Depends on:** none.

### LRS-4 — Add a forward-pointing Discussion; fix the dangling "later"  ·  [P1] · [S] · [authored]
- **Type:** coverage / prose
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`. Two edits:
  (i) the `## Summary` section (lines ~665–681); (ii) the three-way-split aside
  (verbatim: "We elide these details for now but will revise them\nlater.", lines
  ~608–609).
- **Change:**
  (i) Rename `## Summary` to `## Discussion` *or* append a final short paragraph to
  the existing Summary (authored):
  > Everything here was built by hand to make the moving parts visible. In
  > :numref:`sec_linear_concise` we rebuild the same model in a few lines using the
  > framework's built-in layer, loss, and optimizer — same loop, same convergence,
  > less glue. The hand-rolled SGD above is the simplest member of a large family:
  > momentum, AdaGrad, RMSProp, and Adam all replace the single update line, and
  > learning-rate schedules anneal $\eta$ over training; these are developed in the
  > optimization chapter (:numref:`chap_optimization`). And the squared loss is a
  > modelling choice — :numref:`sec_weight_decay` adds a penalty on $\|\mathbf w\|$
  > to control overfitting, the first of many such regularizers.
  (Keep this to ~5 sentences; do NOT explain the optimizers — forward-point only,
  per the scope map.)
  (ii) Mechanical: replace
  `We elide these details for now but will revise them\nlater.`
  →
  `We elide these details for now and develop model selection, validation, and the train/validation/test split in :numref:`sec_generalization_basics`.`
- **Touches:** none. Verify labels `sec_linear_concise`, `sec_weight_decay`,
  `chap_optimization` (or the correct optimization-part label), and
  `sec_generalization_basics` resolve. If `chap_optimization` is not the exact key,
  the implementer should use the optimization part's actual `:label:`.
- **Done when:** the close points forward to concise / optimization / weight-decay;
  the dangling "later" is gone and resolves to a real `:numref:`; `make html` clean
  (no broken cross-references).
- **Depends on:** none.

### LRS-5 — MXNet archived-framework note + de-emphasis  ·  [P1] · [S] · [judgment]
- **Type:** currency
- **Where:** `chapter_linear-regression/linear-regression-scratch.md` — the MXNet
  tabs throughout (e.g. the model cell at lines 92–103, the `SGD` at 250–260, the
  `fit_epoch` at 476–493) and the `:begin_tab:`mxnet`` notes.
- **Change:** This is a **book-wide decision the overview must ratify** (flagged
  below), so do the *conservative* version here: do not delete the MXNet tab, but
  (a) ensure MXNet is ordered last in the `tab.interact_select(...)` list (it
  already is: `'mxnet','pytorch','tensorflow','jax'` puts it first in the select —
  consider moving it last for de-emphasis, matching "weight PyTorch and JAX most
  heavily"), and (b) add a single sentence to the chapter's MXNet handling — best
  placed once at the chapter/cover level, not per-file — noting Apache MXNet was
  archived by the ASF in 2023 and is retained only for reference. If the overview
  decides to drop MXNet, this file's MXNet model/`SGD`/`fit_epoch` cells and
  `:begin_tab:`mxnet`` notes (222–229, 477–493) are straightforward to remove since
  the PyTorch variant is near-identical.
- **Touches:** potentially `_quarto.yml` / tab config if reordering or dropping;
  re-render. Coordinate with the cross-file overview (do not act unilaterally).
- **Done when:** the text no longer presents MXNet as a current, co-equal
  recommendation; decision recorded in the overview; `make html` clean.
- **Depends on:** cross-file MXNet decision (overview).

### LRS-6 — Fix/​sharpen exercises  ·  [P2] · [S] · [mechanical + authored]
- **Type:** teaching (exercises)
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`,
  `## Exercises` (lines 685–708).
- **Change:**
  (i) **Mechanical** — Ex. 7, line 703. Replace
  `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`
  →
  `(y_hat - d2l.reshape(y, y_hat.shape)).abs().mean()`
  (and, optional authored clause, append to the exercise stem: " (Note: if you
  *sum* rather than average, the gradient scales with the batch size, so you must
  reduce the learning rate accordingly — cf. the $\tfrac12$-factor discussion in
  :numref:`sec_linear_regression`.)")
  (ii) **Authored** — append a sub-prompt to Ex. 1 (after line 688): " Does the
  symmetry argument that makes zero-initialization fatal for *deep* networks apply
  to this single-layer linear model? Why or why not?"
  (iii) **Authored** — add a new Ex. between current Ex. 6 and 7: " Compare your
  SGD solution to the closed-form least-squares estimator
  $\mathbf w^\star=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$ from
  :numref:`sec_linear_regression`. How close are they? Why does SGD not return
  exactly $\mathbf w^\star$, even though this problem has a unique minimizer?"
- **Touches:** none.
- **Done when:** Ex. 7 uses `.mean()`; the two new prompts render; `make html`
  clean.
- **Depends on:** none.

### LRS-7 — Trim the TF/JAX compile-cost essays  ·  [P2] · [M] · [judgment]
- **Type:** prose / teaching
- **Where:** `chapter_linear-regression/linear-regression-scratch.md`, the
  `:begin_tab:`tensorflow`` block (lines 391–419) and the `:begin_tab:`jax`` block
  (lines 421–442).
- **Change:** Cut each to ~4–6 sentences that state *what* the wrapper does
  (`tf.function`/`@jax.jit` compile the forward+loss+grad+update into one fused
  graph; side effects like plotting cannot live inside, so we split compute from
  reporting) and *why it matters at trace time* (all variables must exist), then
  forward-point: "Graph compilation and its trade-offs are developed in
  :numref:`...`" (the performance/optimization chapter). Keep the JAX note that the
  hand-rolled `SGD` stays JIT-traceable (it's a nice, in-scope detail). Remove the
  blow-by-blow of `_compile_steps`, `_trainer_update_with_bn`, batch_stats (unused
  here), and the "modest at this scale" hedging. Do **not** touch the code cells —
  only the surrounding `:begin_tab:` prose. Optionally also reconcile the untagged
  JAX `forward` cell (148–151) with the concise sibling's split (concise 194–206)
  by adding a one-line comment that for Flax this is reached via `apply`.
- **Touches:** none (prose only).
- **Done when:** each `:begin_tab:` block is materially shorter and forward-points
  for the depth; the from-scratch SGD remains the visible focus of the section;
  `make html` clean.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The four-part decomposition and intro promise** (lines 9–33 + the
  `## Defining the {Model,Loss,Optimizer}` + `## Training` spine) — it is the right
  way to teach this and exactly mirrors the enumerated intro. Don't restructure.
- **All four implementations genuinely converge with clean output** — every
  committed `error in estimating w` is ≤ 5e-4 per component, b ≤ 1e-3, zero
  warnings in any manifest. This is hard-won and must be preserved across the
  re-capture in `LRS-1`.
- **The PyTorch from-scratch step is the modern, correct idiom** — explicit
  `zero_grad`, in-place update under `torch.no_grad()`, with an accurate comment
  about leaf-tensor mutation. Keep the code; just narrate it (`LRS-2`).
- **The JAX functional treatment is genuinely good**: the `:begin_tab:`jax`` loss
  note (176–184) explaining *why* the JAX loss signature differs from the other
  three, and the hand-rolled `SGD` built as a real, JIT-traceable
  `optax.GradientTransformation` returning `optax.EmptyState()` (293–319). This is
  the rare from-scratch-in-JAX that is both correct and idiomatic — preserve it
  (only trim the *surrounding* `fit_epoch` essay).
- **Clean separation of concerns from siblings**: SGD theory and `eq_mse`/the
  built-in-MSE-LR caveat live in `linear-regression.md` (§3.1); the OO machinery
  (`Module`/`Trainer`/`add_to_class`) lives in `oo-design.md`; the concise version
  is the explicit sequel. This file correctly *implements* rather than *re-derives*.
  Preserve that lane discipline — do not import optimizer/init/generalization
  theory here.
- **The physics-modeling and robust-loss exercises** (Ohm, Planck, Huber-by-hint)
  are above the bar — keep them (just fix the `.sum()` slip in `LRS-6`).
