# §3 Linear Regression, slide-deck technical-depth plan

**Scope:** the 7 decks in `chapter_linear-regression/`. This is a *content
architect's* plan: what each deck must teach to hit the top-5-university bar, the
exact math to stage, the figures to reuse or flag, the cells that earn their place,
and an ordered slide arc. No slides are generated here.

## Status of the corpus (read this first)

**All 7 decks are already in north-star form** (cover / divider / kicker / `.cols` /
`.d2l-note`, figures reused). This is NOT a shallow-legacy-to-deep rewrite; it is a
*depth-refinement* pass. The decks are visually decent and structurally sound. The
gaps are specifically technical: a few key derivations are stated but not staged; the
one experiment that should *compute a number on the slide* (vectorization timing) has
no captured output; and two `@!` output-only figures the decks already reference are
missing from the notebooks. Each deck below gives a depth verdict against the rubric
and the concrete additions.

### Three corpus-wide findings (the most important output of this plan)

1. **The vectorization timing experiment shows code with no measured result, in
   every framework.** `linear-regression-vectorization-for-speed-2` (the Python
   loop) and `-3` (the `+` call) are `f'{...:.5f} sec'` expression statements, but
   **the captured notebooks contain NO output for either, in all four frameworks**
   (verified). So the deck's headline claim "the vectorized call is *dramatically
   faster*" is asserted, never shown. This is the single biggest violation of the
   rubric ("code earns its place ONLY by computing something on the slide"). **Fix
   for the author/owner (not slides-only):** change both cells to `print(...)` (a
   bare f-string at cell end *should* echo, but evidently did not capture under the
   slide pipeline), or wrap as `t0=time.time(); ...; print(f'loop: {time.time()-t0:.5f} sec')`
   so a number lands on the slide, then re-execute + `capture-outputs`. A timing
   *bar chart* (loop vs vectorized, log axis) would be even stronger but is optional.
   Until then the two vectorization slides cannot meet the bar. **Flagging this is
   in scope; the cell edit + re-capture is a content-owner action.**

2. **Two `@!` output-only figures referenced by current decks are not captured:**
   `@!linear-regression-the-normal-distribution-and-squared-loss-2` (the bell-curve
   plot) and `@!oo-design-utilities-7` (the ProgressBoard sin/cos animation) have no
   PNG in any notebook (verified). Those figure columns will render empty. **Fix:**
   re-execute + capture those two cells, OR replace the `@!` figure with a reused
   house-style SVG (a static normal-density picture; for ProgressBoard, drop the
   figure and keep the code-only point). Decide per deck below.

3. **`img/mdl-la-projection.svg` is an *approximate* fit for the normal-equation
   geometry.** The linear-regression deck reuses it for "the closed form is a
   projection," but that figure teaches *projection onto a single direction* (from
   Linear Algebra), not *projection of `y` onto the column space of `X`*. It is
   defensible as a stand-in, but a purpose-built `linreg-column-space-projection`
   (the `y` vector, the `col(X)` plane, `Xw*` as the foot of the perpendicular, the
   residual at a right angle) would be markedly better. Flagged as a genuinely-useful
   new figure (do not draw here).

### Figures that already exist and should be reused (verified present)

`img/mdl-linreg-oo-classes.svg` · `img/mdl-linreg-ridge-geometry.svg` ·
`img/mdl-la-projection.svg` · `img/mdl-prob-bias-variance-u-curve.svg` ·
`img/mdl-mlp-double-descent.svg` · `img/mdl-mlp-kfold.svg` ·
`img/mdl-prob-map-prior.svg` · `img/mdl-prob-mle-kl.svg` ·
`img/mdl-opt-sgd-noise-ball.svg` · `img/mdl-opt-lagrange-tangency.svg` ·
`img/mdl-opt-kkt-active-set.svg` · `img/singleneuron.svg` · `img/neuron.svg` ·
`img/fit-linreg.svg` · `img/capacity-vs-error.svg`.

### New figures flagged across the chapter (described, NOT drawn)

- **`linreg-column-space-projection`** (linear-regression.md), least squares as
  orthogonal projection of `y` onto `col(X)`; residual perpendicular to the plane.
- **`linreg-vectorization-speedup`** (linear-regression.md, optional), loop-vs-
  vectorized timing as a two-bar log-scale chart, value-bound to the timing cells.
  Only worth it once the timing outputs are captured; otherwise skip.
- **`linreg-mle-gaussian`** (linear-regression.md, optional), the Gaussian-noise
  picture: a regression line with a small bell curve of `p(y|x)` standing up at one
  `x`, residual = the deviation being scored. Could substitute for the missing
  `@!...-2` density plot AND carry the MLE intuition. `img/mdl-prob-mle-kl.svg` is a
  weaker reuse-only fallback.
- **`linreg-sgd-loss-bowl`** (linear-regression.md / scratch, optional), the convex
  MSE bowl with an SGD path zig-zagging to the min; `img/mdl-opt-gd-bowl-vs-valley.svg`
  or `img/mdl-opt-sgd-noise-ball.svg` are acceptable reuse-only fallbacks.

Everything else reuses existing SVGs. Prefer reuse; only commission the
column-space-projection figure if the author wants the geometry slide to be exact.

---

## Deck 1, `linear-regression.md` (§3.1)

**Depth verdict: GOOD, with 3 fixable gaps.** Already stages the model, squared loss,
the closed form, the projection reading, minibatch SGD, the Gaussian-MLE derivation,
and the neuron view. This is close to the bar. Gaps:
- **(G1, critical)** The vectorization slides show timing code with **no measured
  number** (corpus finding 1). The "dramatically faster" payoff is invisible.
- **(G2)** The normal-equation `w* = (XᵀX)⁻¹Xᵀy` is stated but never *derived* on a
  slide (the one-line `∂_w ‖y−Xw‖² = 2Xᵀ(Xw−y) = 0` is the elegant move) and never
  *computed* (one `torch.linalg.lstsq` on the synthetic data would verify it recovers
  `[2,−3.4]`, tying §3.1 to §3.3). Currently the analytic solution is asserted.
- **(G3)** The bell-curve figure slide (`@!...-2`) is empty (corpus finding 2), and
  there is no forward cross-link to softmax (§4, same affine map + different loss /
  noise model) or to MLPs (§5, stacked affine maps).

**Core spine (the through-line):**
1. A prediction is an **affine map** `ŷ = wᵀx + b`; fitting = minimizing **squared
   loss**, which is **convex** (one global optimum).
2. **Two ways to solve it:** the closed form (a projection) vs minibatch SGD (the
   recipe reused for every model after this), and *why* we lean on SGD.
3. **Where squared loss comes from:** Gaussian-noise maximum likelihood. This is the
   template "match the loss to the noise model" that recurs (Poisson, cross-entropy).
4. Linear regression **is a one-neuron network**, the on-ramp to deep nets.

**Must-show derivations/theorems (stage with fragments):**

- *Normal equation (G2), the elegant 2-fragment derivation:*
  $$\partial_{\mathbf w}\tfrac12\lVert \mathbf y-\mathbf X\mathbf w\rVert^2
    = \mathbf X^\top(\mathbf X\mathbf w-\mathbf y)\stackrel{!}{=}\mathbf 0
    \;\Longrightarrow\; \mathbf X^\top\mathbf X\,\mathbf w^\star=\mathbf X^\top\mathbf y
    \;\Longrightarrow\; \mathbf w^\star=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y.$$
  Then one fragment: "unique iff `XᵀX` invertible (full-rank features)." Already
  partly present; make the derivation a staged build, not a bare result.

- *Gaussian noise ⇒ MLE = least squares (already present, keep, tighten to 3 stages):*
  $$y=\mathbf w^\top\mathbf x+b+\epsilon,\ \epsilon\sim\mathcal N(0,\sigma^2)
   \;\Rightarrow\;
   P(y\mid\mathbf x)=\tfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\tfrac{(y-\hat y)^2}{2\sigma^2}\Big)$$
  $$-\log P(\mathbf y\mid\mathbf X)=\underbrace{\tfrac{n}{2}\log 2\pi\sigma^2}_{\text{const}}
    +\tfrac{1}{2\sigma^2}\sum_i\big(y^{(i)}-\hat y^{(i)}\big)^2.$$
  Punchline fragment: minimizing NLL `=` minimizing MSE (σ drops out).

- *The minibatch SGD update in closed form for this loss* (already present):
  $$(\mathbf w,b)\leftarrow(\mathbf w,b)-\tfrac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B}
    \big(\hat y^{(i)}-y^{(i)}\big)\,(\mathbf x^{(i)},1).$$

**Lead diagram(s):** `img/fit-linreg.svg` (cover/loss); `img/mdl-la-projection.svg`
for the projection slide *or* the new `linreg-column-space-projection` (preferred,
flagged); `img/singleneuron.svg` and `img/neuron.svg` for the network/biology slides.
For the MLE slide, the missing `@!...-2` should become either a recaptured density
plot or the flagged `linreg-mle-gaussian` SVG.

**Code that earns its place:**
- `@linear-regression-vectorization-for-speed-1/-2/-3`, the timing experiment. **MUST
  carry the measured `… sec` outputs** (G1). Currently uncaptured → flag for re-capture.
- *Proposed new computed cell* (content-owner, optional but high-value): a
  `torch.linalg.lstsq` (or `np.linalg.lstsq`) on the §3.3 synthetic `X, y` printing
  `ŵ ≈ [2, -3.4]`, so the closed form is *verified*, not asserted. Scope `only="pytorch"`
  if added only there; ideally added to source for all tabs.
- `@linear-regression-the-normal-distribution-and-squared-loss-1`, the `normal(...)`
  function (illustrative, supports the MLE slide). Its plot `-2` is the missing figure.
- Framework deltas: timing cells differ (`tf.Variable` assign loop; JAX
  `c.at[i].set`); the existing decks handle these via `#@tab`. No `only=` needed for
  the math slides. `linear-regression` here has **no pytorch-only cells** beyond
  what's already tab-split.

**Cross-links / forward-points:** back to §2.3 (dot product / matrix-vector),
§2.4 (gradients), §2.3 projection (`sec_mdl-geometry-linear-algebraic-ops`). Forward:
**§3.3** (recover `w*` on synthetic data), **§3.4** (this update, coded by hand),
**§4 softmax** (same affine map, swap Gaussian→categorical noise ⇒ cross-entropy),
**§5 MLP** (stack affine maps + nonlinearity), **chap_optimization** (SGD variants),
modern note: closed form is `O(d³)`; SGD/iterative is why we scale to billions of
params, and ridge `λI` (exercise 4) is the numerically-stable closed form.

**Slide arc (~15):**
1. Cover, "Linear Regression: the line through the data, and the recipe behind
   everything after."
2. Why/what, predict a number; features `x`, label `y`; want `E[Y|x]`. Fig: `fit-linreg`.
3. Divider 01, The Model.
4. The affine model `ŷ=wᵀx+b`; vectorized `ŷ=Xw+b`. (note: bias ⇒ affine not linear)
5. Squared loss; convex ⇒ one global optimum. Fig: residual gaps on `fit-linreg`.
6. **Closed form, derived** (2-fragment normal-equation build).
7. The closed form **is a projection** onto `col(X)`; residual ⟂. Fig:
   `linreg-column-space-projection` (flagged) / `mdl-la-projection` fallback.
8. **Verify the closed form** (new lstsq cell → recovers `[2,−3.4]`), *if added*.
9. Why SGD anyway: closed form needs an inverse + is `O(d³)` + only linear; SGD is
   the universal recipe. The minibatch update.
10. Divider 02, Vectorization.
11. Two ways to add two vectors (`@...-1`); "only the # of interpreter trips differs."
12. **Loop vs one library call, WITH the measured `… sec`** (G1; flag if uncaptured).
13. Divider 03, Loss meets probability.
14. **Gaussian noise ⇒ MLE = least squares** (3-fragment derivation). Fig:
    `linreg-mle-gaussian` (flagged) / recaptured density plot.
15. Linear regression as **one neuron** (Fig `singleneuron`); biology aside (Fig
    `neuron`, warn: inspiration not blueprint).
16. Recap, model / loss / projection / SGD / vectorize / loss-from-noise; forward to
    softmax + MLP.

---

## Deck 2, `oo-design.md` (§3.2)

**Depth verdict: GOOD.** This deck already nails the rubric's #1 for a *design*
section: it leads with the **rationale** (write the loop once; separate model / data /
optimization), shows the object collaboration, and the standout slide "Why `draw` is
asynchronous" teaches the compile-and-stay-busy theme that recurs all book. Gaps are
minor:
- **(G1)** The `@!oo-design-utilities-7` ProgressBoard figure is empty (corpus
  finding 2). The "watch the loss fall, live" slide loses its picture.
- **(G2)** The object **lifecycle** (who-calls-whom during one `fit`) is described in
  prose but never drawn as a sequence/flow. The static class diagram
  (`mdl-linreg-oo-classes.svg`) shows *structure*; a small *call-flow* would show
  *behaviour* (`Trainer.fit → DataModule.train_dataloader → Module.training_step →
  loss.backward → optim.step`), which is the depth a senior reviewer wants.

**Core spine:**
1. Every model = the **same loop**; factor it once → `Module` / `DataModule` /
   `Trainer`. New work is a **subclass**, not a new loop.
2. Two notebook-friendly mechanics that make OO teachable: `add_to_class` (grow a
   class across cells) and `HyperParameters` (kill `__init__` boilerplate).
3. The async `ProgressBoard` previews the book-wide rule: **keep the hot path pure +
   compiled; push logging/plotting/checkpointing off to the side.**

**Must-show (design rationale, not theorems):**
- The **separation of concerns** as the load-bearing idea: a change to *optimization*
  (grad clipping, LR schedule) touches only `Trainer`; a new *dataset* touches only a
  `DataModule`; a new *model* only a `Module`. State this as the "why," with the class
  figure.
- `add_to_class` in three lines (`setattr` onto a mutable class namespace), the trick
  that lets a long class be taught in short cells.
- The async argument, stated crisply: a compiled step must be **pure** (a `print`
  breaks the trace) and the device runs **ahead** of Python (asking for a number
  *blocks*). Hence `draw` queues + returns; a background thread does the copy + render.

**Lead diagram(s):** `img/mdl-linreg-oo-classes.svg` (the three classes, already
reused well). **Flag a new `linreg-oo-fit-lifecycle`** (a call-flow / sequence sketch
of one `fit` epoch) for G2, genuinely useful, do not draw. For G1: either recapture
`oo-design-utilities-7` (the sin/cos plot) or drop the figure column and keep the
code-only point ("the curve animates as training runs").

**Code that earns its place:**
- `@oo-design-utilities-2` + `@oo-design-utilities-3`, declare-shell-then-attach (the
  `add_to_class` demo). Output of `-3` is `Class attribute "b" is 1` (small, shows the
  bound method sees `self`). **Note: `-3` output not captured**, verify/recapture, or
  the demo's payoff is missing.
- `@oo-design-utilities-1`, the 3-line decorator itself.
- `@oo-design-utilities-5`, `HyperParameters` demo; output `self.a = 1 self.b = 2 /
  There is no self.c = True`. **Also not captured**, verify/recapture.
- `@oo-design-data`, the *entire* `DataModule` base (short, shows the one
  `get_dataloader` hook).
- `Module` / `Trainer` bodies: show the **short `fit` body** (already inlined as a
  trimmed snippet, good; the full per-framework class is too long for a slide).
- **Framework framing (`only=`):** JAX is genuinely different and must stay scoped:
  `Module` is a **dataclass** (no `__init__`), `training_step` returns *(loss, grads)*
  via `jax.value_and_grad`, and `Trainer.fit` threads an explicit PRNG `key` +
  `TrainState` (params live *outside* the module). The current deck already has
  `only="jax"` variants for `Module` and `Trainer`, keep them. TF's `_compile_steps`
  / `tf.function` story is deferred to §3.4 (correct).
- **No pytorch-only cells** here (all four tabs present for every shown cell).

**Cross-links / forward-points:** Lightning as the real-world ancestor (already cited);
forward to **§3.3** (`DataModule` subclass), **§3.4** (`Module` + `fit_epoch`),
GPUs/parallel training + `chap_optimization` (where `Trainer` grows). Modern note: the
async/compiled theme = today's `torch.compile` / `jax.jit` / CUDA-graph training.

**Slide arc (~14):** cover → "one loop, written once" (Fig classes) → divider
*Utilities* → declare-then-grow (`-2`,`-3`) → `add_to_class` in 3 lines (`-1`) → kill
`__init__` boilerplate (`-5`) → ProgressBoard live (code `-7`; Fig recaptured or
dropped) → **why `draw` is async** (the compile theme) → divider *Three base classes*
→ `Module` (except jax) / `Module` jax-dataclass (only jax) → `DataModule` (whole base)
→ `Trainer.fit` body (except jax) / `Trainer` TrainState (only jax) → **fit lifecycle**
(Fig `linreg-oo-fit-lifecycle`, flagged) → recap.

---

## Deck 3, `synthetic-regression-data.md` (§3.3)

**Depth verdict: GOOD.** Leads with the *principle* (synthetic data isolates the bug:
known `w*,b*` ⇒ any failure to recover them is the algorithm's fault), shows the
generative law, the hand-rolled vs framework loader contrast, and the JAX
`drop_remainder` framing. Solid. Gaps are small:
- **(G1)** The "why minibatches" question is answered only implicitly. A senior deck
  should name the three reasons crisply (full batch = accurate but slow + redundant;
  single point = noisy + cache-inefficient; minibatch = the practical middle, and one
  matmul beats a Python loop), currently this lives in §3.1, but §3.3 *introduces the
  minibatch loader* and should own the "why."
- **(G2)** Ground-truth *recovery* is promised but the actual recovery numbers live in
  §3.4/§3.5; a one-line forward pointer with the target `[2,−3.4]` on the peek slide
  would close the loop the deck opens.

**Core spine:**
1. **Why fabricate data:** real data conflates model / optimizer / data failures;
   synthetic removes the third. Recover `w*,b*` ⇒ method works; miss ⇒ your bug.
2. Package the generative process as a **`DataModule`** (separation from §3.2):
   *where batches come from*, kept apart from how a model consumes them.
3. The **same `get_dataloader` protocol twice**: a transparent hand-rolled minibatch
   iterator (teaches what shuffling + batching *are*) and the framework loader
   (shuffles, prefetches, parallelizes). One interface, two implementations.

**Must-show:**
- The generative equation (already present):
  $$\mathbf y=\mathbf X\mathbf w^\star+b^\star+\boldsymbol\epsilon,\quad
    \boldsymbol\epsilon\sim\mathcal N(0,\sigma^2 I),\ \sigma=0.01.$$
- **Why minibatches (G1)** as a 3-bullet build: full batch (slow, redundant) / single
  point (noisy, cache-bound: matrix-vector ≫ many vector-vector) / minibatch
  (32–256, the middle).
- The shuffle-then-slice idea behind the hand-rolled loader (permute indices; yield
  `batch_size` rows), and what it *costs* (in-memory, single-threaded, no prefetch).

**Lead diagram(s):** `img/mdl-linreg-oo-classes.svg` (the `DataModule` slot, already
reused). Two **flagged new** figures would lift this from good to excellent (both
described in the pre-existing outline at `reviews/slides/03-linear-regression/`):
- **`linreg-synthetic-pipeline`**, the generative flow `X∼N → ·w*+b* → +ε → y`,
  value-bound to `w*=[2,−3.4], b*=4.2`. One picture for "we built it, so we know the
  answer."
- **`linreg-minibatch-shuffle`**, a long index vector shuffled then sliced into
  batches of 32 (last = 8), annotating 31 full + 1 partial = 32 (JAX drops → 31).
  Carries the 31-vs-32 framing *visually*.

**Code that earns its place (outputs verified):**
- `@synthetic-regression-data-generating-the-dataset-1`, the `__init__` that draws
  `X` and computes `y` (the generator).
- `@...-generating-the-dataset-2`, instantiate `w=[2,−3.4], b=4.2` (tiny; the ground
  truth we will chase).
- `@...-generating-the-dataset-3`, peek: `features: tensor([-0.2621, -0.2395]) /
  label: tensor([4.4742])` (computed ✓).
- `@...-reading-the-dataset-1`, hand-rolled `get_dataloader` (shuffle + yield).
- `@...-reading-the-dataset-2`, `X shape: [32, 2] / y shape: [32, 1]` (computed ✓).
- `@...-concise-...-data-loader-1/-2`, the framework loader + rewire.
- `@...-concise-...-data-loader-4`, `len(dl)` = **32** (pt/tf/mxnet) vs **31** (JAX,
  computed ✓), the framing-divergence slide.
- **Framework framing (`only=`):** JAX `__init__` threads a `key` + `split`s it for
  independent `X`/`ε`; JAX loader's `drop_remainder=train` ⇒ 31 batches. The current
  deck scopes these (`only="jax"` / `except="jax"`), keep. **No pytorch-only cells.**

**Cross-links / forward-points:** back to §3.2 (`DataModule`); forward to §3.4/§3.5
(recover `w*`), and the modern note already in source (real loaders = workers +
prefetch + streaming; out-of-core shuffle via pseudorandom permutations,
exercise 2). JAX PRNG functional-randomness point connects to §2.5 autograd-era
state handling.

**Slide arc (~12):** cover → why fabricate (3-suspects → synthetic removes one; the
generative eq; Fig `linreg-synthetic-pipeline` flagged) → divider *Generating* →
`DataModule` that builds itself (except jax) / jax functional-key variant (only jax) →
fix ground truth + peek (`-2`,`-3`; add forward-pointer "we'll recover [2,−3.4]") →
**why minibatches** (G1, 3-bullet build) → divider *Reading* → hand-rolled sampler
(`-1`,`-2`; Fig `linreg-minibatch-shuffle` flagged; warn: in-memory/Python/no-prefetch)
→ divider *Built-in loader* → hand to the framework (except jax / only jax) → same
interface + `len` 32-vs-31 (`-3`,`-4`) → recap.

---

## Deck 4, `linear-regression-scratch.md` (§3.4)

**Depth verdict: ADEQUATE → the biggest single depth opportunity in the chapter.**
The deck is structurally north-star and the "four steps, in order" slide is good. But
the **§3 depth anchor for this deck is the gradient of squared loss derived by hand**
(`∂ℓ/∂w`, `∂ℓ/∂b`), and **the deck never derives it.** It shows the loss code and the
SGD update line, and *names* "backward fills each parameter's gradient," but the actual
calculus, the thing that makes "from scratch" meaningful, is absent. This is the gap to
fix.
- **(G1, the headline gap)** No gradient derivation. The whole point of "from scratch"
  is to see what `loss.backward()` computes; show it.
- **(G2)** The training run slide uses `@!linear-regression-scratch-training-3` (the
  live loss plot), that PNG **is** captured (verified ✓), good. Keep it.
- **(G3)** The param-recovery slide is strong (`error in w: [3.5e-05, -3.0e-04]`,
  computed ✓), keep; it pays off §3.3's promise. Add the one-line "exact recovery
  needs full-rank features; in practice we want accurate prediction, not the truth."

**Core spine:**
1. A trainable model = **four pieces**: model (params + forward), loss, optimizer,
   training loop. Build each by hand once.
2. **The gradient is the engine.** Derive `∂ℓ/∂w`, `∂ℓ/∂b` for squared loss; that
   derivative *is* what the optimizer consumes. (G1)
3. **One minibatch = four moves, order matters:** zero grads (autograd accumulates) →
   forward+loss (recording) → backward (fills grads) → update (outside the graph).
4. It **works**: loss → noise floor with **no train/val gap** (2-D model, 1000 pts);
   recovers ground-truth `w,b`.

**Must-show derivation (G1), the centerpiece, staged in fragments:**

For one example, `ℓ = ½(ŷ − y)² = ½(wᵀx + b − y)²`. By the chain rule:
$$\frac{\partial \ell}{\partial \mathbf w}=(\hat y-y)\,\mathbf x,
  \qquad
  \frac{\partial \ell}{\partial b}=(\hat y-y).$$
Averaged over a minibatch `B`:
$$\nabla_{\mathbf w}L=\tfrac1{|\mathcal B|}\sum_{i\in\mathcal B}(\hat y^{(i)}-y^{(i)})\,\mathbf x^{(i)},
  \qquad
  \nabla_b L=\tfrac1{|\mathcal B|}\sum_{i\in\mathcal B}(\hat y^{(i)}-y^{(i)}).$$
Punchline fragments: (a) the `½` is why the gradient is just *(error)·x*, no stray 2;
(b) the gradient is the **error-weighted input**, large residual ⇒ large push; (c)
this is *exactly* what `loss.backward()` computes for you, and what `SGD.step`
subtracts. This single slide turns "from scratch" from a code tour into a teaching of
the mechanism.

**Lead diagram(s):** `img/mdl-linreg-oo-classes.svg` (the four pieces map to the
classes, already reused). For the SGD step, optional reuse
`img/mdl-opt-gd-bowl-vs-valley.svg` or `img/mdl-opt-sgd-noise-ball.svg` (a convex bowl
with a descent path) to picture "step downhill." The live loss plot
(`@!...-training-3`) is the computed figure for the run slide.

**Code that earns its place:**
- `@linear-regression-scratch-defining-the-model-1`, params: small-Gaussian `w`, zero
  `b`, `requires_grad`.
- `@linear-regression-scratch-defining-the-model-2`, `forward`: one matmul + bias.
- `@linear-regression-scratch-defining-the-loss-function`, MSE/2 (pairs with the
  derivation slide, code beside the math it implements).
- `@linear-regression-scratch-defining-the-optimization-algorithm-1/-2`, the ten-line
  SGD (`param -= lr*param.grad`), the update *is* the algorithm.
- `@-linear-regression-scratch-training-seed`, `torch.manual_seed(1)` (**pytorch-only
  cell**; scope `only="pytorch"`, code-only `@-`; it's why the figures/numbers are
  reproducible). The current deck already scopes this correctly.
- `@-linear-regression-scratch-training-3` (code) + `@!...-training-3` (the loss plot,
  captured ✓), the run.
- `@linear-regression-scratch-training-4`, param recovery (computed ✓).
- **Framework framing (`only=`):** JAX stateless loss (`only="jax"`, "params in, loss
  out" → pure ⇒ jit/grad-friendly); TF `assign_sub` optimizer + `tf.function`
  `_compile_steps` story; JAX optax `GradientTransformation` (init/update). The deck
  already has these scoped, keep. The four-step-loop slide is shared (the *idea* is
  universal); the per-framework guard (`torch.no_grad` vs `GradientTape` vs
  `autograd.record`) is a one-line note.
- **pytorch-only cells:** `linear-regression-scratch-training-seed` (no jax/tf/mxnet
  sibling) → `only="pytorch"`. (Already scoped.)

**Cross-links / forward-points:** back to §2.5 (autograd accumulates → why step 1
zeroes), §3.1 (the update derived), §3.2 (the classes), §3.3 (the data + recovery).
Forward: **§3.5** (the same loop, framework API), **§3.7** (add `λ‖w‖²` → one extra
gradient term), **chap_optimization** (momentum/Adam/schedules all replace the single
update line, already noted in source). Modern note: "strip the bookkeeping and
training *any* net is this four-step loop" is the honest through-line.

**Slide arc (~14):** cover (+ "the two-line `nn.Linear`+`MSELoss` hides four parts")
→ four parts, one object graph (Fig classes) → divider *The Model* → params (`-1`) →
forward = one matmul (`-2`) → divider *Loss & Optimizer* → **loss code beside the MSE
formula** (`defining-the-loss-function`) → **GRADIENT DERIVED** (G1 centerpiece) →
stateless loss (only jax) → SGD by hand (`-1`,`-2`; except tf,jax) / TF assign_sub
(only tf) / optax transform (only jax) → divider *Training* → one minibatch = four
steps in order (warn: skip-zero leaks, drop-guard differentiates the update) → seed
(only pytorch, `@-`) → run it: loss → noise floor, no gap (code `@-...-3` + Fig
`@!...-3`) → did it recover `w,b`? (`-4`, computed) → recap (forward to concise + decay).

---

## Deck 5, `linear-regression-concise.md` (§3.5)

**Depth verdict: GOOD.** The deck's thesis is exactly right: a *mapping table* (by
hand → built-in: `w,b`→layer, MSE math→loss, update loop→optimizer) and "WHAT the
framework hides." It reuses `singleneuron.svg` and `mdl-linreg-oo-classes.svg`, scopes
the lazy-init / functional-state framework deltas, and shows the recovered-params
payoff (computed ✓: `error in w: [-0.0001, -0.0002]`). Gaps are minor:
- **(G1)** "What the framework hides" is asserted but could be *shown*: the lazy layer
  defers the input dim until first forward (PyTorch), autograd + the optimizer step are
  hidden inside `fit`. One crisp slide enumerating the three hidden things (init,
  autograd/backward, the optimizer's in-place step), mapped to the three from-scratch
  slides they replace, would sharpen the contrast.
- **(G2)** No timing/overhead comparison vs §3.4. Exercise 6 *asks* to time scratch vs
  concise across 10/100/1000 epochs; surfacing even a sentence ("framework ops are
  fused + tested; the Python-bookkeeping overhead grows with epochs") would add the
  honest engineering depth a senior reviewer expects. (Optional; no captured timing.)

**Core spine:**
1. The four hand-rolled pieces are **so universal frameworks ship them**, tuned +
   tested. Swap each for its built-in counterpart; the **scaffold is unchanged**.
2. **What the API hides** (the depth): lazy shape inference, autograd, the optimizer's
   in-place update, convenience, but know what you're reaching into.
3. Meaningful **framework deltas**: PyTorch `LazyLinear` + `.data` init quirk; Keras /
   Gluon lazy `Dense`; JAX/Flax dataclass + params-in-`TrainState` (no mutable
   weights). The convergence + recovery are identical.

**Must-show:**
- The **by-hand → built-in mapping table** (already present, keep; it's the spine).
- The "what's hidden" enumeration (G1): `init` (was explicit `normal_`/`zeros`),
  `backward` (was the recorded forward + `loss.backward`), `step` (was `param -=
  lr*grad`), each now inside the layer/optimizer/`fit`.
- The MSE-factor gotcha (already present): built-in MSE omits the `½`, averages by
  default, so a swapped-in loss needs the LR halved (ties back to §3.1's note and
  §3.7's λ-mismatch).

**Lead diagram(s):** `img/singleneuron.svg` (one-output `Dense` = the neuron picture,
already reused) and `img/mdl-linreg-oo-classes.svg` (scaffold unchanged, already
reused). No new figure needed.

**Code that earns its place (outputs verified):**
- `@linear-regression-concise-concise-implementation-of-linear-regression`, the one
  import line.
- `@linear-regression-concise-defining-the-model-1`, the layer *is* the model
  (`LazyLinear(1)` / `Dense(1)` / Flax `nn.Dense`).
- `@linear-regression-concise-defining-the-model-2`, `forward` = call the layer.
- `@linear-regression-concise-defining-the-loss-function`, built-in MSE.
- `@linear-regression-concise-defining-the-optimization-algorithm`, one optimizer
  object.
- `@linear-regression-concise-training-1`, the `fit` call (loss plot, verify capture;
  it drives the same loop).
- `@linear-regression-concise-training-2/-3`, reach into the layer; recovery
  `error in w: [-0.0001, -0.0002]` (computed ✓).
- **Framework framing (`only=`):** the model slide splits four ways (PyTorch lazy +
  `.data` quirk; MXNet/TF lazy `Dense`; JAX dataclass+`setup`), already scoped, keep.
  The recover-params slide is jax-vs-rest (JAX reads `kernel`/`bias` from `state`). **No
  pytorch-only cells** (model-1 has all four variants).

**Cross-links / forward-points:** back to §3.4 (every line maps to a hand-rolled one),
§3.1 (`fig_single_neuron`). Forward: **§3.7** (concise weight decay = one optimizer
arg), deeper nets where lazy inference pays off (conv layers, variable-length seqs,
already noted), `chap_optimization` (swap one line for Adam). Modern note: "use the
batteries; but a researcher inventing a new component must know they *can* be built by
hand" (already in source, keep).

**Slide arc (~12):** cover → from hand-rolled to high-level (mapping table) → divider
*The Model* → linear regression is one neuron (Fig `singleneuron`) → one layer, not a
weight vector (only pytorch / only mxnet / only tf / only jax variants) → forward is a
one-liner (`-2`) → **what the API hides** (G1: init/backward/step) → divider *Loss &
Optimizer* → built-in MSE (except mxnet) / Gluon L2Loss quirk (only mxnet) → optimizer
in one call → divider *Training* → same `Trainer` drives it (Fig classes) → fit + same
convergence (`training-1`) → recover params (except jax / only jax; computed) → summary
(+ optional overhead note, G2).

---

## Deck 6, `generalization.md` (§3.6)

**Depth verdict: GOOD, already the chapter's most figure-rich deck.** Ellie/Irene
opener, the train-vs-generalization-error formalism, the bias-variance U-curve, the
**polynomial experiment with the dramatic measured result** (`degree 19: train 0.0000
/ test 49270853259149`, 49 *trillion*, computed ✓), K-fold, and a modern double-
descent coda. It reuses `mdl-prob-bias-variance-u-curve.svg`, `mdl-mlp-kfold.svg`,
`mdl-mlp-double-descent.svg`. This deck is close to exemplary. Gaps are small:
- **(G1)** The two error definitions are shown, but the *reason* training error is
  **biased** (the model was *chosen using* that data ⇒ it flatters itself; test error
  on a *fixed* model is just mean estimation) is stated in one line; it deserves the
  emphasis of a fragment build, since it's the conceptual core.
- **(G2)** The `except="pytorch"` fallback slide ("What the sweep produces") shows the
  U-curve SVG instead of the measured numbers, fine, but the **measured `train/test`
  table** (`-4`) and the **measured U-curve plot** (`@!...-5`) are the rubric's
  "expose a computed result" win and should be the *primary* telling. Confirm `-5`'s
  PNG is captured (the `-4` text is ✓).

**Core spine:**
1. **Fitting the training data is not the goal**, generalizing is. Memorize (Ellie)
   vs pattern (Irene).
2. **Two errors:** training error (a *statistic* on data we have) vs generalization
   error / risk (an *expectation* we can never compute, only estimate on held-out
   data). Training error is **biased**; the model was picked using it. (G1)
3. **Bias-variance / model complexity:** test error is **U-shaped** in capacity; the
   sweet spot. Complexity = parameter count **and** the range of values (Popper's
   falsifiability; kernels have ∞ params yet stay controlled).
4. **Measure it:** polynomial-degree sweep traces the U-curve from real numbers.
5. **Model selection:** never on the test set; validation / K-fold. Modern twist:
   **double descent** breaks the classical U for huge models.

**Must-show:**
- The two definitions (already present):
  $$R_{\text{emp}}=\tfrac1n\sum_i \ell(\mathbf x^{(i)},y^{(i)},f),\qquad
    R=\mathbb E_{(\mathbf x,y)\sim P}[\ell(\mathbf x,y,f)].$$
- **Why `R_emp` is biased (G1)** as a 2-fragment build: test error evaluates a *fixed*
  `f` on fresh samples ⇒ plain mean estimation, unbiased; training error's `f` *depends
  on* the sample ⇒ optimistic. This is the hinge of the whole section.
- The bias-variance trade-off named (full decomposition deferred to
  `sec_mdl-statistics`, correct).
- The IID assumption + what breaks without it (`P→Q` ⇒ nothing can be said).

**Lead diagram(s), all reused, verified present:**
`img/mdl-prob-bias-variance-u-curve.svg` (the U-curve) · `img/mdl-mlp-kfold.svg`
(K-fold) · `img/mdl-mlp-double-descent.svg` (double descent). `img/capacity-vs-error.svg`
is the book's original; the `mdl-` versions are the polished house-style and are what
the deck already uses, keep. No new figure needed.

**Code that earns its place (the measured experiment, the deck's spine):**
- `@-generalization-polynomial-curve-fitting-2`, build the data (degree-3 target, 20
  train points so high degrees overfit).
- `@-generalization-polynomial-curve-fitting-3`, `fit_degree(d)` (least squares on the
  first `d+1` power columns; train + test MSE).
- `@generalization-polynomial-curve-fitting-4`, **the result table** (computed ✓):
  `underfitting (deg 1): train 1.67 / test 1.73`, `just right (deg 3): 0.0136 / 0.0129`,
  `overfitting (deg 19): 0.0000 / 4.9e13`. The 49-trillion test error is the slide's
  punch, scope `only="pytorch"` (the deck already does this for the result slides).
- `@!generalization-polynomial-curve-fitting-5`, the measured U-curve plot
  (`only="pytorch"`; **verify PNG captured**).
- **pytorch-only scoping:** the result slides (`-4`, `-5`) are `only="pytorch"`
  (numeric/plot output is shown on the pytorch tab); the other tabs get the SVG-only
  "what the sweep produces" fallback (`except="pytorch"`). Already scoped this way;
  keep, and **list these in the report.** The data/fit cells (`-2`,`-3`) are
  framework-agnostic numpy, shown on all tabs as `@-` code-only.

**Cross-links / forward-points:** forward to **§3.7 weight decay** (the *next*
regularizer; "range of values parameters take" is exactly what `λ‖w‖²` controls, the
deck already plants this) · `chap_classification_generalization` (rigorous theory) ·
`sec_mdl-statistics` (bias-variance derived) · `sec_generalization_deep` (double
descent in full). Modern note: the double-descent coda is the right "modernize but
forward-point" move, keep it as a *coda*, not the main story.

**Slide arc (~17, already close):** cover → two students (Ellie/Irene) → divider *Two
Errors* → training vs generalization error (the two formulas) → **why training error
is biased** (G1 build) → IID + what breaks without it → divider *Model Complexity* →
bias-variance U (Fig u-curve) → reading the gap → what makes a model complex (Popper;
range-of-values) → divider *The Demo* → poly fitting = least squares (`-2`) → one fit
per degree (`-3`) → under/just-right/over **result table** (`-4`, only pytorch) → sweep
= measured U-curve (`@!-5`, only pytorch) / SVG fallback (except pytorch) → more data,
more room → divider *Model Selection* → never select on test → K-fold (Fig kfold) →
divider *A Modern Twist* → double descent (Fig double-descent) → rules of thumb.

---

## Deck 7, `weight-decay.md` (§3.7)

**Depth verdict: GOOD, the most theoretically complete deck in the chapter.** It
already hits *all three* §3 depth anchors: the ℓ2 penalty + shrink-and-update
derivation, the **MAP / Gaussian-prior reading**, and the **constrained-optimization
norm-ball geometry** (reusing the excellent `mdl-linreg-ridge-geometry.svg`), plus
AdamW as the modern decoupled variant and the rigged-to-overfit experiment with
measured weight-norm shrinkage (`0.0092` → `0.00037`, an order of magnitude, computed
✓). The two "readings worth keeping" slide (AdamW + Bayesian) is exactly the
modernize-but-forward-point bar. Gaps are minor:
- **(G1)** The **MAP derivation** is stated as prose in a callout, not *staged*. The
  elegant move (NLL + Gaussian prior `−log p(w)` = `(λ/2)‖w‖²` + const ⇒ MAP = ridge)
  deserves a 2-3 fragment build, since it's a §3 must-hit and it ties directly back to
  §3.1's MLE derivation (the symmetry is the teaching moment).
- **(G2)** The constrained-optimization view is shown geometrically (the ridge figure)
  but the *equivalence* itself (`min L + (λ/2)‖w‖²` ⟺ `min L s.t. ‖w‖ ≤ t`, with λ the
  Lagrange multiplier) could be one crisp line; `img/mdl-opt-lagrange-tangency.svg` or
  `img/mdl-opt-kkt-active-set.svg` would reinforce the Lagrangian reading if a second
  geometry slide is wanted (optional; the ridge figure already carries the main point).
- **(G3, optional)** "vs early stopping" appears in the §3 anchor list but not the deck;
  a one-line note (early stopping ≈ implicit ℓ2 in the SGD trajectory) would complete
  the regularizer comparison. Low priority.

**Core spine:**
1. **When more data isn't an option, restrict the weights.** Among all `f`, `f=0` is
   simplest; measure complexity by distance of `w` from zero. Dropping features is too
   blunt (monomials blow up `binom(k-1+d, k-1)`).
2. **The ℓ2 penalty** `L + (λ/2)‖w‖²` ⇒ the **shrink-and-update** rule (= "weight
   decay"); `λ` is a *continuous* complexity dial.
3. **Three readings of the same penalty:** (a) geometry, ridge **shrinks** (round
   ball, tangential contact) vs lasso **selects** (diamond corner); (b) Bayesian, a
   Gaussian **prior**, the fit is the **MAP** estimate; (c) optimizer, frameworks add
   `λw` to the gradient (and AdamW *decouples* it for adaptive methods).
4. It **works**: rigged 200-dim / 20-sample problem overfits at `λ=0`, generalizes at
   `λ=3`, weight norm shrinks 10×.

**Must-show derivations (stage all three "readings"):**

- *Shrink-and-update (already present, keep):*
  $$\mathbf w\leftarrow(1-\eta\lambda)\mathbf w-\tfrac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B}
    \mathbf x^{(i)}(\hat y^{(i)}-y^{(i)}).$$
  The `(1−ηλ)` factor *decays* every weight before the data term, the name.

- *Constrained-optimization equivalence (G2):*
  $$\min_{\mathbf w} L(\mathbf w)+\tfrac{\lambda}{2}\lVert\mathbf w\rVert^2
    \iff \min_{\mathbf w} L(\mathbf w)\ \text{s.t.}\ \lVert\mathbf w\rVert\le t,$$
  λ is the Lagrange multiplier; the solution is where a loss contour first **touches**
  the constraint region (tangent for ℓ2 ⇒ shrink; corner for ℓ1 ⇒ sparsity).

- *MAP / Gaussian-prior (G1), the centerpiece build, mirror of §3.1's MLE:*
  $$\mathbf w\sim\mathcal N(\mathbf 0,\lambda^{-1}I)\Rightarrow
    -\log p(\mathbf w)=\tfrac{\lambda}{2}\lVert\mathbf w\rVert^2+\text{const}.$$
  $$\underbrace{-\log p(\mathbf y\mid\mathbf X,\mathbf w)}_{\text{MLE = }\,\tfrac{1}{2\sigma^2}\sum(\hat y-y)^2}
    \;\underbrace{-\log p(\mathbf w)}_{=\,\tfrac{\lambda}{2}\lVert\mathbf w\rVert^2}
    \;\Rightarrow\; \text{MAP estimate} = \text{ridge regression.}$$
  Punchline: "§3.1 said squared loss = Gaussian-noise MLE; weight decay just adds a
  Gaussian *prior* on `w`. λ = prior precision: bigger λ = stronger belief weights are
  small." This closes the loop the chapter opened.

**Lead diagram(s):** `img/mdl-linreg-ridge-geometry.svg` (the ridge-vs-lasso norm-ball
geometry, already reused, the deck's best figure). For the Bayesian slide,
`img/mdl-prob-map-prior.svg` (prior pulling the estimate, strong reuse). Optional
second geometry: `img/mdl-opt-lagrange-tangency.svg` (the Lagrangian tangency) for G2.
No new figure required.

**Code that earns its place (outputs verified):**
- `@weight-decay`, the one import cell (setup).
- `@-weight-decay-high-dimensional-linear-regression`, the rigged `Data` (200 inputs,
  20 train, tiny signal + noise), code-only, the overfit setup.
- `@weight-decay-defining-ell-2-norm-penalty`, `l2_penalty` (one line; pairs with the
  penalty math).
- `@weight-decay-defining-the-model-1`, subclass + fold penalty into the loss.
- `@weight-decay-training-without-regularization`, `λ=0`: `L2 norm of w: 0.0092`
  (computed ✓) + the (captured?) loss plot, the overfit.
- `@weight-decay-using-weight-decay`, `λ=3`: `L2 norm of w: 0.00037` (computed ✓):
  10× shrink, the payoff. **These two slides are the experiment; the contrast in the
  two norms is the measured result, foreground it.**
- `@weight-decay-concise-implementation-1`, the built-in decay (per-framework).
- `@weight-decay-concise-implementation-2`, fit + matching shrunken norm.
- **Framework framing (`only=`):** the concise decay genuinely differs and must stay
  scoped, PyTorch per-param-group `weight_decay`; MXNet `wd` + `wd_mult=0` on bias;
  Keras `kernel_regularizer` + `wd/2` factor (warn); JAX `optax.chain` +
  `optax.masked` (no `weight_decay` in `optax.sgd`). The deck already has four
  `only=` slides, keep. **No pytorch-only cells** (every shown cell has four
  variants); the scratch experiment cells are framework-agnostic.

**Cross-links / forward-points:** back to §2.3 (ℓp norms), §3.1 (MLE, the MAP slide's
mirror), §3.6 ("range of values parameters take" = exactly what λ controls). Forward:
**`sec_adam` / AdamW** (decoupled decay, already in source), `chap_optimization`,
lasso/ℓ1 (feature selection, exercise 3), and "apply to every layer of a deep net" (the
generalization to nonlinear models, already noted). The AdamW + Bayesian "two
readings" slide is the model for modernize-but-forward-point.

**Slide arc (~16, already close):** cover → when more data isn't an option (drop-
features too blunt; restrict values; `‖w‖²`) → add the penalty `L+(λ/2)‖w‖²` (λ knob;
½ cosmetic; ridge vs lasso named) → **ridge shrinks / lasso selects** (Fig
ridge-geometry; + optional Lagrange-tangency for G2 equivalence) → why it's "weight
decay" (the `(1−ηλ)` shrink-update) → divider *From Scratch* → setup (`@weight-decay`)
→ rigged-to-overfit problem (`-high-dimensional...`; warn 200 params / 20 points) →
penalty + model (`l2_penalty`, `-the-model-1`) → **λ=0 overfits** (`0.0092`, plot) →
**λ=3 generalizes** (`0.00037`, 10× shrink) → divider *The Built-In Way* → decay in
optimizer (only pytorch) / Gluon (only mxnet) / Keras regularizer (only tf, warn) /
optax chain+mask (only jax) → same effect, less code (`-2`; the λw-on-grad vs
penalty-on-loss subtlety) → **two readings: AdamW + the MAP / Gaussian-prior build**
(G1 centerpiece; Fig map-prior) → summary.

---

## Cross-deck observations (top 3)

1. **The chapter is already north-star; the work is depth, not rebuild.** Every deck
   has covers/dividers/kickers/figures and reuses the polished `mdl-*` SVGs. The
   regeneration should treat the *current* slide blocks as the baseline and surgically
   add the missing **derivations** (gradient of squared loss in §3.4; staged MAP in
   §3.7; staged normal-equation in §3.1) and **computed results**, not start over.
   The pre-existing outlines in `reviews/slides/03-linear-regression/` were written
   against the *legacy* blocks and now over-state the rebuild needed; this plan
   supersedes them for depth.

2. **Two concrete capture/owner actions gate the bar, flag them loudly.** (a) The
   §3.1 **vectorization timing cells produce no output in any framework**, the deck's
   "dramatically faster" claim is unsubstantiated until the cells are re-authored to
   `print(...)` and re-captured. (b) Two `@!` figure slides are empty
   (`linear-regression-...-squared-loss-2` bell curve; `oo-design-utilities-7`
   ProgressBoard), and three small text outputs in §3.2 (`utilities-3`, `-5`) and the
   §3.6 U-curve plot (`-5`) need verifying. These are content-owner / re-capture
   actions, **not** slides-block edits; a slides-only author cannot fix them and should
   either recapture or swap in a reused SVG / code-only point per the per-deck notes.

3. **The chapter has one beautiful unifying thread, make every deck point at it.** §3.1
   derives *squared loss = Gaussian-noise MLE*; §3.4 derives *the gradient that SGD
   consumes*; §3.7 derives *weight decay = a Gaussian prior ⇒ MAP = ridge*. Staged
   side-by-side, MLE (§3.1) and MAP (§3.7) are the same derivation plus a prior, that
   symmetry is the single most valuable cross-link in the chapter and is currently
   under-played (it lives in a §3.7 callout). The only genuinely-new figure worth
   commissioning is **`linreg-column-space-projection`** (§3.1 geometry, the reused
   `mdl-la-projection.svg` teaches the wrong projection); the optional
   `linreg-synthetic-pipeline` / `linreg-minibatch-shuffle` (§3.3) and
   `linreg-oo-fit-lifecycle` (§3.2) would each lift a good deck to excellent.
   Everything else reuses figures that already exist.
