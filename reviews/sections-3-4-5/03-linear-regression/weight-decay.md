# Review — chapter_linear-regression/weight-decay.md  (§3.7 "Weight Decay")

**Role in the chapter:** The chapter's first *regularization* technique. It picks
up where `generalization.md` ends (which explicitly forward-points to "weight
decay, your first practical regularization technique"), grounds the
"range-of-parameter-values" notion of complexity in a concrete penalty, derives
the shrinkage update, and demonstrates it on a deliberately overfit
high-dimensional synthetic regression — from scratch and with the framework
optimizer.

**Verdict:** A solid, correct, well-paced section that already mostly meets the
bar: the intuition ($f=0$ is simplest), the $\frac{\lambda}{2}\|\mathbf w\|^2$
penalty, the closed-form shrinkage update $(1-\eta\lambda)\mathbf w - \ldots$,
and the scratch-vs-concise pairing are all clean and teach well. The single
highest-value change is **fixing the PyTorch concise example**, whose committed
output ($\|\mathbf w\|^2 = 0.013$) is ~8× the from-scratch result ($0.0016$) and
~8× the other three frameworks' concise results ($\approx 0.0015$–$0.0021$),
directly contradicting the prose claim that "the plot looks similar to … from
scratch." After that, the section needs three small currency/scope touches: a
one-line AdamW pointer, a back-pointer tying the $\ell_2$ penalty to the MLE/MAP
view already developed in `linear-regression.md`, and a *geometry* figure (the
one thing the section lacks that the gold-standard chapters all have). It is not
yet best-in-class, but it is close, and the gaps are surgical.

**Grade:** B+. Assignable today with one caveat (the PyTorch number must be
reconciled or the claim softened); a figure + the MLE/MAP back-pointer + an
AdamW one-liner would lift it to A−/A.

**Top priorities (ranked):**
1. [P0] **PyTorch concise output contradicts the text.** `weight-decay-concise-implementation-2` prints `L2 norm of w: 0.013` for PyTorch — *larger* than the unregularized scratch run (0.0081) and ~8× the other frameworks' concise norms (jax 0.0017, tf 0.0021, mxnet 0.0015). The prose ("plot looks similar to … from scratch") is false for the primary framework. Root cause is the loss-scale mismatch (scratch loss carries an explicit $1/2$; the concise `LinearRegression.loss` uses `nn.MSELoss()` with no $1/2$, so the data-fit gradient is 2× larger relative to the same `wd`, raising the equilibrium norm) compounded by SGD not having fully converged in 10 epochs at lr 0.01. **Fix:** reconcile the convention (see §3) so all four concise norms land near the scratch value, *or* re-word the claim to "comparable" and add one sentence explaining why a framework's `weight_decay` (gradient-side $\lambda\mathbf w$) and a manual $\frac{\lambda}{2}\|\mathbf w\|^2$ loss penalty need not coincide numerically. **This is itself a teachable point and should not be papered over.**
2. [P1] **No figure.** Every exemplar section (calculus, linear-algebra) opens on or builds around a picture; this one has only loss curves. Add **one** schematic SVG: the ridge geometry — elliptical loss contours of $L(\mathbf w)$ meeting the circular constraint $\|\mathbf w\|^2\le t$, with the unconstrained optimum $\hat{\mathbf w}$ outside and the regularized solution where the contour kisses the ball. This single figure carries the whole idea and connects the penalty form to the constrained form.
3. [P1] **Currency — AdamW, in one line + `:numref:`.** The text already hedges ("$\ell_2$ regularization may not be equivalent to weight decay for other optimization algorithms," line 216) but cites nothing. Add the modern result by name (Loshchilov & Hutter 2019; decoupled weight decay / AdamW) with a forward pointer to `:numref:`sec_adam``. *Do not* expand into a treatment — Optimization owns it.
4. [P1] **Tie the penalty to the probabilistic view already in the chapter.** `linear-regression.md` develops squared loss as Gaussian-noise MLE (its §"…as maximum likelihood estimation"). The $\ell_2$ penalty is exactly the *MAP* estimate under a Gaussian prior on $\mathbf w$ — and Exercise 6 already asks the reader to discover this. Add two sentences making the link explicit (back-pointer to the MLE section; forward to `chap_regression` resources for ridge-as-MAP), turning a dangling exercise into a through-line.
5. [P2] **MXNet currency.** Apache MXNet was archived by the ASF (2023). It is still presented as a co-equal first-class tab (e.g. the `:begin_tab:`mxnet`` Gluon note, lines 440–451). Recommend de-emphasizing per the chapter-wide policy.

## 1. Coverage

### Add
- **The MLE/MAP bridge (P1, high value, in-scope).** Lines 154–219 justify $\ell_2$
  on *computational* grounds ("we do this for computational convenience," line 156)
  and *robustness* grounds, but never connect to the probabilistic story the
  chapter already told. A best-in-class treatment closes the loop. Proposed
  insertion after line 219:
  > Equivalently, weight decay has a Bayesian reading. Recall
  > (:numref:`sec_linear_regression`) that minimizing the squared loss is
  > maximum likelihood estimation under Gaussian noise. Adding
  > $\frac{\lambda}{2}\|\mathbf w\|^2$ is exactly placing an isotropic Gaussian
  > *prior* $\mathbf w \sim \mathcal N(\mathbf 0, \lambda^{-1} \mathbf I)$ on the
  > weights and computing the *maximum a posteriori* (MAP) estimate; $\lambda$ is
  > the prior precision. Larger $\lambda$ asserts a stronger prior belief that
  > weights are near zero. This is the classical *ridge regression* estimator,
  > whose shrinkage geometry the references in :numref:`chap_regression`
  > (ESL, Bishop) develop in depth.

  This is squarely in scope (it uses results already in the chapter and forward-
  points the depth), and it pre-rewards Exercise 6.
- **AdamW / decoupled weight decay (P1, one line + pointer — scope-limited).**
  Per the guide's scope map this is *mostly out of scope*; honour currency with a
  single corrected sentence. Replace the vague hedge at lines 215–219 with:
  > For plain SGD the penalty $\frac{\lambda}{2}\|\mathbf w\|^2$ and the
  > shrink-then-step update above are identical. For adaptive optimizers such as
  > Adam this is *not* the case: an $\ell_2$ loss penalty gets rescaled by the
  > per-coordinate second-moment estimate, which is why decoupled *weight decay*
  > (AdamW, :citet:`loshchilov2019decoupled`) is now the default for transformers
  > and is what the framework `weight_decay` argument actually implements. We
  > develop optimizers in :numref:`sec_adam`.

  (The `optax.add_decayed_weights` already used in the JAX tab *is* decoupled
  decay — worth a half-sentence so the JAX reader isn't misled into thinking it
  matches the from-scratch loss penalty.)
- **A closed-form sanity check (P2, optional).** For ridge there is a closed form
  $\mathbf w = (\mathbf X^\top\mathbf X + \lambda \mathbf I)^{-1}\mathbf X^\top
  \mathbf y$; one sentence noting the penalty makes the normal equations
  invertible even when $d>n$ (exactly the $d=200, n=20$ regime of the example!)
  would deepen the example's payoff and connect to `linear-regression-scratch`'s
  analytic-solution discussion. Optional; do not let it bloat the section.

### Remove / trim
- **Monomial preamble is too long for its payoff (lines 21–47).** Two and a half
  paragraphs and the $\binom{k-1+d}{k-1}$ combinatorics motivate "we need a
  finer tool than dropping features," then are never used again. Tighten to ~5
  lines: the count of degree-$d$ monomials blows up combinatorially, so feature
  pruning is too coarse — motivating a *continuous* complexity knob. The
  `${k - 1 + d} \choose {k - 1}$` markup at line 43 is also malformed LaTeX
  (it renders the braces literally / mis-stacks; should be
  `\binom{k-1+d}{k-1}`) — **fix regardless of trimming.**
- **RKHS aside (lines 581–590) is a dead end.** A two-sentence Wikipedia-linked
  detour about RKHS that "scale[s] poorly" then drops the reader at "apply weight
  decay to all layers." Either cut it or convert to a one-line forward pointer.
  As written it teaches nothing and breaks the flow into the Summary.

### Reorder / restructure
- Ordering relative to siblings is correct: `generalization.md` → `weight-decay.md`
  is the right sequence and the handoff is explicit. No cross-file move needed.
- *Within* the file, §3.7.2 "High-Dimensional Linear Regression" (data only) is
  separated from where the data is *used* (§3.7.3.3) by the entire penalty/model
  definition. This is fine, but consider folding the one-paragraph data
  description into the top of §3.7.3 so the reader meets the $d=200,n=20$ "perfect
  overfitting setup" immediately before running it. Minor.

## 2. Teaching quality

### Structure & flow
Good spine: Norms/penalty → data → scratch → concise → summary. Five top-level
`##` sections, properly nested `###`. Matches the exemplar shape. The opening
hook (the $f=0$-is-simplest intuition, lines 92–98) is genuinely nice — it is the
"intuition before formalism" the guide asks for. Two flow defects: the monomial
over-build at the top (trim, above) and the RKHS dead-end before the Summary
(cut, above).

### Figures
- **Inventory:** three computed loss-curve plots (`...training-without-regularization`,
  `...using-weight-decay`, `...concise-implementation-2`), all log-y, all four
  frameworks. These are legitimate *data plots that teach a result* (per the
  house rule) and should stay. No illustrative SVGs exist.
- **No figure-drawing code in teaching cells** — clean on the house convention.
- **Missing (the P1):** the ridge-geometry contour figure described in the Top
  Priorities. This is the canonical picture for this topic (ESL Fig 3.11; Bishop
  Fig 3.4) and its absence is the clearest gap vs. the calculus/linear-algebra
  exemplars. Build it as a committed generator SVG
  (`img/mdl-…-ridge-geometry.svg` via the `mdl-figure` workflow), included with
  no drawing code in the notebook. An optional second panel contrasting $\ell_2$
  (circular ball → shrinkage) with $\ell_1$ (diamond → corner solution / sparsity)
  would also pay for the lasso paragraph at lines 164–188, which currently
  asserts "concentrate weights … clearing the other weights to zero" with no
  picture.

### Prose & clarity
- Mostly breathes well. Worst offenders:
  - **Line 43:** malformed binomial `${k - 1 + d} \choose {k - 1}$` → `$\binom{k-1+d}{k-1}$`. (Correctness, not just style.)
  - **Lines 154–161:** the "why squared not standard norm" explanation is correct but wordy ("the sum of derivatives equals the derivative of the sum"); tighten to one sentence: "Squaring drops the square root, so the gradient is the clean $\lambda\mathbf w$ rather than $\lambda\mathbf w/\|\mathbf w\|$." This also surfaces *why* the update has the simple shrink form.
  - **Lines 215–219:** the "may not be equivalent to weight decay for other optimization algorithms … the idea still holds true" hedge is exactly the AdamW point made vaguely. Replace with the named version above.
- The shrinkage update derivation (lines 190–210) is excellent — the
  $(1-\eta\lambda)$ factor and the "why it's called *decay*" explanation are
  textbook-clean. Keep verbatim.

### Exercises
Reasonable but uneven; sharpen for a top course:
- **Ex 1 mislabeled:** asks to "Plot training and validation **accuracy** as a
  function of $\lambda$" — this is a *regression* task; there is no accuracy.
  Should be **loss** (or norm). Fix the word. (P1 — it's wrong, not just weak.)
- **Ex 6 (Bayesian / MAP) is the best one** but currently arrives with no
  scaffolding because the MLE/MAP link is absent from the prose. Adding the §1
  MLE/MAP paragraph turns this from a leap into a guided derivation. Keep, and
  consider extending: "Show that the prior variance is $1/\lambda$."
- **Add a numerical-insight exercise** worthy of the bar, e.g.: "The example uses
  $d=200, n=20$. Without regularization the least-squares problem is
  underdetermined (infinitely many zero-training-loss solutions). Explain, using
  the closed form $(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}$, why any
  $\lambda>0$ makes the solution unique. What happens to the solution as
  $\lambda\to 0^+$?" This connects the toy setup to *why* ridge exists.
- **Add a decoupling micro-experiment (forward-pointer flavored):** "Swap SGD for
  Adam in the concise model with the same `weight_decay`. Does the converged
  $\|\mathbf w\|^2$ match? Why might it differ? (See :numref:`sec_adam`.)" —
  makes the AdamW point experiential without importing the chapter.
- Ex 4 (Frobenius) and Ex 3 ($\ell_1$ update) are good mechanical builders — keep.

## 3. Code & examples

### Does the code teach?
Yes — every cell computes something the prose discusses (the one-line `l2_penalty`,
the `WeightDecayScratch` subclass that adds exactly the penalty, the
$\lambda=0$ vs $\lambda=3$ contrast). No boilerplate walls. The
scratch→concise pairing is pedagogically strong. The defect is *numerical
consistency across tabs* (below), not teaching intent.

### PyTorch
- **The P0.** Concise run prints `0.013` (manifest
  `weight-decay-concise-implementation-2`), vs scratch λ=3 → `0.00155` and
  unregularized → `0.0081`. The number is essentially unregularized-scale,
  falsifying the "looks similar to from scratch" claim. Mechanism: `LinearRegression.loss`
  is `nn.MSELoss()` (no $1/2$), whereas `WeightDecayScratch` adds
  $\frac{\lambda}{2}\|\mathbf w\|^2$ to a loss that *does* carry $1/2$; PyTorch's
  `weight_decay` adds $\lambda\mathbf w$ on the gradient side, which against the
  2×-larger data gradient produces a different (larger) equilibrium norm, and 10
  epochs at lr 0.01 do not close the gap. **Fix options, in order of preference:**
  (a) make the conventions match (e.g. document that concise `wd` here is not the
  same $\lambda$ as the scratch penalty, and/or train longer) so the numbers
  agree and the claim holds; or (b) keep the divergence and *teach it* — add a
  sentence: "Note the converged norm differs from the scratch run: framework
  `weight_decay` decays on the gradient ($\lambda\mathbf w$) while our scratch
  penalty was $\frac{\lambda}{2}\|\mathbf w\|^2$ on the loss; with `MSELoss` (no
  $1/2$) these are not the same $\lambda$." Either way the current text is wrong
  and must change.
- **Stray warning (P2):** the same cell's *scratch* sibling
  (`weight-decay-training-without-regularization`) emits
  `UserWarning: Converting a tensor with requires_grad=True to a scalar …` from
  `float(l2_penalty(model.w))` (manifest stderr). Wrap in
  `with torch.no_grad():` or use `.detach()` — a top-tier text shouldn't print
  autograd warnings in a clean run.
- Otherwise idiomatic and modern (torch 2.11): the per-parameter-group
  `weight_decay` excluding bias (lines 496–499) is exactly right and is a nice
  teaching detail.

### JAX
- Modern and correct (`jax==0.10`, `optax`). The `optax.chain(masked(add_decayed_weights), sgd)`
  pattern (lines 524–534) is the *right, current* way to do decoupled decay and
  the masking-out-of-bias comment is excellent. Converged norm `0.0017` — sensible,
  matches scratch.
- **Subtle teaching gap:** `add_decayed_weights` *is* decoupled (AdamW-style)
  weight decay, not an $\ell_2$ loss penalty. With `sgd` they coincide, so it's
  fine here, but a half-sentence (per the AdamW add) would prevent the JAX reader
  from over-generalizing. The cleanest of the four concise implementations.

### TensorFlow
- The `kernel_regularizer=l2(wd/2)` with the inline comment explaining Keras'
  `l2` has no $1/2$ factor (lines 508–513) is **exactly the right instinct** — and
  ironically it's the convention-matching the *PyTorch* tab fails to do. Converged
  norm `0.0021` — consistent. Modern Keras-3-compatible API; fine.
- Minor: `tf.add_n(self.net.losses)` is idiomatic; no change needed. Note this tab
  regularizes via the loss (like scratch), which is *why* its number matches and
  PyTorch's doesn't — reinforcing that the PyTorch discrepancy is a real
  convention bug, not noise.

### MXNet
- Code is correct for the legacy Gluon API (`wd` + `wd_mult=0` for bias, lines
  480–485); converged norm `0.0015`, consistent. **But MXNet was archived by the
  ASF in 2023** and is presented here as a co-equal, even leading, tab (it owns
  the first `:begin_tab:` note). Per the chapter-wide currency policy, de-emphasize
  or drop. At minimum it should not lead the concise-API exposition.

### Cross-framework consistency & d2l conventions
- **The four concise norms should agree and don't** (pt 0.013 / jax 0.0017 / tf
  0.0021 / mxnet 0.0015). Root cause is *whether weight decay is applied as a
  loss penalty with a $1/2$ (tf, scratch) or on the gradient (pt, jax/optax) and
  whether `MSELoss`'s missing $1/2$ is compensated*. TensorFlow compensates
  (`wd/2`); PyTorch does not. This is fixable and is the single most important
  code action. Three of four converging to ~0.0015–0.0021 confirms the target;
  PyTorch is the outlier.
- Imports: one per-framework imports cell at top (lines 49–80), `%matplotlib inline`
  each — clean, follows convention. Stable `#cell-id`s present throughout.
  `l2_penalty` is correctly framework-agnostic (`d2l.reduce_sum`).
- The `lambd: int = 0` annotation in the JAX scratch model (line 344) should be
  `float` for type honesty (it's used with 3 here, but $\lambda$ is real); trivial.

## 4. Prioritized change list

| # | Sev | Dimension | Change (specific, actionable) | Effort |
|---|-----|-----------|-------------------------------|--------|
| 1 | P0  | code/correctness | Reconcile PyTorch concise `weight_decay` so its converged norm matches the scratch/other-framework result (~0.0015), **or** add a sentence teaching why gradient-side `weight_decay` + `MSELoss` (no $1/2$) ≠ the scratch $\frac{\lambda}{2}\|\mathbf w\|^2$ penalty. Current "plot looks similar to … from scratch" is false for PyTorch (0.013 vs 0.0016). | M |
| 2 | P1  | teaching/figure | Add a committed ridge-geometry SVG (elliptical loss contours kissing the $\|\mathbf w\|^2\le t$ ball; optional $\ell_1$-diamond panel) via the `mdl-figure` workflow; reference by `:numref:`. | L |
| 3 | P1  | coverage/currency | Replace the vague lines 215–219 hedge with a named decoupled-weight-decay sentence (AdamW; :citet:`loshchilov2019decoupled`) + forward pointer `:numref:`sec_adam``; half-sentence noting the JAX `optax.add_decayed_weights` is decoupled decay. | S |
| 4 | P1  | coverage | Add the MLE/MAP paragraph after line 219 (Gaussian prior ⇒ ridge ⇒ MAP), back-pointing to the MLE section in `linear-regression.md` and `chap_regression` resources; scaffolds Ex 6. | S |
| 5 | P1  | prose/correctness | Fix malformed binomial line 43 → `\binom{k-1+d}{k-1}`; fix Ex 1 "accuracy" → "loss". | S |
| 6 | P2  | code | Silence the PyTorch `requires_grad` UserWarning (`.detach()` / `torch.no_grad()`) in the L2-norm prints. | S |
| 7 | P2  | coverage/trim | Trim the monomial preamble (lines 21–47) to ~5 lines; cut or shrink the RKHS aside (lines 581–590) to a forward pointer. | S |
| 8 | P2  | currency | De-emphasize the MXNet tab (archived 2023); don't let it lead the concise-API note. | M |
| 9 | P2  | exercises | Add the closed-form-uniqueness exercise and the Adam-decoupling micro-experiment; widen Ex 6 to "show prior variance $=1/\lambda$." | S |

## 5. Keep — what is already excellent (do not lose this)
- The **$f=0$-is-simplest** opening intuition (lines 92–98) and the honest "no
  single right answer / functional analysis" aside — exactly the
  intuition-before-formalism the bar wants.
- The **shrinkage-update derivation** (lines 190–210): the $(1-\eta\lambda)$
  factor and the "why it's called *weight decay*" explanation are textbook-clean.
- The **scratch → concise pairing** with the $\lambda=0$ vs $\lambda=3$ contrast;
  the loss curves *show* the regularization payoff (training up, validation down).
- The **TensorFlow `l2(wd/2)` convention-matching comment** (lines 508–513) and
  the **JAX `optax.chain`/masked-decay** implementation — both are the modern,
  correct idiom and worth preserving as written.
- The **explicit handoff from `generalization.md`** ("your first practical
  regularization technique") — the chapter arc is well wired; don't disturb it.
- The **per-parameter-group bias-exclusion** shown in all four frameworks — a
  genuine best-practice detail most intros omit.
