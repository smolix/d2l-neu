# Slide outline — §4.3 The Base Classification Model (`classification.md`)

**Status: fresh, no blockers.** Code is class-definition only, so the notebooks
have **no displayed outputs in any framework** — that is expected (no expression
cells), **not** a staleness flag. The prose was clearly given a quality pass
(the scores→loss/accuracy fork, "two numbers and why", top-k exercises) and the
section already ships the `fig_mdl-clf-loss-accuracy` figure. The existing
`<!-- slides -->` block is decent but legacy-shaped (8 flat `.slide`s, no
cover/divider/kicker, the figure is not used on a slide) — rebuild to the bar
and **wire in the existing figure**.

**Reuse the existing chapter figure:** `img/mdl-clf-loss-accuracy.svg`
(`fig_mdl-clf-loss-accuracy`). It is exactly the deck's central idea: one forward
pass → logits $\mathbf{o}$, top branch softmax→probabilities→cross-entropy
(differentiable, drives SGD), bottom branch argmax→hard decision→accuracy
(discrete). The figure's numbers are the softmax/CE of logits $(1.0,2.2,0.3)$
for true class $y=1$ — keep any new figure numerically consistent with this.
(It is a wide matplotlib SVG, 790×317 — show full-bleed `.fig .big`, not in a
narrow column.) Reference as `![](../img/mdl-clf-loss-accuracy.svg)`.

**Per-framework cell presence — one mxnet-only cell:**
- `classification-accuracy-2` exists **only in mxnet** (the `get_scratch_params`
  / `parameters` Gluon fallback — needed because Gluon's `collect_params` misses
  bare-`np.ndarray` scratch weights; the prose explicitly says the other three
  frameworks don't need it). **This is genuinely MXNet-only plumbing, NOT a
  port-before-other-decks gap.** Show it on the MXNet deck only, scoped
  `only="mxnet"` (or simply omit — it is library glue, low teaching value on a
  slide). The other 4 cells are shared 4-way `#@tab` sets.

**No cells to port.** No other per-framework omissions.

**Per-framework framing:** the JAX `Classifier` differs structurally
(`training_step` returns aux data; `validation_step`/`accuracy` take
`params`+`state` because Flax modules are stateless; `accuracy` is
`@jax.jit`-decorated). This is **code-level divergence the `#@tab` swap already
handles** for `…the-classifier-class-1` and `…accuracy-1` — the *concept*
(report loss + accuracy; accuracy = argmax-eq-label-mean) is identical, so **no
`only=`/`except=` framing split is needed**, only the automatic code swap. Note
in captions that JAX threads `params`/`state` explicitly.

**Diagrams:** **no new engine diagram required** — the existing
`fig_mdl-clf-loss-accuracy` carries the section's one structural idea. (Optional
nicety: a tiny `argmax`-on-a-score-vector chip figure, but the existing figure
already shows the bottom branch.)

---

## Outline

### 0. Cover
- `::: {.cover}` — kicker `Dive into Deep Learning · §4.3`.
- "One forward pass, **two numbers**: the loss we optimize and the accuracy we
  report — collected once in a shared `Classifier` base."

### 1. Why / what opener — "Why a classifier needs two numbers"
- One idea: a forward pass gives scores $\mathbf{o}\in\mathbb{R}^q$; from there
  the picture **forks** — softmax→cross-entropy (smooth, optimizable) vs
  argmax→accuracy (discrete, reportable). We optimize the loss, we report the
  accuracy.
- `.fig .big`, full-bleed: **`![](../img/mdl-clf-loss-accuracy.svg)`**
  (`fig_mdl-clf-loss-accuracy`). This *is* the opener.
- `.d2l-note`: accuracy's gradient is zero almost everywhere (a tiny score nudge
  rarely flips the argmax) ⇒ it can't be the training objective.

### 2. The shared `Classifier` base
- One idea: every classifier (linear softmax now → deep CNNs later) shares a
  validation step that logs loss **and** accuracy, and a default optimizer; put
  them once on a `Classifier(d2l.Module)` base.
- 2-col or bullets: subclasses supply only `forward` (+ a custom `loss` if not
  plain cross-entropy) and inherit train/eval. Pair with
  `@classification-the-base-classification-model` (imports) — or `@-` it.
- `.d2l-note`: same payoff that motivated `Module` in §3 (OO design).

### 3. The `Classifier` class
- One idea: `validation_step` reports both loss and accuracy per validation
  batch (averaged over the epoch).
- `@classification-the-classifier-class-1`. Caption: JAX additionally overrides
  `training_step` and threads `params`/`state` because Flax modules are
  stateless (auto-swapped by framework). One slide, no scoping.

### 4. A default optimizer hook
- One idea: install plain minibatch SGD as the default `configure_optimizers` on
  `Module` so no subclass repeats it (subclasses can override later).
- `@classification-the-classifier-class-2`. Short; fine as one slide.
- *(MXNet deck only, optional)* a `only="mxnet"` continuation showing
  `@classification-accuracy-2` (`get_scratch_params`) with a one-line caption:
  "MXNet needs a fallback to find bare-array scratch params." Low priority —
  omit if it crowds the deck.

### 5. Accuracy in one line
- One idea: argmax along the class axis, compare to the label elementwise,
  average → fraction correct.
- `@classification-accuracy-1`. Caption: "`argmax → ==y → mean`." Mention the
  dtype-cast detail in a half-line. (JAX variant threads `params`/`state` and is
  jitted — auto-swapped.)
- Optional `.d2l-note`: Gmail-folder analogy (must commit to one folder even if
  it scores probabilities internally) — the "hard decision" intuition.

### 6. Why report both? (loss vs accuracy)
- One idea: two models can tie on accuracy yet differ in confidence; only the
  loss separates "confidently right" from "barely right". Disagreement between
  the two curves is diagnostic, not a bug.
- Full-width bullets (or 2-col with the loss-accuracy figure reused small):
  loss = calibration/optimization signal; accuracy = hard-decision quality;
  watch both. Fragment to reveal the "disagreement is diagnostic" point.
- Tie to the §4.3 exercise (two classifiers, 0.91 vs 0.51 avg confidence at the
  same 90% accuracy) as the concrete example.

### 7. Recap
- `Classifier(d2l.Module)` = validation step logging **loss + accuracy** + a
  default SGD optimizer.
- Accuracy = `argmax → ==y → mean`; discrete, so it's a metric, not a loss.
- Every classifier in the rest of the book subclasses this and supplies only its
  `forward`.

---

## Diagram inventory
- **Reuse only:** `img/mdl-clf-loss-accuracy.svg` (`fig_mdl-clf-loss-accuracy`),
  on slides 1 and (optionally, smaller) 6.
- **No new engine diagram needed.** (If the author wants the section to have a
  matching engine-style figure, a `classification-scores-fork` redraw of the
  loss/accuracy branch in the slide palette would unify the look with §4.1's
  pipeline figure — but the existing matplotlib figure already teaches it. Flag
  as optional polish, deferred.)

## Per-framework summary
- **`only="mxnet"` (optional):** `classification-accuracy-2`
  (`get_scratch_params`) — MXNet-only library glue; show only on the MXNet deck
  or omit. **Not a gap to port.**
- **No `only=`/`except=` framing splits** otherwise — JAX's structural
  differences are handled by the automatic `#@tab` code swap; the concept is
  identical across frameworks.
- **No cells to port.**
- **No staleness blockers** (empty outputs are expected for definition-only
  cells).
