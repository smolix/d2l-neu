# Slide outline — §4.1 Softmax Regression (`softmax-regression.md`)

**Status: diagram-driven, no code.** This section has **zero `.python .input`
cells** (confirmed: `grep -c` = 0) and **no executed notebook in any
framework** — it is pure theory (the linear model for classification, the
softmax, the cross-entropy loss, its gradient, the info-theory tie-in). So the
deck carries **no `@id` cells at all**; it is built entirely from diagrams +
math, the way the calculus deck is. This is the model to follow: calculus
carries the chapter on ~6 geometric figures with incidental code; here we carry
it on diagrams with *no* code.

**No staleness blockers.** Content is current and well-written (the §4.1 prose
was clearly already given a quality pass: temperature scaling, exponential-family
framing, log-partition convexity, the sigmoid special case, the
"cross-entropy = code length" duality are all present). Nothing to flag for
rewrite.

**Per-framework:** N/A — no code, so **one shared deck, zero `only=`/`except=`
scoping**. Identical across all four frameworks.

**Existing book figure to reuse:** `img/softmaxreg.svg`
(`fig_softmaxreg`, the single-layer fully-connected network, 4 inputs → 3
outputs). Reference it as a plain `![](../img/softmaxreg.svg)` inside a slide
(it is a book SVG, not an `img/auto` engine diagram).

**New engine diagrams needed (all NEW — there is no `diagrams/softmax-regression.mjs`):**
small module of ~5 figures. These are the main new work for this deck.

---

## Outline

### 0. Cover
- `::: {.cover}` — kicker `Dive into Deep Learning · §4.1`.
- One line: "From *how much?* to *which class?* — turning scores into
  probabilities with the **softmax**, and scoring them with **cross-entropy**."

### 1. Why / what opener — "Classification, not regression"
- One idea: regression answers *how much?*; classification answers *which
  category?* We want a model with **one output per class**, read as a
  probability distribution.
- 2-col. Left: the bullet contrast (spam/inbox, cat/chicken/dog, which movie).
  Right: **NEW diagram `softmax-regression-onehot`** — the one-hot encoding:
  three label vectors $(1,0,0),(0,1,0),(0,0,1)$ drawn as 3 stacked 3-cell
  column vectors colored by the hot entry (cat/chicken/dog). Grounds "label =
  vector" before any math.
- `.d2l-note`: classification splits into **hard** assignment (pick one) vs
  **soft** assignment (probability per class); we model soft even when we only
  need hard.

### 2. Divider 01 — "The Linear Model"

### 3. One affine function per class
- One idea: $q$ classes ⇒ $q$ affine functions; with 4 features and 3 classes,
  12 weights + 3 biases.
- 2-col. Left: the scalar expansion $o_1,o_2,o_3 = \sum_i x_i w_{\cdot i}+b$
  (show the 3-line system, trimmed) and the vector form
  $\mathbf{o}=\mathbf{W}\mathbf{x}+\mathbf{b}$, $\mathbf{W}\in\mathbb{R}^{3\times4}$.
- Right (`.fig .big`): **`![](../img/softmaxreg.svg)`** — the single-layer
  fully-connected network (`fig_softmaxreg`). This is the structural picture of
  the whole section.
- `.d2l-note`: every output depends on every input ⇒ a **fully connected
  layer**; same single-layer shape as linear regression, just vector-valued.

### 4. Divider 02 — "The Softmax"

### 5. Why raw outputs aren't probabilities
- One idea: the affine outputs $\mathbf{o}$ are unconstrained — they need not be
  nonnegative and need not sum to 1, so they cannot be read as probabilities.
- 2-col. Left: the two failure bullets ($\sum o_i\neq1$; $o_i$ can be negative /
  exceed 1; the "mansion" outlier). Right: **NEW diagram
  `softmax-regression-squish`** — a number line / bar pair showing raw logits
  (mixed sign, e.g. $(2.0,-0.5,1.0)$) on the left and the same after softmax
  (three bars in $[0,1]$ summing to 1) on the right, an arrow "squish + normalize"
  between them. The *picture* of what softmax buys us.

### 6. The softmax: exponentiate, then normalize
- One idea: $\hat y_i = \exp(o_i)/\sum_j\exp(o_j)$ — exp makes it positive,
  dividing by the sum makes it a distribution.
- Full-width math slide. Show :eqlabel:`eq_softmax_y_and_o`. Fragment (`. . .`):
  argmax is preserved — $\arg\max_j \hat y_j = \arg\max_j o_j$, so you don't need
  the softmax to *decide*, only to get calibrated-looking scores.
- `.d2l-note`: monotone in each $o_i$; ordering preserved.

### 7. The softmax pipeline (one example)
- One idea: trace a single example end to end: features → logits → exp →
  normalize → probabilities → argmax → predicted class.
- `.fig .big`, full-bleed: **NEW diagram `softmax-regression-pipeline`** — a
  left-to-right flow: $\mathbf{x}$ (4 cells) → $[\mathbf{W}\cdot+\mathbf{b}]$ block
  → $\mathbf{o}$ (3 cells) → $\exp$ → $\sum$/normalize → $\hat{\mathbf{y}}$ (3
  bars summing to 1) → argmax → "dog". Use the same logits the loss-accuracy
  figure uses, $(1.0, 2.2, 0.3)$, so this deck agrees numerically with §4.3's
  `fig_mdl-clf-loss-accuracy`. (This is the conceptual heart of the deck.)

### 8. Translation invariance & the sigmoid special case
- One idea: only **differences** of logits matter — adding $c$ to every logit
  leaves $\hat{\mathbf{y}}$ unchanged (the "one redundant parameter"); the
  two-class case collapses to the logistic sigmoid.
- 2-col. Left: $\hat y_1=\sigma(o_1-o_2)$, :eqlabel:`eq_softmax_to_sigmoid`, and
  the "pin $o_q\equiv0$" remark. Right: **NEW diagram
  `softmax-regression-sigmoid`** — the logistic curve $\sigma(o)$ on axes, with
  the two-class softmax annotated as "sigmoid of the logit gap $o=o_1-o_2$".
  (Geometric payoff; ties binary logistic regression to softmax.)
- Optional `.d2l-note`: Boltzmann/Gibbs origin (energy = error), temperature $T$
  — keep to one sentence; it returns in the calibration note later.

### 9. Vectorize over a minibatch
- One idea: stack $n$ examples in rows; $\mathbf{O}=\mathbf{X}\mathbf{W}+\mathbf{b}$
  is one matrix–matrix product, softmax applied **rowwise**.
- 2-col. Left: :eqlabel:`eq_minibatch_softmax_reg`, the shapes
  $\mathbf{X}\in\mathbb{R}^{n\times d}$, $\mathbf{W}\in\mathbb{R}^{d\times q}$
  (note the transpose vs the per-example $\mathbf{W}$). Right: **NEW diagram
  `softmax-regression-minibatch`** — block-matrix picture $\mathbf{X}\,(n\times d)
  \times \mathbf{W}\,(d\times q) = \mathbf{O}\,(n\times q)$, then a per-row
  softmax brace on $\mathbf{O}$. (Reuses the grid/block engine vocabulary;
  same idea as the linear-algebra matmul figures.)
- `.d2l-note` (`.warn`): naive exp/log overflows — the stable fix
  (subtract $\max_k o_k$, fuse into log-sum-exp) is derived in §4.5; forward
  pointer only.

### 10. Divider 03 — "The Loss"

### 11. Maximum likelihood → cross-entropy
- One idea: maximize the probability the model assigns to the true labels;
  equivalently minimize the negative log-likelihood, which for one-hot $\mathbf{y}$
  is $l(\mathbf{y},\hat{\mathbf{y}})=-\sum_j y_j\log\hat y_j=-\log\hat y_{\text{true}}$.
- Full-width math. Show the product → sum-of-logs step and
  :eqlabel:`eq_l_cross_entropy`. Fragment: because $\mathbf{y}$ is one-hot the
  sum is a single term — the negative log-prob of the correct class.
- `.d2l-note`: $l\ge0$, and $=0$ only at certainty (unreachable for finite
  weights); over-confident wrong answers cost $-\log 0=\infty$.

### 12. The gradient is "prediction − observation"
- One idea: write the loss on logits as $g(\mathbf{o})-\mathbf{y}^\top\mathbf{o}$
  where $g$ is the log-partition function; then
  $\partial_{o_j} l=\mathrm{softmax}(\mathbf{o})_j-y_j$ — the same residual shape
  as least squares.
- 2-col. Left: the $g(\mathbf{o})=\log\sum_k\exp o_k$ definition and the gradient
  identity (softmax = gradient of log-partition). Right: **NEW diagram
  `softmax-regression-gradient`** — the residual: a probability bar vector
  $\hat{\mathbf{y}}$ minus the one-hot $\mathbf{y}$ → a signed residual bar
  $\hat{\mathbf{y}}-\mathbf{y}$ (the gradient), echoing the regression
  "$\hat y - y$" picture. *Or* reuse `fig_mdl-clf-loss-accuracy`
  (`img/mdl-clf-loss-accuracy.svg`) here — but that figure's job is the
  loss-vs-accuracy fork (better placed in §4.3); a dedicated residual figure
  teaches the gradient idea more directly. **Author's call: new `gradient`
  figure vs reuse loss-accuracy.**
- `.d2l-note`: exponential-family fact — log-likelihood gradient is always this
  residual; Hessian of $g$ = covariance of softmax ⇒ convex in $\mathbf{o}$.

### 13. Why it's called "cross-entropy"
- One idea: entropy = expected surprisal of $P$; cross-entropy $H(P,Q)$ = the
  cost of coding $P$'s draws under a wrong model $Q$, minimized at $Q=P$. Our
  loss is exactly $H(\mathbf{y},\hat{\mathbf{y}})$.
- 2-col. Left: $H[P]=\sum_j -P(j)\log P(j)$, $H(P,Q)=\sum_j -P(j)\log Q(j)$; the
  MLE ↔ code-length duality in one line. Right: **NEW diagram
  `softmax-regression-crossentropy`** — surprisal curve $-\log q$ with the
  true-class probability marked, "few bits when confident & right, many bits when
  confident & wrong." (Optional; if cut, fold into a `.d2l-note`.) Forward
  pointer to §26 Information Theory.

### 14. Recap
- Linear model: one affine function per class, $\mathbf{o}=\mathbf{W}\mathbf{x}+\mathbf{b}$.
- Softmax: exp + normalize ⇒ a probability distribution; only logit
  *differences* matter; binary case = sigmoid.
- Cross-entropy = negative log-likelihood = $H(\mathbf{y},\hat{\mathbf{y}})$;
  its gradient is the clean residual $\mathrm{softmax}(\mathbf{o})-\mathbf{y}$.
- Caveat forward pointers: numerical stability (§4.5), calibration / temperature
  (§4.3 + §26).

---

## Diagram inventory for this deck (all NEW)

Create `diagrams/softmax-regression.mjs`, register in `registry.mjs`:

| id | what it draws | engine reuse |
|---|---|---|
| `softmax-regression-onehot` | three one-hot label column vectors (cat/chicken/dog) | `grid`, `tx` |
| `softmax-regression-squish` | raw logits (mixed sign) → softmax bars in [0,1] summing to 1 | bars via `rc`/`block`, `arrow` |
| `softmax-regression-pipeline` | x → Wx+b → o → exp → normalize → ŷ → argmax → label; logits $(1.0,2.2,0.3)$ | `grid`,`block`,`arrow`,`chip` |
| `softmax-regression-sigmoid` | logistic curve; two-class softmax = σ(logit gap) | axes via `arrow`+`tx`, curve path |
| `softmax-regression-minibatch` | X(n×d)·W(d×q)=O(n×q), rowwise-softmax brace | `grid`,`block` (matmul vocab) |
| `softmax-regression-gradient` | ŷ − y residual bar (gradient) | bars, `arrow` |
| `softmax-regression-crossentropy` *(optional)* | surprisal −log q curve | curve + `tx` |

**Reused, not new:** `img/softmaxreg.svg` (book fig, slide 3);
optionally `img/mdl-clf-loss-accuracy.svg` (slide 12, if author prefers reuse to
a new `gradient` figure).

## Open questions for author
1. Slide 12: dedicated **`softmax-regression-gradient`** residual figure, or
   **reuse `fig_mdl-clf-loss-accuracy`** (which is really the §4.3 loss/accuracy
   fork)? Outline assumes new.
2. Slide 13 cross-entropy figure: keep, or compress to a callout? (Deck is
   already ~14 slides / 7 figures — comfortably calculus-density, but slide 13 is
   the most cuttable.)
3. The Boltzmann/temperature aside (slide 8) and the calibration note: how much
   to keep on slides vs leave to the book? Outline keeps each to one sentence.
