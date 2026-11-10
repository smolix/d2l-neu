# Writing pass: Numerical Stability and Initialization (section 4.4)

Source: `chapter_multilayer-perceptrons/numerical-stability-and-init.md`.
Checker: "OK: only prose changed"; `tools/lint_source.py` clean.

## Words

Prose words 3907 before, 3907 after. The derivation of the gradient factor
(NS-2) and the candour about training (NS-1) added about 280 words; cuts
elsewhere paid for them.

## Before and after

1. Before: "A network with ten hidden layers of width 256, initialized the
   same way, does not learn."
   After: "Consider a network with ten hidden layers of width 256,
   initialized the same way. It is in trouble before the first step of
   training."
2. Before: "Whether a deep network can be trained at all thus depends on the
   scale of its initial weights"
   After: "A word of caution: every measurement is taken on a randomly
   initialized network, and we train none. That a first layer with so small
   a gradient cannot learn at a learning rate the last layer tolerates is an
   argument, which Exercise ... lets you test on networks of two and four
   hidden layers."
3. Before: "To see why the scale of the weights decides whether the gradient
   survives, we write the backward pass as a product with one factor per
   layer."
   After: "Why should the scale of the weights matter so much? To see this,
   let us write the backward pass as a product with one factor per layer."
4. Before: "As for the repeated matrix products of
   :numref:`sec_linear-algebra`, with $\rho = 0.9$ the gradient shrinks
   200-fold over fifty layers"
   After: "We saw the same effect for repeated matrix products in
   :numref:`sec_linear-algebra`. With $\rho = 0.9$ the gradient shrinks
   200-fold over fifty layers"
5. Before: "(:numref:`subsec_xavier` derives this)."
   After: "The word "about" is there for a reason. Independence of the
   weights and the slopes is a rather bold assumption, since the slope
   $\phi'(z_j)$ was computed in the forward pass from the very weights that
   now multiply it. ... We do not prove it; the measurement below is its
   test."
6. Before: "Larger weights do not simply compensate, because they push more
   pre-activations into the tails of the sigmoid"
   After: "Could larger weights not compensate? Unfortunately not, because
   they push more pre-activations into the tails of the sigmoid"
7. Before: "The same computation shows that the variance of the gradient is
   multiplied by $n_\textrm{out}\tau^2$"
   After: "For the backward pass the work is already done. The sum behind
   :eqref:`eq_grad-factor` runs over the $h = n_\textrm{out}$ outputs of the
   layer, and without activation function the slope is 1, so the mean square
   of the gradient is multiplied by $n_\textrm{out}\tau^2$."
8. Before: "the activations at the last layer are about $10^{-19}$, and the
   network is useless before training begins."
   After: "the mean square of the activations at the last layer is about
   $10^{-19}$. Whichever input we present, the last hidden layer answers
   with entries of size about $3 \times 10^{-10}$."

## Repetition removed

- Opening recap of `sec_mlp` (depth and exponentially many units): one
  clause with the reference.
- Sigmoid derivative formula and "made deep sigmoid and tanh networks hard
  to train": owned by `mlp.md` (`subsec_activation-functions`); here only
  the bound 1/4 is used, and the citation is kept.
- The forward factor $h\tau^2/2$ in "Exploding Gradients": one clause with a
  reference to `subsec_he_init`, which derives it.
- "Architectures that keep the scale under control during training": was in
  the opening and at the end of "Activations and Gradients across Depth";
  kept at the end only.
- The sentence that restated :eqref:`eq_grad-product` in words; the
  restatement of the chain rule after :eqref:`eq_grad-step`.
- After the symmetry check: "Random initialization breaks the symmetry, and
  the scale of the random draws comes from the variance argument" (said
  before the check).
- Formulaic closing "We can now choose ..." removed; the closing is the one
  sentence that leads to the next section.
- Summary: the worked numbers $10^{17}$ and $10^{19}$ are no longer repeated
  there.
- "whatever" reduced from five uses to one in the body.

## Review findings

Applied:

- NS-1 (prose part): the section no longer says the network "does not
  learn". The opening and the end of "Activations and Gradients across Depth"
  state that all measurements are on untrained networks and that the
  consequence for training is an argument, with a pointer to the exercise of
  `mlp-implementation.md`.
- NS-2: the backward factor is derived on the page in "Products of
  Jacobians" (entry $g_i = \sum_j w_{ij}\phi'(z_j)g'_j$, cross terms, the
  two independence assumptions, why "about"). The forward pointer is gone,
  and the Xavier subsection now refers back instead of saying "the same
  computation shows".
- NS-4: mean squares throughout; the opening says once that a typical entry
  is the square root.
- NS-5: "Default Initialization" moved after "He Initialization" (no code
  cell is crossed), and each tab note compares its default with
  $2/n_\textrm{in}$. "this book sets the initialization of its models
  explicitly" became "from the next section on our models set He
  initialization explicitly", which I checked in `generalization-deep.md`,
  `dropout.md` and `kaggle-house-price.md`.
- NS-6, NS-7, NS-8 (one sentence kept), NS-9 (dropout aside deleted).
- NS-10: the opening says that the same factor shrinks the signal on the way
  up.
- NS-3, in part: one sentence explains why $\mathbf{W}^{(k)}$ appears
  without a transpose in the single-example form.

Deferred:

- NS-1 (the training experiment): needs a new cell and outputs.
- NS-3 (switch to the batch notation of `sec_backprop`, drop
  $\mathbf{M}^{(k)}$): changes two labelled display equations and exercise
  1, and is a chapter-wide notation decision.
- Structural reorder (measurement, derivation, cure, measurement): would
  move code cells.
- Title change: other files reference the title in their slide tables.
- Exercise remarks (code exercise, LARS name): exercises not touched.

## Factual claims added or changed

- "each multiplication shrinks the mean square of the gradient about
  eightyfold" ($1/0.0128 = 78$), replacing "the magnitude ... about
  ninefold".
- "With weights of standard deviation 1, both grow at a similar rate"
  (factor 128 per layer for activations and gradients).
- The derivation: $\mathbb{E}[g_i^2] = h\tau^2\,\mathbb{E}[\phi'(z)^2]\,\mathbb{E}[(g'_j)^2]$ under
  the stated independence assumptions; the remark that the assumption is
  plausible in a wide layer because $w_{ij}$ is one of $h$ weights entering
  $z_j$.
- "The independence assumption has passed its test, at least for random ReLU
  networks of this width" (all four stores agree with $0.0128^9$ and $128^9$
  within a factor of 1.5).
- "entries of size about $3 \times 10^{-10}$" (square root of the stored
  $8 \times 10^{-20}$ to $1.6 \times 10^{-19}$).
- Default initializers against He: MXNet $0.07^2/3 \approx 0.0016$, equal to
  $2/n_\textrm{in}$ only at about 1,200 inputs; PyTorch one sixth; Keras
  (equal widths) and Flax one half. Arithmetic on the facts already in the
  tab notes; the library defaults themselves were not re-verified.
- "None of the four frameworks defaults to it".
- Removed: "the updates overshoot so far that the loss grows instead of
  falling" (now "overshoot"); "the network is useless before training
  begins"; the He-paper count reconciliation.

## Slides that now disagree with the body

- Opening slide: "The initial scale decides whether a deep network trains at
  all" and "The first layers either never move or overshoot": the body now
  presents this as an argument, not a result.
- Symmetry slide: "Dropout, later in this chapter, also breaks ties": the
  body no longer mentions dropout.
- Vanishing slide: "Vanishing gradients long made deep sigmoid and tanh
  networks hard to train": the body keeps only the citation.
- No disagreement on order: the deck already places "The Default
  Initialization Is Not He" after the He slides, as the body now does.
