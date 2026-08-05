# Chapter Overview — chapter_multilayer-perceptrons

Best external sources by a wide margin: Simon Prince's "Understanding Deep
Learning" (verified against the book's own text) has numbered end-of-chapter
problems that map almost one-to-one onto this chapter's math-heavy sections —
ch.3/4 on piecewise-linear regions and ReLU homogeneity (mlp.md), ch.7 on
gradients/computational graphs/initialization (backprop.md,
numerical-stability-and-init.md), and ch.8 on capacity/double descent
(generalization-deep.md). Michael Nielsen's NNDL (ch.1-4, verified via direct
fetch) is the next-best match, especially for backprop derivations and early
stopping. For kaggle-house-price.md, Kaggle Learn's "Intermediate Machine
Learning" course is an unusually exact match: same Ames-derived dataset,
same missing-value/categorical/cross-validation tasks (verified via the
Kaggle/learntools GitHub source). CS231n and EECS 498-007 assignments are
strong for the from-scratch/dropout implementation sections. Two explicit
gaps: MIT 6.390's problem sets sit behind an authenticated CAT-SOOP/Canvas
wall and could not be verified at all; and double-descent/grokking
reproduction has essentially no coursework tradition anywhere (both are
compute-heavy and research-adjacent), which is itself a finding, not a
failure — this book's own exercises 6-7 in generalization-deep.md are already
the rigorous, hands-on version of that material. Existing exercises fare
well throughout: 52 exercises reviewed across 7 sections, 40 kept, 12
rewritten, **zero dropped** — the chapter's only recurring defect is "Can
you/we...?" filler phrasing, concentrated overwhelmingly in
numerical-stability-and-init.md (why that section alone needed 4 of 6 items
rewritten, versus 1-2 elsewhere).

---

## chapter_multilayer-perceptrons/mlp.md — Multilayer Perceptrons

**Topic:** Why hidden layers need a nonlinearity, illustrated by a hand-built
XOR network, and formalized by the universal approximation theorem and its
depth-vs-width piecewise-linear refinement.

**Current exercises:** 9; disposition: keep 7, rewrite 2, drop 0 — an
internal style review found zero formatting or clarity defects in items
1-7 (this is the cleanest file in the whole 22-file group); only items 8
and 9 lack a checkable deliverable, so only those two are rewritten.

**External sources found:**
- Michael Nielsen, *Neural Networks and Deep Learning*, ch. 4 problems
  (2015; verified by direct fetch) — asks the reader to prove universality
  holds with a *single* hidden layer (the book's own construction uses two),
  and separately whether ReLUs and linear units are/aren't universal — the
  exact family of representational questions this section raises about
  width and nonlinearity choice — http://neuralnetworksanddeeplearning.com/chap4.html
- Simon Prince, *Understanding Deep Learning*, Problem 3.5 (2023; verified
  against book text) — prove ReLU's non-negative homogeneity,
  ReLU(α·z) = α·ReLU(z) for α>0 — https://udlbook.github.io/udlbook/
- Simon Prince, *Understanding Deep Learning*, Problem 3.18 (verified) — what
  is the maximum number of linear regions a shallow network can create as a
  function of hidden-unit count — the exact "count the pieces" question this
  section demonstrates numerically with `count_pieces` —
  https://udlbook.github.io/udlbook/
- Simon Prince, *Understanding Deep Learning*, Problems 4.8-4.9 (verified) —
  given three ReLU units' slopes and joint positions, find weights producing
  an oscillating piecewise-linear function, then ask how many pieces result
  from composing the network with itself — directly the "depth folds the
  graph" argument this section makes intuitively in exercise 6 —
  https://udlbook.github.io/udlbook/
- CMU 11-785, Intro to Deep Learning, HW1P1 (per course syllabus
  descriptions) — implements MLP activation functions and their derivatives
  from scratch as part of a personal deep-learning library, the same
  activation-derivative territory as this section's exercises 3-4 —
  https://deeplearning.cs.cmu.edu
- Stanford CS231n course notes ("Neural Networks Part 2," verified by direct
  fetch) discuss dead-ReLU and saturation narratively but contain **no
  numbered exercises at all** on this page; the specific pReLU/Swish
  derivative exercises here have a thin tradition outside textbook problem
  sets — Nielsen and Prince, not course notes, are the primary sources.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Batch-Wise Nonlinearity Pitfalls.** Consider a
   nonlinearity that operates jointly across a minibatch (such as batch
   normalization) applied during training with minibatch size n, versus at
   inference time on a single example. Identify at least two specific
   things that break when the same code path is naively applied to a batch
   of size 1 (not just a general impression) — for example, an ill-defined
   statistic at n=1, and a train/test mismatch in what gets computed.
   *Provenance:* original.
1. [short-code] **Sigmoid Depth Underflow.** Extend the section's own
   sigmoid/gradient plot: for a chain of L identical sigmoid layers
   evaluated near the origin, find the depth L at which the compounded
   gradient factor $0.25^L$ first drops below float32's smallest normal
   positive value ($\approx 1.2\times10^{-38}$), by direct computation or
   repeated multiplication. Compare against float64, and plot compounded
   gradient magnitude vs. depth for both dtypes on a log scale.
   *Provenance:* original (numeric target inspired by the section's own
   "$0.25^{10}\approx 10^{-6}$" remark).
1. [conceptual] **Single-Layer Universal Approximation.** Nielsen's book
   proves universal approximation using *two* hidden layers (one to
   localize a "bump," one to combine bumps) and leaves single-hidden-layer
   sufficiency as an open problem. Sketch how to collapse the two-hidden-
   layer bump construction into one hidden layer for the two-input case,
   following this section's own hinge-construction idea extended to 2-D.
   *Provenance:* adapted from Nielsen, NNDL ch. 4 problem (overlap: high;
   cite on adoption).
1. [conceptual] **ReLU Homogeneity and Folding.** Prove ReLU's non-negative
   homogeneity, $\text{ReLU}(\alpha z) = \alpha\,\text{ReLU}(z)$ for
   $\alpha>0$. Then, for one specific 1-D example, use this property to
   show exactly why composing a second ReLU hidden layer *folds* an
   existing piecewise-linear function's graph rather than merely adding a
   new joint — an exact version of exercise 6's "roughly doubles" claim.
   *Provenance:* adapted from Prince, UDL Problem 3.5/4.8 (overlap: high;
   cite on adoption).
1. [short-code] **Exact Region Count.** For a hand-chosen (non-random)
   weight setting of a 1-input, 1-output, single-hidden-layer ReLU network
   with $D=4$ hidden units, compute the exact number of linear pieces
   analytically from the hidden units' joints ($-b_k/w_k$). Verify your
   hand count matches `count_pieces`'s output on that same weight setting.
   *Provenance:* adapted from Prince, UDL Problem 3.2/3.13 (overlap: med).

---

## chapter_multilayer-perceptrons/mlp-implementation.md — Implementation of Multilayer Perceptrons

**Topic:** Building the same one-hidden-layer Fashion-MNIST MLP from scratch
and via framework layers, then probing its hyperparameter sensitivity as a
bridge to the chapter's next three sections.

**Current exercises:** 9; disposition: keep 8, rewrite 1, drop 0 — one of
the strongest sets in the chapter; only the speed-comparison exercise (6)
lacks a concrete benchmark specification.

**External sources found:**
- Stanford CS231n, Assignment 1, `two_layer_net.ipynb` (verified via public
  mirror; official assignment index) — implements a two-layer network's
  loss/train/predict methods, numerically gradient-checks the backward
  pass, and tunes hyperparameters via a validation-accuracy sweep, the same
  "build it, then sweep against a held-out accuracy curve" pattern as this
  section's exercises 1/4/5 — https://cs231n.github.io/assignments2023/assignment1/
- Michigan EECS 498-007/598-005 (Justin Johnson), Assignment 2, Q2
  "Two-layer Neural Network" (Fall 2020; verified by direct fetch) — walks
  through implementing a two-layer classifier with vectorized gradients
  checked against a naive implementation and numeric gradient checking —
  https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment2.html
- CS231n, Assignment 2, `FullyConnectedNets.ipynb`, "Inline Question 1"
  (verified via public mirror) — after hand-tuning only learning rate and
  weight-initialization scale to overfit 50 examples with 3-layer and
  5-layer nets, asks which depth is more sensitive to initialization scale
  and why — nearly identical in spirit to this section's exercise 2 (a
  deeper from-scratch net trains *worse* under fixed $\sigma=0.01$) and
  exercise 9 (three init scales) —
  https://github.com/mantasu/cs231n/blob/master/assignment2/FullyConnectedNets.ipynb
  (mirror; official assignment at cs231n.github.io/assignments2023/assignment2/)
- CMU 11-785, HW1P1/HW1P2 (per course materials) — HW1P1 implements MLP
  activations, loss, and batch normalization as a from-scratch library
  exercise; HW1P2 applies the same architecture to a Kaggle-hosted speech
  classification task, pairing "build it" with "make it work on a real,
  harder dataset" much like this section's from-scratch/concise split —
  https://deeplearning.cs.cmu.edu
- No good external tradition found for exercise 7's specific systems
  question (tensor-matmul throughput at aligned vs. misaligned dimensions,
  memory bus width) — this is a systems/hardware-benchmarking question,
  rarely assigned in ML courses even when the underlying architecture
  concept (accelerator-friendly tensor shapes) is discussed.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Benchmarking a Concrete Workload.** Fix a specific
   "challenging problem": train the from-scratch and concise MLPs at three
   widths (e.g., 256, 1024, 4096 hidden units) for 3 epochs each on
   Fashion-MNIST, recording wall-clock seconds per epoch for each
   implementation at each width on your own hardware. Plot the
   from-scratch/concise time ratio against width, and state whether it
   grows, shrinks, or stays flat, with a hypothesis why.
   *Provenance:* original (rewrite of exercise 6; adds the concrete
   benchmark spec it lacked).
1. [conceptual] **Depth vs. Width Sensitivity.** Following exercise 2's
   observation that a deeper from-scratch network (same $\sigma=0.01$) can
   train worse, predict in writing, before running anything: between a
   2-hidden-layer and a 4-hidden-layer version of the from-scratch MLP
   (same total width budget), which do you expect to be more sensitive to
   the initialization scale, and why? Then verify by sweeping $\sigma$ over
   {0.001, 0.003, 0.01, 0.03, 0.1} for both depths and reporting the range
   of $\sigma$ that reaches at least 70% training accuracy within 5 epochs
   for each.
   *Provenance:* adapted from CS231n, Assignment 2, `FullyConnectedNets.ipynb`
   Inline Question 1 (overlap: med; cite on adoption).
1. [short-code] **Numeric Gradient Check.** Implement a numeric gradient
   checker (central-difference approximation) for the from-scratch MLP's
   $\mathbf{W}^{(1)}$, following the debugging pattern used in comparable
   course assignments. Compare against autograd's gradient for a single
   minibatch and report the maximum relative error; a correct
   implementation should be below 1e-5 in float64.
   *Provenance:* adapted from the CS231n/EECS 498-007 gradient-checking
   pattern (overlap: med).
1. [extended] **Learning-Rate/Width Interaction Map.** Train the
   from-scratch MLP across a small grid of learning rate × hidden width
   (e.g., lr in {0.01, 0.03, 0.1, 0.3} × width in {32, 64, 128, 256}) for a
   fixed 10 epochs each, and produce a heatmap of final validation
   accuracy. Does the best learning rate shift systematically with width?
   Give one sentence of intuition for the direction of any shift observed.
   *Provenance:* original (extends exercise 5's joint-hyperparameter theme
   into a concrete, plottable grid).
1. [conceptual] **Parameter Count vs. Capacity.** Give a closed-form
   expression, in terms of `num_hiddens` $h$, for the total parameter
   count of the one-hidden-layer Fashion-MNIST MLP (784→$h$→10). Using
   this expression, find the $h$ at which the model has as many parameters
   as there are training examples (60,000), and state whether this $h$ is
   smaller or larger than the 256 used in the section.
   *Provenance:* original.

---

## chapter_multilayer-perceptrons/backprop.md — Forward Propagation, Backward Propagation, and Computational Graphs

**Topic:** Deriving forward/backward propagation and the computational
graph for a one-hidden-layer MLP with weight decay, worked numerically and
cross-checked against each framework's autograd.

**Current exercises:** 6; disposition: keep 4, rewrite 2, drop 0 — item
5(a) uses bare filler phrasing with no concrete deliverable, and item 6 is
the section's excellent capstone autograd-engine exercise, kept at full
difficulty, but its subpart (d) has a real clarity bug (six Greek-letter
variables never defined anywhere), which the rewrite fixes without
weakening the exercise.

**External sources found:**
- Michael Nielsen, NNDL ch. 2, "Problem: Fully matrix-based approach to
  backpropagation over a mini-batch" (verified by direct fetch) — asks the
  reader to reformulate per-example backprop equations to operate on an
  entire minibatch as matrices at once instead of looping over examples —
  http://neuralnetworksanddeeplearning.com/chap2.html
- Michael Nielsen, NNDL ch. 2, "Exercises: Backpropagation with a single
  modified neuron" and "Backpropagation with linear neurons" (verified) —
  asks how to modify the backprop equations when one neuron's activation
  is changed to an arbitrary $f$, and separately what happens if the
  nonlinearity is removed entirely — the same "adapt the equations to a
  changed activation" move this section's own worked example rests on —
  http://neuralnetworksanddeeplearning.com/chap2.html
- Michael Nielsen, NNDL ch. 2, "Problem: Prove Equations (BP3) and (BP4)"
  (verified) — a from-first-principles derivation of the bias- and
  weight-gradient backprop equations, the exact kind of derivation this
  section's text walks through — http://neuralnetworksanddeeplearning.com/chap2.html
- Simon Prince, *Understanding Deep Learning*, Problems 7.12-7.13 (verified
  against book text) — computes derivatives of a composed function on a
  general acyclic computational graph, first by reverse mode
  ("backpropagation"), then by forward-mode differentiation, and asks for
  a comparison — directly the graph-traversal machinery this section's
  `prod` operator formalizes, and a strong source for its reverse-vs-
  forward-mode aside — https://udlbook.github.io/udlbook/
- Simon Prince, *Understanding Deep Learning*, Problem 7.10 (verified) —
  derive the backward pass for a leaky-ReLU activation, extending the
  standard ReLU backprop derivation to a nonzero-negative-slope variant —
  a close analogue for extending this section's worked ReLU example to the
  pReLU activation introduced elsewhere in the chapter —
  https://udlbook.github.io/udlbook/
- Michigan EECS 498-007/598-005, Assignment 3, Q1 "Fully-Connected Neural
  Network" (Fall 2020; verified by direct fetch) — explicitly requires
  implementing "modular backpropagation" as a chained sequence of
  forward/backward layer functions, the same computational-graph-as-
  composed-functions view this section teaches —
  https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment3.html

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Multi-GPU Partition Strategies.** The computational graph
   for a much larger network no longer fits on one GPU. Name two concrete
   parallelization strategies that address this (e.g., splitting layers
   across devices — model/pipeline parallelism — vs. splitting the batch
   across devices with replicated parameters — data parallelism), and for
   each, state one advantage and one disadvantage relative to simply
   shrinking the minibatch size to fit on a single GPU.
   *Provenance:* original (rewrite of exercise 5; removes filler phrasing,
   adds a concrete deliverable).
1. [extended] **Miniature Autograd Engine.** Build a scalar `Value` class
   (`+`, `*`, `relu`, a topologically-sorted `backward()` that accumulates
   gradients at forks) and reproduce this section's worked example, as in
   the original exercise 6(a)-(c).
    1. For the path-counting subpart, use a concrete, fully-named instance
       in place of the undefined Greek letters: a chain of three named
       inputs $x_1,x_2,x_3$, each feeding all three of three named
       intermediate outputs $y_1,y_2,y_3$, which all feed one loss $L$ (9
       edges from the $x$'s to the $y$'s).
    1. Show that your engine computes each $\partial L/\partial x_i$ in one
       pass touching each of the 9 edges once, that this equals the
       sum-over-paths formula expanded and refactored by hand, and explain
       why this shared computation is what makes reverse mode affordable
       versus enumerating all paths.
   *Provenance:* original (fixes an internal clarity gap; problem
   structure and difficulty otherwise unchanged from the book's own
   exercise 6).
1. [short-code] **Vectorizing Backprop Over a Minibatch.** The section
   derives forward/backward propagation for a single example $\mathbf{x}$.
   Extend the from-scratch worked-example code to operate on a minibatch
   of $N$ examples at once ($\mathbf{X}$ of shape $(N,d)$), using matrix
   rather than vector operations throughout, and verify your batched
   gradients match calling autograd separately on each of the $N$ examples
   and averaging.
   *Provenance:* adapted from Nielsen, NNDL ch. 2, "Fully matrix-based
   approach to backpropagation over a mini-batch" (overlap: high; cite on
   adoption).
1. [conceptual] **Reverse Mode vs. Forward Mode.** For this section's own
   worked example (2 inputs, 2 hidden units, 1 output, no regularization),
   compute the same four gradients using forward-mode differentiation:
   propagate one directional derivative forward per input, rather than
   seeding one gradient at the loss and sweeping backward. Compare the
   number of passes each mode needs, and explain why reverse mode is
   preferred whenever there are many parameters and one scalar loss.
   *Provenance:* adapted from Prince, UDL Problems 7.12-7.13 (overlap:
   high; cite on adoption).
1. [conceptual] **Backprop Through a Modified Activation.** Suppose the
   hidden layer's activation is replaced by an arbitrary differentiable
   function $f$ (not necessarily ReLU). Rewrite the section's five
   backward equations ($\partial J/\partial\mathbf{o}$ through
   $\partial J/\partial\mathbf{W}^{(1)}$) in terms of $f'$ instead of
   ReLU's indicator derivative, and identify exactly one place in the
   derivation that depends on the specific choice of $f$ versus one that
   does not.
   *Provenance:* adapted from Nielsen, NNDL ch. 2, "Backpropagation with a
   single modified neuron" (overlap: high; cite on adoption).

---

## chapter_multilayer-perceptrons/numerical-stability-and-init.md — Numerical Stability and Initialization

**Topic:** Why gradients vanish or explode as a product of per-layer
Jacobians, and how Xavier/He initialization fix this for linear vs. ReLU
layers.

**Current exercises:** 6; disposition: keep 2, rewrite 4, drop 0 — this
section has the chapter's highest concentration of vague "Can you/we...?"
filler phrasing (3 of 6 items, per an internal style review) plus one
underspecified item; items 3 and 4 are exceptional and tie directly to the
section's own code, so they stay untouched. This is the section most in
need of rework in the chapter.

**External sources found:**
- Michael Nielsen, NNDL ch. 3, "Exercise: Verify that the standard
  deviation of $z=\sum w_jx_j+b$ ... is $\sqrt{3/2}$" (verified by direct
  fetch) — the same naive-initialization variance computation this
  section's Xavier derivation generalizes, for an older $1/\sqrt{n}$-scale
  scheme — http://neuralnetworksanddeeplearning.com/chap3.html
- Michael Nielsen, NNDL ch. 3, "Problem: Connecting regularization and the
  improved method of weight initialization" (verified) — asks for a
  heuristic argument connecting L2 regularization (weight decay) to a
  smaller-variance weight-init scheme, i.e., that scaled init can act like
  an implicit early regularizer — a natural bridge between this section
  and the chapter's later weight-decay/dropout material —
  http://neuralnetworksanddeeplearning.com/chap3.html
- Simon Prince, *Understanding Deep Learning*, Problems 7.14-7.15
  (verified against book text) — Problem 7.14 proves the same "ReLU halves
  the second moment of a zero-mean symmetric variable" fact this section
  derives, confirming this section's own kept exercise 3 is already
  well-aligned with the external tradition rather than needing external
  replacement; Problem 7.15 asks what happens if all weights and biases
  are initialized to zero, the same symmetry-breaking question as this
  section's exercise 1 — https://udlbook.github.io/udlbook/
- Simon Prince, *Understanding Deep Learning*, Problems 3.5/4.3 (verified)
  — proves and then applies ReLU's non-negative-homogeneity property,
  exactly the identity this section's own depth-sweep code exploits to
  renormalize activations without approximation — a good source for
  making that renormalization trick an explicit, provable exercise rather
  than an unexplained code comment — https://udlbook.github.io/udlbook/
- Stanford CS231n course notes, "Neural Networks Part 2: Setting up the
  Data and the Model" (verified by direct fetch) — states and motivates He
  initialization (`w = randn(n) * sqrt(2.0/n)`) as the ReLU-specific fix to
  Xavier/Glorot, in the same variance-preserving spirit as this section,
  but the page contains **no numbered exercises**; a narrative companion,
  not a source of adaptable problems — https://cs231n.github.io/neural-networks-2/
- CS231n, Assignment 2, `FullyConnectedNets.ipynb`, "Inline Question 1"
  (verified via mirror) — after tuning only learning rate and weight-scale
  to overfit 50 examples with 3-layer vs. 5-layer nets, asks which depth
  is more sensitive to initialization scale and why — a close empirical
  cousin of this section's own depth-sweep experiment —
  https://github.com/mantasu/cs231n/blob/master/assignment2/FullyConnectedNets.ipynb

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Symmetry Beyond Permutation.** Besides the permutation
   symmetry among an MLP's hidden units, name one other concrete
   architectural symmetry in a standard feedforward layer that would
   likewise prevent gradient descent from breaking it under identical
   initialization. Give the specific transformation that leaves the
   function unchanged, and state what kind of initialization breaks it.
   *Provenance:* original (rewrite of exercise 1; removes filler phrasing,
   requires a specific transformation).
1. [conceptual] **Zero-Init Failure in Linear Models.** For linear
   regression and for softmax regression (both single-layer, no hidden
   units), initialize all weights to the same constant $c$ and walk
   through one step of gradient descent by hand. Does the symmetry-
   breaking failure this section describes for MLP hidden units also
   apply here? Explain in one or two sentences, referencing the presence
   or absence of permutation symmetry among the output units.
   *Provenance:* original (rewrite of exercise 2; removes filler phrasing,
   requires an explicit worked step).
1. [short-code] **Verifying the Renormalization Trick.** The section's
   depth-sweep code renormalizes activations after each layer by dividing
   by $\sqrt{\text{gain}}$, justified by ReLU's non-negative homogeneity,
   $\text{ReLU}(\alpha x)=\alpha\,\text{ReLU}(x)$ for $\alpha>0$. Prove
   this identity. Then modify the depth-sweep code to skip renormalization
   and run only 15 layers (to avoid overflow); confirm numerically that
   the un-renormalized second moment at layer 15 equals the renormalized
   curve's value at layer 15 times the product of the discarded per-layer
   gains.
   *Provenance:* adapted from Prince, UDL Problems 3.5/4.3 (overlap: high;
   cite on adoption).
1. [conceptual] **A Named Conditioning Bound.** State the submultiplicativity
   property of the spectral norm, $\|AB\|\le\|A\|\cdot\|B\|$. Using this
   bound, derive an upper bound on the growth rate of the $L$-layer
   gradient product from this section (the product of $L$ Jacobians) in
   terms of the largest per-layer spectral norm, and state what condition
   on that per-layer norm keeps the bound from growing or shrinking
   geometrically with depth.
   *Provenance:* original (rewrite of exercise 5; names a specific bound
   in place of the vague original prompt).
1. [short-code] **Applying LARS to a Diverging Toy Example.** Summarize, in
   your own words, the per-layer learning-rate-scaling rule proposed by
   You, Gitman, and Ginsburg (2017) (LARS): each layer's effective
   learning rate is scaled by the ratio of that layer's weight norm to its
   gradient norm. Construct a small 3-layer linear network whose naive-
   scale ($\mathcal{N}(0,1)$) initialization causes the loss to diverge
   within a few SGD steps (reusing this section's exploding-gradient
   setup), then apply the LARS scaling rule by hand to the first update
   and show it keeps the update bounded where the unscaled rule does not.
   *Provenance:* adapted from You, Gitman & Ginsburg, "Large Batch
   Training of Convolutional Networks" (LARS), 2017 (overlap: med; already
   cited in-section, now given a concrete artifact) (rewrite of exercise
   6).

---

## chapter_multilayer-perceptrons/generalization-deep.md — Generalization in Deep Learning

**Topic:** Why parameter count fails to predict deep-network test error;
double descent; the nonparametric/NTK lens; early stopping and implicit
regularization; grokking as a capstone example of optimization-time-
dependent generalization.

**Current exercises:** 7; disposition: keep 6, rewrite 1, drop 0 — items 6
(epoch-wise double descent) and 7 (grokking) are outstanding extended
exercises, kept exactly as-is; only item 3 is pure fact-recall with no
artifact to produce.

**External sources found:**
- Michael Nielsen, NNDL ch. 3, "Problem: Modify network2.py so that it
  implements early stopping using a no-improvement-in-$n$-epochs strategy"
  (verified by direct fetch) — asks for exactly this section's patience
  criterion as a coding task, plus an open invitation to design an
  alternative stopping rule — the closest possible external match for
  upgrading this section's exercise 3 —
  http://neuralnetworksanddeeplearning.com/chap3.html
- Simon Prince, *Understanding Deep Learning*, Problem 8.4 (verified
  against book text) — asks what happens to train/test performance when
  training-set size is increased to exactly match a model's parameter
  count, i.e., crossing the interpolation threshold from below —
  https://udlbook.github.io/udlbook/
- Simon Prince, *Understanding Deep Learning*, Problem 8.5 (verified) —
  asks for the implications of model capacity exceeding the number of
  training points for a heteroscedastic model, and a proposed fix —
  https://udlbook.github.io/udlbook/
- No good external tradition found for reproducing epoch-wise double
  descent or grokking as graded coursework: both require training a model
  far past convergence (hundreds of epochs, or $10^5$+ steps), expensive
  for a homework budget. The closest material is the original research
  papers (Nakkiran et al. 2021; Power et al. 2022), not course assignments
  — this section's own exercises 6-7 are already the rigorous, hands-on
  version of this material and should stay exactly as written.
- MIT 6.390's problem sets are hosted on an authenticated CAT-SOOP/Canvas
  platform (introml.mit.edu links only to canvas.mit.edu and catsoop.org,
  verified by direct fetch); no publicly viewable exercise could be
  verified from this course for this section, or for this chapter
  generally.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Implementing Patience-Based Early Stopping.** Implement a
   no-improvement-in-$n$-epochs early-stopping rule for the Fashion-MNIST
   MLP from mlp-implementation.md: track validation loss after each epoch
   and stop once it fails to improve by more than a small $\epsilon$ for
   $n$ consecutive epochs (choose $n,\epsilon$ explicitly). Run it on a
   schedule long enough to normally overfit, and report the epoch at which
   it stops versus the epoch of best true validation performance.
   *Provenance:* adapted from Nielsen, NNDL ch. 3, early-stopping problem
   (overlap: high; cite on adoption) (rewrite of exercise 3).
1. [short-code] **Locating the Interpolation Threshold.** For the
   one-hidden-layer Fashion-MNIST MLP ($795h+10$ parameters for $h$ hidden
   units), solve for the $h$ at which parameter count equals the
   60,000-example training-set size. Train three models — at that $h$, at
   roughly $4h$ (comfortably over-parametrized), and at roughly $h/4$
   (comfortably under-parametrized) — for enough epochs to approach zero
   training error where possible, and plot test error against $h$. Does
   test error behave non-monotonically near your computed threshold?
   *Provenance:* adapted from Prince, UDL Problem 8.4 (overlap: med; cite
   on adoption).
1. [conceptual] **Capacity Exceeding Data: A Concrete Case.** Consider a
   version of the Fashion-MNIST MLP with enough capacity to drive training
   loss to exactly zero, interpolating every training label including any
   accidental duplicate or near-duplicate inputs with conflicting labels.
   Explain what this implies for how the model must treat two training
   examples with identical pixels but different labels, and propose one
   concrete change (to the loss, architecture, or training procedure) that
   would handle this case more sensibly than forcing zero training loss.
   *Provenance:* adapted from Prince, UDL Problem 8.5 (overlap: med; cite
   on adoption).
1. [conceptual] **Implicit Bias Beyond Linear Models.** The section notes
   that gradient descent on linearly separable data with logistic loss
   converges to the max-margin separator (Soudry et al. 2018), explicitly
   flagging that this is *not* established for general deep networks. Give
   one concrete reason the linear-case proof technique doesn't
   straightforwardly extend to a two-layer ReLU network, referencing the
   non-convexity or non-uniqueness of solutions in the deep case.
   *Provenance:* original.
1. [short-code] **A Falsifiable NTK Prediction.** The section describes the
   neural-tangent-kernel limit: as hidden width grows, a randomly
   initialized MLP's training dynamics approach a fixed kernel method.
   Train the Fashion-MNIST MLP at three widths spanning at least a 10x
   range (e.g., 32, 256, 2048 hidden units), holding all other
   hyperparameters fixed, and check whether wider networks' loss curves
   (vs. training step, not wall-clock time) become progressively more
   similar to each other, as the NTK picture predicts. Report the metric
   used to compare curve similarity.
   *Provenance:* original (operationalizes the section's own NTK
   discussion into a checkable experiment).

---

## chapter_multilayer-perceptrons/dropout.md — Dropout

**Topic:** Dropout as structured multiplicative noise during training
(inverted-dropout formula, thinned-subnetwork/ensemble/anti-co-adaptation
views), implemented from scratch and via framework layers on the
Fashion-MNIST MLP.

**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — one of
the strongest, most concretely specified exercise sets in the chapter
(most items already name a metric or plot as the deliverable); only item
8's "can you develop..." phrasing needs fixing.

**External sources found:**
- Stanford CS231n, Assignment 2, `Dropout.ipynb` (verified via public
  mirrors) — implements inverted dropout in a from-scratch fully connected
  network and compares training/validation accuracy curves with and
  without dropout on a reduced (500-example) training set, the same
  paired comparison this section's own kept exercise 2 asks for —
  https://cs231n.github.io/assignments2023/assignment2/ (assignment index;
  mirror verified: github.com/mantasu/cs231n/blob/master/assignment2/Dropout.ipynb)
- Michigan EECS 498-007/598-005, Assignment 3, Q1 "Fully-Connected Neural
  Network" (Fall 2020; verified by direct fetch) — explicitly lists
  "Implement Dropout to regularize networks" as one of the assignment's
  goals, alongside batch normalization and modular backprop, in the same
  from-scratch spirit as this section —
  https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment3.html
- Nitish Srivastava et al., "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting," JMLR 2014 (the paper this section is built
  on; URL verified) — Figure 9 visualizes features learned with and
  without dropout on MNIST to argue against co-adaptation, and Section 7
  compares dropout against other regularizers on several datasets — a
  natural source for a "visualize what dropout changes" exercise beyond
  the loss-curve comparisons already in this section —
  https://jmlr.org/papers/v15/srivastava14a.html
- Simon Prince, *Understanding Deep Learning*, ch. 9 narrative (Figures
  9.8-9.9, verified against book text) — illustrates dropout as removing
  a "kink" from a piecewise-linear ReLU decision function when a hidden
  unit is dropped, tying dropout directly to the piecewise-linear
  structure this book's own mlp.md establishes. Notably, Prince's
  end-of-chapter Problems 9.1-9.6 cover L0/L1/L2 weight decay and label
  smoothing but include **no numbered problem on dropout specifically** —
  a thin formal-exercise tradition for dropout even in a book that
  discusses it at length.
- Michael Nielsen, NNDL ch. 3, "Problems: Modify the code to implement L1
  regularization ... can you find a regularization parameter that enables
  you to do better than unregularized" (verified) — an L1-regularization
  analogue of this section's own kept exercise 6 (dropout vs. weight
  decay); Nielsen's book has no dedicated dropout exercise either,
  consistent with the thin-tradition finding above.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **A New Noise-Injection Method.** Design one concrete
   alternative to dropout and DropConnect for injecting random noise
   during training — for example, additive Gaussian noise on
   post-activation values, or randomly rescaling (rather than zeroing)
   each unit's output. State your method's randomization rule and any
   rescaling needed to preserve the expectation, implement it, and train
   it on Fashion-MNIST with the same architecture and epoch budget as
   `DropoutMLPScratch`. Report whether it matches, exceeds, or falls short
   of standard dropout's validation accuracy.
   *Provenance:* original (rewrite of exercise 8; keeps the match/outperform
   criterion, removes the filler phrasing).
1. [conceptual] **Dropout as Region Removal.** This chapter's own mlp.md
   section shows a ReLU network computes a continuous piecewise-linear
   function whose joints are set by the hidden units. For a single hidden
   layer, explain what happens to that function's joints when one hidden
   unit is dropped for a given forward pass: does a joint disappear, move,
   or stay fixed? Sketch, by hand, a 1-D toy example with 2-3 hidden units
   before and after dropping one unit.
   *Provenance:* inspired by Prince, UDL ch. 9 dropout/kink discussion
   (overlap: low).
1. [short-code] **Visualizing Feature Co-adaptation.** Train the
   `DropoutMLP` and a no-dropout MLP of the same architecture on
   Fashion-MNIST, then visualize each first-layer hidden unit's incoming
   weight vector as a $28\times28$ image (reshaping the 784-dim weight
   row). Compare at least 16 units side by side for each model: do the
   no-dropout weights look noisier or more redundant with each other than
   the dropout-trained ones?
   *Provenance:* adapted from Srivastava et al. 2014, JMLR, Figure 9
   (overlap: med; cite on adoption).
1. [conceptual] **Dropout Rate and Effective Ensemble Size.** A network
   with $n$ hidden units has $2^n$ possible dropout masks. For this
   section's two-hidden-layer (256, 256) `DropoutMLP`, compute $2^n$ for
   $n=512$ total hidden units, and separately compute the *expected*
   number of units retained in a single mask (a binomial-mean calculation)
   for $p=0.2$ and $p=0.5$, as used in this section's two layers. Explain
   in one sentence why a very large $2^n$ does not mean each of the
   $2^n$ "subnetworks" is trained to convergence independently.
   *Provenance:* original.
1. [short-code] **Input Dropout.** This section applies dropout only to
   hidden-layer outputs. Add dropout directly to the input features
   (before the first linear layer) at a small rate (e.g., $p=0.1$), on top
   of the existing hidden-layer rates (0.2, 0.5), and compare validation
   accuracy against the section's original configuration. Does adding
   input dropout help, hurt, or make no measurable difference here? Give
   one sentence of intuition tying your answer to how much redundancy
   Fashion-MNIST's 784 raw pixel features have.
   *Provenance:* original (extends the section's own placement convention
   to a location — the input — it doesn't test).

---

## chapter_multilayer-perceptrons/kaggle-house-price.md — Predicting House Prices on Kaggle

**Topic:** Applying the chapter's preprocessing/regularization/validation
toolkit end to end (impute, standardize, one-hot, log-RMSE, K-fold CV, MLP
vs. linear baseline, submission) to the Ames-housing Kaggle competition.

**Current exercises:** 7; disposition: keep 6, rewrite 1, drop 0 — a
strong, well-scoped set; item 6 already anticipates the tree-vs-MLP
comparison this dataset invites, and item 7's out-of-fold target-encoding
requirement is unusually rigorous; only item 2's hint phrasing needs
fixing.

**External sources found:**
- Kaggle Learn, "Intermediate Machine Learning," Exercise 2 "Missing
  Values" (verified against the actual notebook source) — on the same
  "Housing Prices Competition for Kaggle Learn Users" Ames-derived
  dataset, has the learner first investigate which columns have missing
  values and how many, then choose and justify a strategy (drop columns
  vs. mean imputation vs. imputation with a missing-indicator column)
  before comparing mean absolute error — the closest external analogue to
  this section's own missing-value handling and this catalog's rewritten
  exercise 2 —
  https://github.com/Kaggle/learntools/blob/master/notebooks/ml_intermediate/raw/ex2.ipynb
  (source; hosted at kaggle.com/learn/intermediate-machine-learning)
- Kaggle Learn, "Intermediate Machine Learning," Exercise 3 "Categorical
  Variables" (verified) — same dataset; compares dropping categoricals,
  ordinal encoding, and one-hot encoding by mean absolute error, and has
  the learner reason quantitatively about cardinality (e.g., how many
  columns a 100-unique-value categorical adds under each encoding) —
  https://github.com/Kaggle/learntools/blob/master/notebooks/ml_intermediate/raw/ex3.ipynb
- Kaggle Learn, "Intermediate Machine Learning," Exercise 5
  "Cross-Validation" (verified) — same dataset; has the learner write a
  `get_score()` function reporting mean cross-validated MAE for a
  pipeline, sweep eight values of `n_estimators`, and pick the best by CV
  score — directly the same "use K-fold CV to select a hyperparameter"
  task as this section's own kept exercise 3 —
  https://github.com/Kaggle/learntools/blob/master/notebooks/ml_intermediate/raw/ex5.ipynb
- Kaggle Learn, "Intermediate Machine Learning," Exercise 7 "Data Leakage"
  (verified) — poses scenario-based questions (a shoelace-demand model
  with a feature only knowable after the fact, a too-accurate
  cryptocurrency predictor, a surgeon-infection-rate feature, and — the
  most directly relevant — a housing-price scenario asking which of four
  candidate features is most likely to leak information) — a strong
  source for a *new* target-leakage exercise this section currently lacks
  — https://github.com/Kaggle/learntools/blob/master/notebooks/ml_intermediate/raw/ex7.ipynb
- Kaggle Learn, "Intro to Machine Learning," Exercise 5 "Underfitting and
  Overfitting" (verified) — has the learner loop over candidate values of
  a capacity-controlling hyperparameter (`max_leaf_nodes`), score each by
  validation MAE, select the best, then refit on all available data with
  that choice — the same "tune on a validation/CV score, then refit on
  everything" two-step this section's own K-fold-then-submit discussion
  describes —
  https://github.com/Kaggle/learntools/blob/master/notebooks/machine_learning/raw/ex5.ipynb

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Missingness Correlated with Price.** Construct one
   concrete, plausible example of a feature in this dataset whose
   missingness is *not* random with respect to sale price — for instance,
   a pool-quality feature recorded only for houses that have a pool, so
   "missing" is itself informative. Explain what bias mean-imputing that
   feature introduces, and in which direction it would bias predicted
   prices for houses that do and don't have the feature.
   *Provenance:* original (rewrite of exercise 2; keeps the
   missing-not-at-random idea, removes filler phrasing).
1. [short-code] **A Target-Leakage Audit.** Identify at least one raw
   feature in this dataset that could plausibly be filled in only after
   (or because of) the sale, and explain why including it would inflate
   cross-validated performance without helping a real deployed model.
   Then verify empirically: add a synthetic feature to the preprocessed
   training data that is a noisy function of the log-price itself (e.g.,
   $\log(\text{SalePrice})$ plus small Gaussian noise), rerun K-fold
   cross-validation with it included, and report how much the CV
   log-RMSE improves — that gap is the leakage this exercise makes
   concrete.
   *Provenance:* adapted from Kaggle Learn, "Intermediate Machine
   Learning" Exercise 7, "Data Leakage" (overlap: med; cite on adoption).
1. [conceptual] **Cardinality and Encoding Cost.** Among this dataset's
   categorical columns, find the one with the highest cardinality and
   report that count. If one-hot encoded, how many columns would it
   contribute to the preprocessed feature matrix? If ordinally or
   target-encoded instead, how many? State which choice you would make
   for this specific column and why, given the dataset has only ~1,500
   rows.
   *Provenance:* adapted from Kaggle Learn, "Intermediate Machine
   Learning" Exercise 3, "Categorical Variables" (overlap: med; cite on
   adoption).
1. [short-code] **Validation-Then-Refit Gap.** Sweep at least five
   candidate values of the small MLP's hidden-unit count (e.g., 8, 16, 32,
   64, 128), selecting the best by K-fold CV log-RMSE as this section
   already does for other hyperparameters. Then compare two ways of
   producing a final model at the selected width: (a) ensembling the $K$
   already-trained fold models, as this section's submission code does,
   versus (b) refitting one fresh model on the complete training set.
   Report each option's CV-estimated (or leaderboard, if available)
   log-RMSE, and discuss why a training-set-only proxy would be
   optimistic.
   *Provenance:* adapted from Kaggle Learn, "Intro to Machine Learning"
   Exercise 5, "Underfitting and Overfitting" (overlap: low-med; cite on
   adoption).
1. [conceptual] **Why Log-Space Averaging.** This section averages the $K$
   fold models' *log*-price predictions before exponentiating, calling
   this a geometric mean in price space. Prove algebraically that this is
   *not* the same as exponentiating each fold's prediction first and then
   taking the arithmetic mean in price space, and state which of the two
   better matches what the RMSLE scoring metric actually penalizes.
   *Provenance:* original.
