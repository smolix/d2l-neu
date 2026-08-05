# chapter_linear-regression — Exercise Catalog

**Chapter overview.** Seven sections carry `## Exercises`: linear-regression (3.1),
oo-design (3.2), synthetic-regression-data (3.3), linear-regression-scratch (3.4),
linear-regression-concise (3.5), generalization (3.6), weight-decay (3.7). Per the
prior style review, this is the strongest-exercised group in the book (109
exercises, near-zero defects); our job here is mostly *addition*, not repair.
Best external sources, independently verified by fetching each page: Stanford
CS229 Problem Set #1 (normal equations via Newton's method; the Poisson-regression
GLM problem is an almost exact structural twin of this chapter's own apple-counts
exercise); Andrew Ng's Coursera ML Exercise 1 (the canonical "implement gradient
descent, check against a known number" exercise linear-regression-scratch.md is
built from); Boyd & Vandenberghe's VMLS Additional Exercises (least-squares
projection/robustness problems, fully verified from the PDF); ISL ch. 2/5/6
(bias-variance curve-sketch, bootstrap, and ridge/lasso conceptual exercises —
all confirmed); and Prince's *Understanding Deep Learning* ch. 8/9 notebooks
(double descent, L2 regularization, Bayesian regularization — all three fetched
directly from the book's own repo). Two real coverage gaps: (1) oo-design.md's
Module/DataModule/Trainer scaffold has **no** external exercise tradition at all
(checked PyTorch Lightning's own docs — nothing); (2) ISL ch. 5's excellent
bootstrap exercise doesn't transfer, because generalization.md never introduces
the bootstrap in its own prose (K-fold only) — a gap worth flagging to the book,
not to us. The one recurring defect pattern, showing up independently in four
different sections (weight-decay ex2/ex5, scratch ex6, concise ex4,
generalization ex7), is the unbounded "vary X and see what happens" prompt with
no stated range or metric; every rewrite below targets exactly that.

---

## chapter_linear-regression/linear-regression.md — Linear Regression

**Topic:** The linear model, squared loss, the analytic (normal-equations /
projection) solution, minibatch SGD, and the probabilistic motivation of squared
loss via Gaussian noise, generalized to a noise-model-to-loss table (Laplace,
log-Gaussian, Poisson).

**Current exercises:** 8; disposition: keep 7, rewrite 0, drop 1 — the style
review flagged zero defects and zero clarity issues across all 8; the one drop
(exercise 3, quadratic features "in a deep network") is a scope call, not a
quality one — it references hidden-layer composition the book hasn't introduced
yet and sits structurally apart from the loss/noise-model thread running through
the rest of the list.

**External sources found:**
- Stanford CS229, Problem Set #1 (Public Course materials), Problem 1 — proves
  that one step of Newton's method on the least-squares objective lands exactly
  on the normal-equation solution θ* = (XᵀX)⁻¹Xᵀy — https://see.stanford.edu/materials/aimlcs229/problemset1.pdf
- Stanford CS229, Problem Set #1 (2018 Autumn offering), "Poisson Regression" —
  show the Poisson PMF is exponential-family, derive E[T(y)], derive the SGA
  update, then implement and fit — a near-exact structural twin of this
  section's own apple-counts exercise (8) — https://github.com/maxim5/cs229-2018-autumn/blob/main/problem-sets/PS1/PS1-3%20Poisson%20Regression.ipynb
- Boyd & Vandenberghe, VMLS Additional Exercises, Ex. 12.5 — five true/false
  statements about a tall, full-column-rank least-squares fit, including whether
  the residual must be orthogonal to the columns of A — directly tests the
  orthogonal-projection reading this section develops in prose but never quizzes
  directly — https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf (p.49-50)
- Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, Ex. 3.12 —
  shows ridge regression is recoverable as ordinary least squares on X augmented
  with √λI rows and y augmented with zeros — the same λI-augmentation idea this
  section's exercise 4(f) gestures at — https://www.danli.org/2021/03/23/esl-chapter-3-exercises/
- Boyd & Vandenberghe, VMLS Additional Exercises, Ex. 13.10 — compares a plain
  linear fit against a "sign-dependent" (kinked) feature-engineered fit on
  train/test RMS error, the same flavor of loss-choice-affects-robustness
  question this section's outlier demo raises — https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf (p.55)

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Constant fit and its loss.** Given scalars $x_1,\ldots,x_n$, find
   the $b$ minimizing $\sum_i (x_i-b)^2$, connect the answer to the mean of a
   normal distribution, then repeat for $\sum_i |x_i-b|$ and identify what
   changes.
   *Provenance:* original (kept from the book).
1. **Affine equals linear, augmented.** Prove that affine functions
   $\mathbf{x}^\top\mathbf{w}+b$ are exactly the linear functions of
   $(\mathbf{x},1)$.
   *Provenance:* original (kept from the book).
1. [conceptual] **Singular design matrix.** With $\mathbf{X}^\top\mathbf{X}$
   singular, characterize the set of minimizers, identify which one the
   pseudoinverse picks, show that adding coordinate-wise Gaussian noise to
   $\mathbf{X}$ fixes it in expectation, describe what SGD does in this regime,
   and connect the $\lambda\mathbf{I}$ fix to the ridge estimator of
   :numref:`sec_weight_decay`.
   *Provenance:* original (kept from the book).
1. [conceptual] **Laplace noise and its pathology.** Derive the negative
   log-likelihood under Laplace-distributed noise, note whether it has a closed
   form, then design a minibatch SGD scheme for it and diagnose what goes wrong
   near the optimum (hint: the subgradient of $|\cdot|$ doesn't vanish there).
   *Provenance:* original (kept from the book).
1. **Composing two linear layers.** Explain why stacking two linear layers
   collapses to a single linear map, and what property must an intermediate
   operation have to avoid this collapse.
   *Provenance:* original (kept from the book).
1. [conceptual] **Prices are not Gaussian.** Argue why additive Gaussian noise is
   the wrong model for house or stock prices, why regressing on $\log(\text{price})$
   fixes the negative-price problem, and what specifically goes wrong for
   penny stocks (tick-size granularity), referencing Black–Scholes for context.
   *Provenance:* original (kept from the book).
1. [conceptual] **Counting apples with Poisson regression.** Explain why Gaussian
   noise is a poor model for counts, prove the Poisson rate $\lambda$ equals the
   expected count, then design loss functions for estimating $\lambda$ directly
   and for estimating $\log\lambda$.
   *Provenance:* original (kept from the book); independently overlaps with the
   CS229 Poisson-Regression problem above (overlap med — same GLM derivation
   shape, different worked example).
1. [short-code] **A third loss on the outlier demo.** Extend this section's own
   corrupted-label demo (20 points on $y=2x$, one label corrupted to 10000): fit
   a third estimator that regresses on $\log y$ (clipping/shifting as needed to
   keep $y>0$), and report its recovered slope alongside the existing
   squared-loss and MAE fits. State which of the three the corrupted point
   distorts least and why, tying the answer back to the noise-model table.
   *Provenance:* inspired by VMLS Ex. 13.10 (compare model choices on a
   corrupted/robustness axis; overlap low — different mechanics, same
   "which fit survives contamination" question).

---

## chapter_linear-regression/oo-design.md — Object-Oriented Design for Implementation

**Topic:** The book's `Module` / `DataModule` / `Trainer` scaffold, the
`add_to_class` monkey-patching decorator, `HyperParameters.save_hyperparameters()`,
and the asynchronous `ProgressBoard` logger.

**Current exercises:** 6; disposition: keep 5, rewrite 1, drop 0 — the only
defect the style review found anywhere in this section was cosmetic (exercise 1's
(a)/(b) crammed into one inline paragraph instead of a nested list); the rewrite
below is purely a reformat plus one added deliverable, no content change.

**External sources found:** none. This section teaches the book's own bespoke
code architecture (a training-loop scaffold, a monkey-patch decorator, an
async plotting queue), not a topic any of the suggested course/textbook sources
address — ISL, ESL, CS229, VMLS, and CS189 are all statistics/ML-content
courses, not software-architecture-for-ML-pedagogy courses. We checked the one
plausible outside tradition directly: PyTorch Lightning's own introductory docs
(https://lightning.ai/docs/pytorch/stable/starter/introduction.html), which the
scaffold explicitly credits as its inspiration — fetched and confirmed it
contains explanatory code samples but no exercises, challenges, or "try this
yourself" prompts of any kind. This is a clean case of "no good external
exercise tradition" rather than a search failure: the topic is intrinsically
book-specific.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Monkey-patch after the fact.** Add a `greet(self)` method to
   class `A` *after* an instance already exists, using `@add_to_class(A)`, and
   confirm the instance can call it. Then define `greet` on `A` the same way but
   *without* the decorator and call it again; explain from `setattr`'s semantics
   why the decorated version works and the undecorated one doesn't.
   *Provenance:* original (kept from the book).
1. [conceptual] **Where the optimizer lives.** `Module.configure_optimizers` puts
   the optimizer choice on the model rather than passing it into `Trainer`.
   Argue the advantage of this placement, then describe a training setup (e.g.,
   shared optimizer state across multiple models) where it becomes awkward.
   *Provenance:* original (kept from the book).
1. [short-code] **Add a test split.** Extend `DataModule` with a
   `test_dataloader` method and extend `Trainer.fit` to run one evaluation pass
   over it after training completes. State the one invariant a test loader must
   satisfy that a validation loader need not (hint: how many times may each be
   consulted during model development).
   *Provenance:* original (kept from the book).
1. [short-code] **`save_hyperparameters` without `inspect`.** Implement a version
   of `save_hyperparameters` that does not use Python's `inspect` module — for
   instance, by requiring the caller to pass `locals()` explicitly — and verify
   it reproduces the same attributes as the original on class `B`. State one
   thing the `inspect`-based version buys you that your version gives up.
   *Provenance:* original (kept from the book).
1. [short-code] **A synchronous `ProgressBoard`.** `ProgressBoard.draw` hands
   values to a background thread; implement a synchronous variant that plots
   immediately on the calling thread instead. Time both across a training run
   with frequent `draw` calls and report when the synchronous version is
   measurably slower, and when the two are indistinguishable.
   *Provenance:* original (rewritten from the book's "(Advanced) sketch a
   synchronous alternative" to require an actual implementation and a
   measurement, rather than a sketch with no checkable deliverable).
1. [conceptual] **Life without `save_hyperparameters`.** Remove the
   `save_hyperparameters()` call from class `B`. Predict whether `self.a` and
   `self.b` still print correctly, then say why (or why not) based on how
   Python resolves attribute lookups on an instance versus a class.
   *Provenance:* original (kept from the book).

---

## chapter_linear-regression/synthetic-regression-data.md — Synthetic Regression Data

**Topic:** Building a `SyntheticRegressionData` `DataModule` with known
ground-truth $\mathbf{w}^*, b^*$, and comparing a hand-rolled minibatch loader
against the framework's built-in one.

**Current exercises:** 5; disposition: keep 4, rewrite 1, drop 0 — the style
review's only flag was exercise 2(a) ("what happens if we cannot hold all data
in memory?") lacking a deliverable, unlike its sibling 2(b); the rewrite gives
2(a) a concrete artifact without touching 2(b), which is already excellent.

**External sources found:** thin, and that thinness is itself the finding. None
of the suggested sources (CS229, ISL, ESL, VMLS, CS189) treat "build a synthetic
dataset with known generating parameters to validate an implementation" as its
own graded exercise — it's universal *practice* in ML pedagogy, not a topic
anyone assigns a problem about. The closest verified parallel is the general
"check your implementation against a known/expected numeric answer" epistemic
used throughout Andrew Ng's Coursera ML Exercise 1 (e.g., "with theta = [0, 0],
cost computed = 32.07") — https://github.com/dibgerge/ml-coursera-python-assignments
(Exercise1/exercise1.ipynb, fetched directly) — which is the same
verify-against-ground-truth logic this section teaches, just applied to a
pre-supplied dataset rather than a self-generated one. Exercise 2(b)'s own
on-disk-shuffle ask already cites its ideal source directly in the book's own
text (Naor & Reingold 1999 on pseudorandom permutations) — that citation is
already better than anything we could add.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **`drop_last` and its cousins.** State what PyTorch's
   `drop_last` argument and TensorFlow's `batch(..., drop_remainder=...)` do to
   the final partial minibatch, and give one concrete training scenario where
   you would want it enabled and one where you would not.
   *Provenance:* original (kept from the book).
1. [conceptual] **Data too large to hold at once.** Suppose both the parameter
   dimension and the number of examples are too large to fit in memory. Name
   two concrete constraints this creates for a training loop (e.g., on
   minibatch construction and on parameter storage), and then design an
   efficient on-disk reshuffling scheme that avoids storing an explicit
   permutation table (hint: pseudorandom permutation generators).
   *Provenance:* original (exercise 2(a) rewritten to require two named,
   checkable constraints in place of the open "what happens" prompt; 2(b)
   unchanged, already citing Naor & Reingold 1999).
1. [short-code] **An on-the-fly generator.** Implement a data generator that
   produces a fresh minibatch of synthetic $(\mathbf{X},\mathbf{y})$ pairs every
   time it is called, rather than pre-generating and storing the full dataset,
   and confirm two successive calls return different data.
   *Provenance:* original (kept from the book).
1. [conceptual] **Designing for reproducibility.** Design a random data
   generator that produces the *same* dataset on every call. Contrast a
   single-global-seed API against one that threads an explicit random state
   through every draw, and explain why re-using one state for both
   $\mathbf{X}$ and $\boldsymbol{\epsilon}$ (instead of advancing it between
   the two draws) would silently break reproducibility or correctness.
   *Provenance:* original (kept from the book).
1. [short-code] **Recovering $\mathbf{w}^*$ under noise.** Sweep the noise
   standard deviation over $\{0.001, 0.01, 0.1, 0.5, 1.0\}$, fit a linear model
   on each resulting dataset, and plot $\|\hat{\mathbf{w}}-\mathbf{w}^*\|_2$
   against $\sigma$. State your predicted scaling with $\sigma$ and with the
   number of training examples before plotting, then say whether the plot
   agrees.
   *Provenance:* original (kept from the book).

---

## chapter_linear-regression/linear-regression-scratch.md — Linear Regression Implementation from Scratch

**Topic:** Implementing linear regression with only tensors and autograd —
parameter init, forward pass, squared loss, a hand-rolled SGD class, and the
training loop — checked against the synthetic data's known ground truth.

**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — the style
review's only flag was exercise 6 (learning-rate experiment) lacking a stated
range or target metric; every other exercise, including the distinctive
physics-themed ones (Ohm's law, Planck's law) and the build-your-own-Huber-loss
exercise 7, was flagged clean and should stay.

**External sources found:**
- Andrew Ng, Coursera Machine Learning, Exercise 1 (Linear Regression) — students
  implement `computeCost`, batch `gradientDescent`, `featureNormalize`, and the
  closed-form `normalEqn`, and check results against stated expected numbers
  (e.g., "with theta = [0, 0], cost computed = 32.07"); the notebook explicitly
  states "your value of J(θ) should never increase, and should converge to a
  steady value" — the exact convergence check this section's own training loop
  relies on — https://github.com/dibgerge/ml-coursera-python-assignments (Exercise1/exercise1.ipynb, fetched)
- Stanford CS229, Problem Set #1 (Public Course materials), Problem 1 — one
  Newton step on the least-squares objective converges exactly to the
  closed-form solution — a nice closed-form cross-check for whatever
  $\hat{\mathbf{w}}$ the from-scratch SGD loop converges to —
  https://see.stanford.edu/materials/aimlcs229/problemset1.pdf
- Boyd & Vandenberghe, VMLS Additional Exercises, Ex. 12.1 — solve the same
  least-squares problem three ways (backslash, normal equations, pseudoinverse)
  and confirm they agree up to roundoff — the same "does my from-scratch
  answer match the closed form" check this section's whole implementation is
  building toward — https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf (p.48)

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Initialization at the extremes.** Predict whether training
   still succeeds if weights are initialized to exactly zero, and separately if
   they are drawn with variance 1000 instead of 0.01; explain both answers in
   terms of this being a single convex linear layer rather than a deep network.
   *Provenance:* original (kept from the book).
1. [short-code] **Ohm's law by autograd.** Treat resistance as an unknown linear
   parameter relating voltage and current, generate or use measured
   voltage/current pairs, and fit the relationship using this section's own
   autograd-based training loop.
   *Provenance:* original (kept from the book).
1. [short-code] **Planck's law by curve fitting.** Given the spectral-density
   formula $B(\lambda,T)$ and measured energy at several wavelengths, fit the
   temperature $T$ by treating it as a learnable parameter under this section's
   training loop.
   *Provenance:* original (kept from the book).
1. [conceptual] **When second derivatives misbehave.** Identify what goes wrong
   if you try to compute second derivatives of this section's loss with the
   tools introduced so far, and propose a fix.
   *Provenance:* original (kept from the book).
1. [conceptual] **Why `reshape` guards the loss.** Explain what silently goes
   wrong in the loss computation if `y_hat` and the reshaped `y` have
   mismatched shapes, in terms of broadcasting rather than an outright error.
   *Provenance:* original (kept from the book).
1. [short-code] **A bounded learning-rate sweep.** Train the from-scratch model
   at each learning rate in $\{0.001, 0.01, 0.03, 0.1, 0.3, 1.0\}$ for a fixed
   30 epochs, plot the training-loss curve for each, and report the smallest
   number of epochs needed to reach within 10% of the noise floor
   $\sigma^2/2$ at each rate (marking "never" where it diverges or fails to
   reach that band).
   *Provenance:* adapted from Andrew Ng's Coursera ML Exercise 1 convergence
   check ("J(θ) should never increase, and should converge"; overlap med) —
   replaces the book's open "experiment with different learning rates" with a
   fixed grid and a concrete stopping criterion.
1. [short-code] **Building Huber loss.** Implement the absolute-value loss and
   compare it against squared loss on (a) the regular synthetic data and (b) a
   version with one label perturbed to $10000$; then design and implement a
   loss combining both behaviors (quadratic near zero, linear in the tails) and
   confirm it recovers a fit close to the uncorrupted case even after
   perturbation.
   *Provenance:* original (kept from the book).
1. [conceptual] **Why reshuffle at all.** Explain why each epoch reshuffles the
   training order, then construct a small dataset ordering (e.g., sorted by
   label or grouped by class) that would break minibatch SGD if reshuffling
   were disabled, and say specifically how it breaks.
   *Provenance:* original (kept from the book).

---

## chapter_linear-regression/linear-regression-concise.md — Concise Implementation of Linear Regression

**Topic:** Rebuilding the same linear-regression model with framework layers,
built-in losses, and a built-in `SGD` optimizer, replacing each hand-rolled
piece from the previous section.

**Current exercises:** 6; disposition: keep 5, rewrite 1, drop 0 — the style
review's only flag was exercise 4 (learning-rate/epoch effect) lacking a bound
or metric, the same pattern as scratch.md's exercise 6; the sample-size sweep
(exercise 5) and the scratch-vs-concise timing comparison (exercise 6) were
both flagged as already well-specified and should stay untouched.

**External sources found:**
- Boyd & Vandenberghe, VMLS Additional Exercises, Ex. 13.10 — compares a plain
  linear fit against a feature-engineered "sign-dependent" fit on train/test RMS
  error and asks which claims about robustness/overfitting are actually always
  true — the closest external relative to this section's Huber-vs-squared
  outlier re-run (exercise 2) — https://web.stanford.edu/~boyd/vmls/vmls-additional-exercises.pdf (p.55)
- No verified tradition for the section's other framework-engineering asks
  (reduction-mode-to-learning-rate scaling in exercise 1, the sample-size
  scaling-law sweep in exercise 5, or the scratch-vs-concise timing comparison
  in exercise 6): these are specific to writing production-shaped code against
  a particular framework's API rather than a derivation or applied-statistics
  question, and none of ISL/ESL/CS229/CS189 pose homework at that level of
  framework-implementation detail. Exercise 2's own Huber-loss ask likewise has
  no direct textbook-homework match beyond the VMLS entry above — it appears to
  be a genuine strength unique to this book's exercise set.

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **Sum versus mean reduction.** State how the learning rate must
   change if the built-in loss is switched from its default mean reduction to
   sum reduction (e.g., `reduction='sum'`), and why.
   *Provenance:* original (kept from the book).
1. [short-code] **Huber loss on the outlier demo.** Look up your framework's
   available losses, swap in Huber's loss in place of squared error, and rerun
   the one-corrupted-label demo from linear-regression.md. Report whether the
   recovered slope lands closer to the robust (MAE) estimate, the least-squares
   estimate, or between the two, and relate the answer to the noise-model
   figure from that section.
   *Provenance:* original (kept from the book); independently overlaps with
   VMLS Ex. 13.10's robustness-comparison framing (overlap low).
1. [short-code] **Reading out the gradient.** Show how to read the gradient of
   the model's weight parameter directly from the framework after one backward
   pass, and confirm it matches the by-hand formula from linear-regression-scratch.md.
   *Provenance:* original (kept from the book).
1. [short-code] **A bounded learning-rate/epoch grid.** Train the concise model
   at each combination of learning rate in $\{0.01, 0.03, 0.1, 0.3\}$ and epoch
   count in $\{5, 10, 30\}$, and report the final validation loss for each of
   the 12 combinations in a small table. State which combinations fail to beat
   the single-epoch, default-learning-rate baseline.
   *Provenance:* adapted from Andrew Ng's Coursera ML Exercise 1 convergence
   check (overlap med) — same fix as scratch.md exercise 6, applied here to a
   2-D grid since both learning rate and epoch count are in play.
1. [short-code] **Sample size and estimation error.** Plot $\hat{\mathbf{w}}-\mathbf{w}$
   and $\hat b-b$ against the number of training examples, sweeping the count
   logarithmically (5, 10, 20, 50, ..., 10,000) rather than linearly, and
   explain why log spacing is the appropriate choice here rather than a linear
   one.
   *Provenance:* original (kept from the book).
1. [short-code] **From-scratch versus concise, timed.** Time the from-scratch
   implementation against this section's concise one at 10, 100, and 1000
   epochs on the same synthetic dataset. Report which is faster at each epoch
   count and whether the gap grows with epochs, and relate the answer to
   Python-level parameter bookkeeping versus framework-optimized operations.
   *Provenance:* original (kept from the book).

---

## chapter_linear-regression/generalization.md — Generalization

**Topic:** Training error versus generalization error, the classical
bias-variance U-curve via a polynomial-fitting demo, model selection, K-fold
cross-validation, and double descent as a caveat to the classical picture.

**Current exercises:** 8; disposition: keep 6, rewrite 1, drop 1 — the style
review found no clear defects in any of the 8, so both changes here are
judgment calls rather than repairs. The drop (exercise 6, VC dimension) is the
least-integrated item in the set: it's a self-contained aside that defines and
uses a complexity measure the section's own prose never mentions, disconnected
from the bias-variance/K-fold narrative running through everything else here —
better placed in a later chapter that actually develops learning-theoretic
complexity measures. The rewrite (exercise 7) converts an open "how would you
justify..." discussion prompt into a concrete subsampling experiment with a
plotted artifact.

**External sources found:**
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*,
  ch. 2, Conceptual Exercise 1 — for each of four scenarios (large $n$/small
  $p$; large $p$/small $n$; strongly nonlinear; high noise variance), state
  whether a flexible or inflexible method is expected to perform better and why
  — https://github.com/franciscoyira/islr-exercises/blob/master/ch2.md (fetched)
- ISL ch. 2, Conceptual Exercise 3 — sketch typical squared-bias, variance,
  training-error, test-error, and Bayes-error curves against model flexibility
  on one plot — the pencil-and-paper version of exactly the curve this
  section's own figure and polynomial-degree sweep produce numerically —
  same URL as above (fetched)
- ISL ch. 5, Conceptual Exercise 2 — derive that the probability a bootstrap
  sample excludes a given observation is $(1-1/n)^n$, evaluate it at $n=5,100,10000$,
  and show it converges to $1-1/e\approx0.632$ — https://github.com/ppaquay/IntroStatLearning/blob/master/Chap5.md (fetched)
- ISL ch. 5, Conceptual Exercise 3 — describe how $K$-fold CV is carried out and
  its advantages relative to both the validation-set approach and LOOCV —
  same URL as above (fetched); confirms the book's own kept exercises 4/5 on
  K-fold cost and bias are well aligned with the standard treatment
- ISL ch. 5, Applied Exercises 7-8 — implement LOOCV by hand and run
  cross-validation on a simulated dataset — general applied-CV tradition,
  supporting (not duplicating) this section's own kept exercises
- Simon J. D. Prince, *Understanding Deep Learning*, ch. 8 ("Measuring
  Performance"), end-of-chapter Problem 8.4 — for a model with a 200-unit
  hidden layer (50,410 parameters) trained on the MNIST-1D double-descent curve
  of figure 8.10b, predict what happens to training and test performance if the
  number of training examples increases from 10,000 to 50,410 (i.e., crossing
  the interpolation threshold) — verified directly from the book's own text
  (local copy, `udl_book.txt`, "Problems" following §8.4/8.5); its companion
  Notebook `8_3_Double_Descent.ipynb` asks the same predict-then-verify
  question as a runnable experiment — https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap08/8_3_Double_Descent.ipynb (fetched)

Coverage gap worth flagging (not a proposed exercise): ISL's bootstrap exercise
above is excellent, but this section's own prose never introduces the
bootstrap — only $K$-fold CV — so a bootstrap exercise here would assume a
concept the book hasn't taught yet. We did not add one.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **When is polynomial regression exact.** State the condition
   under which polynomial regression can be solved exactly (hint: relate the
   degree to the number of distinct $x$ values).
   *Provenance:* original (kept from the book).
1. [conceptual] **Five failures of IID.** Give five concrete examples where
   dependent observations make treating a dataset as IID inadvisable (e.g.,
   repeated measurements on the same patient, or consecutive frames of video).
   *Provenance:* original (kept from the book).
1. [conceptual] **Zero error, two kinds.** Say whether zero training error is
   achievable in practice and under what circumstances zero generalization
   error would be, distinguishing the two clearly.
   *Provenance:* original (kept from the book).
1. [conceptual] **The price of $K$-fold.** Explain why $K$-fold cross-validation
   is computationally expensive relative to a single train/validation split,
   in terms of the number of models fit.
   *Provenance:* original (kept from the book).
1. [conceptual] **Why $K$-fold is biased.** Explain why the $K$-fold
   cross-validation error estimate is biased relative to the error of a model
   trained on the full dataset, and in which direction.
   *Provenance:* original (kept from the book).
1. [short-code] **A learning curve for your manager.** Given a dataset your
   model underperforms on and no way to collect more data, subsample the
   existing training set at several sizes (e.g., 20%, 40%, 60%, 80%, 100%),
   retrain at each size, and plot validation error against training-set size.
   Use the resulting curve's slope as your evidence for whether more data would
   likely help.
   *Provenance:* original (exercise 7 rewritten: the book's "how would you
   justify..." discussion prompt now requires an actual subsampling experiment
   and a plotted artifact as the justification, rather than a hint alone).
1. [short-code] **The U-curve at three sample sizes.** Rerun the polynomial-degree
   sweep with `n_train` set to 10, 40, and 100, and report the degree at which
   test loss starts climbing in each case. State how this relates to the rule
   that more complex models need more data.
   *Provenance:* original (kept from the book).
1. [extended] **Past the interpolation threshold.** Extend this section's own
   polynomial-degree sweep well past `n_train` (e.g., to degree 60 on 20
   training points, using `lstsq`'s minimum-norm behavior in the rank-deficient
   regime). Before running it, predict whether test error will keep rising
   past the point where training error hits zero, or fall again; then run the
   extended sweep and plot test error across the full range. Report whether
   your prediction matches a second descent, and relate the result to the
   double-descent aside already in this section's text.
   *Provenance:* adapted from Prince, *Understanding Deep Learning*, ch. 8,
   end-of-chapter Problem 8.4 (predict train/test performance as the number of
   training examples crosses the interpolation threshold; overlap med — same
   predict-then-verify question, different model/dataset: MLP-on-MNIST-1D
   there versus this section's own polynomial/least-squares demo here) and its
   companion Notebook `8_3_Double_Descent.ipynb` (overlap low).

---

## chapter_linear-regression/weight-decay.md — Weight Decay

**Topic:** $\ell_2$ regularization (ridge) as the first regularizer, the
penalty-versus-constraint geometric picture against lasso, the multiplicative
weight-shrinkage view of the SGD update, the Bayesian MAP-with-Gaussian-prior
derivation, and the spectral (SVD) view of exactly which directions ridge
shrinks.

**Current exercises:** 6; disposition: keep 4, rewrite 2, drop 0 — the style
review flagged exercise 2 ("is it really the optimal value? does this matter?")
and exercise 5 ("what other ways might help with overfitting?") as open
rhetorical/brainstorm prompts with no stated deliverable; both are rewritten
below to require a concrete comparison or implementation. The remaining four —
the $\ell_1$-update derivation, the Frobenius-norm pointer, the lambda-sweep
experiment, and the MAP-correspondence derivation — were flagged clean and
should stay. Exercise 6 (the MAP derivation) turns out to have **high**
independent overlap with a real external exercise (UDL Problem 9.1, below) —
a good sign that the book's own strongest weight-decay exercise already
matches the standard tradition; we did not need to touch it.

**External sources found:**
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*,
  ch. 6, Conceptual Exercise 2 — which of four statements correctly describes
  ridge/lasso relative to least squares (answer: less flexible, so an
  improvement exactly when the bias increase is outweighed by the variance
  decrease) — https://guillermomtzdibene.github.io/R/ISL_Chapter6_exercises.html (cross-checked against a second solutions repo)
- ISL ch. 6, Conceptual Exercise 3 — as the lasso budget parameter $s$ grows
  from 0, classify how training RSS, test RSS, variance, squared bias, and
  irreducible error each move (steadily decrease / U-shaped / steadily increase
  / constant) — https://github.com/franciscoyira/islr-exercises/blob/master/ch6.md (fetched)
- Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, Ex. 3.12 —
  ridge regression equals ordinary least squares on $\mathbf{X}$ augmented with
  $\sqrt{\lambda}\mathbf{I}$ rows and $\mathbf{y}$ augmented with zeros —
  https://www.danli.org/2021/03/23/esl-chapter-3-exercises/ (fetched)
- Simon J. D. Prince, *Understanding Deep Learning*, ch. 9 ("Regularization"),
  end-of-chapter Problem 9.1 — given a zero-mean Gaussian prior
  $\mathcal{N}(0,\sigma_\phi^2)$ over the parameters, show that maximizing the
  data likelihood times the prior yields a loss equivalent to $\ell_2$
  regularization — essentially the same MAP derivation as this section's own
  exercise 6, verified directly from the book's own text (local copy,
  `udl_book.txt`, "Problems" following §9's augmentation section)
- Prince, *Understanding Deep Learning*, ch. 9, end-of-chapter Problem 9.5 —
  show that the weight-decay parameter update $\phi\leftarrow(1-\lambda)\phi-\alpha\,\partial L/\partial\phi$
  is equivalent to a plain gradient step on $L+\frac{\lambda}{2\alpha}\sum_k\phi_k^2$
  — exactly the shrink-and-update-equals-penalized-loss equivalence this
  section's own prose *states* but never asks the reader to derive — verified
  directly from the book's own text (same location)
- Prince, *Understanding Deep Learning*, ch. 9, end-of-chapter Problem 9.3
  (starred/harder) — for $y=\phi_0+\phi_1 x$ fit by least squares, inject
  zero-mean Gaussian noise into the inputs $x_i$ at every training iteration
  and derive the expected loss — the classical "training with input noise is
  equivalent to a regularization penalty" result, a nice forward-looking
  preview beyond this section's own $\ell_2$-on-the-weights treatment —
  verified directly from the book's own text (same location)
- Prince, *Understanding Deep Learning*, ch. 9, Notebook
  `9_4_Bayesian_Approach.ipynb` — compute a Gaussian posterior's mean and
  covariance under an explicit prior, then vary the prior variance and explain
  the resulting change — the hands-on companion to Problem 9.1 and to this
  section's exercise 6 — https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap09/9_4_Bayesian_Approach.ipynb (fetched)

**Proposed problem set** (8 problems, our reference format):
1. [short-code] **The $\lambda$ sweep.** Plot training and validation loss as a
   function of $\lambda$ on this section's own 20-example/200-feature rig, and
   report where the U-shaped validation curve bottoms out.
   *Provenance:* original (kept from the book).
1. [short-code] **Is validation-selected $\lambda$ stable?** Using a validation
   split, find the $\lambda$ that minimizes validation loss; then repeat with
   two more random train/validation splits of the same data. Report whether
   the selected $\lambda^*$ is stable across the three splits, and by how much
   validation loss differs between the most frequently selected $\lambda^*$ and
   its runner-up.
   *Provenance:* original (exercise 2 rewritten: the rhetorical "is it really
   optimal? does it matter?" now has a concrete stability metric to report,
   inspired loosely by ISL ch. 6 Ex. 3's practice of naming a specific
   quantity — training RSS, variance, etc. — for every regime change; overlap
   low).
1. [conceptual] **The $\ell_1$ update.** Derive the SGD update equations if the
   penalty is $\sum_i |w_i|$ ($\ell_1$) instead of $\|\mathbf{w}\|^2$.
   *Provenance:* original (kept from the book).
1. [conceptual] **A matrix-norm analogue.** Give the analogue of
   $\|\mathbf{w}\|^2=\mathbf{w}^\top\mathbf{w}$ for matrices, in terms of the
   Frobenius norm.
   *Provenance:* original (kept from the book).
1. [short-code] **Early stopping as a second lever.** Implement early stopping
   on this section's own 20-example/200-feature rig with $\lambda=0$: track
   validation loss each epoch, stop at its minimum, and report the stopping
   epoch and the validation loss achieved. Compare both against the
   $\lambda=0$ (no regularization, full training) and $\lambda=3$ runs already
   in this section.
   *Provenance:* original (exercise 5 rewritten: the open "what other ways
   might help with overfitting?" brainstorm now names one specific method —
   early stopping — and requires implementing and measuring it rather than
   listing ideas).
1. [short-code] **Ridge as augmented least squares.** Implement ridge regression
   two ways on this section's own dataset: (a) the closed form
   $(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$,
   and (b) ordinary least squares on $\mathbf{X}$ stacked with
   $\sqrt{\lambda}\mathbf{I}$ and $\mathbf{y}$ stacked with zeros. Confirm the
   two give matching $\hat{\mathbf{w}}$ up to numerical precision.
   *Provenance:* adapted from ESL Ex. 3.12 (overlap high; cite ESL on
   adoption).
1. [conceptual] **Making MAP precise.** Starting from the posterior
   $p(\mathbf{w}\mid\mathbf{X},\mathbf{y})\propto p(\mathbf{y}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w})$
   with noise variance $\sigma^2$ and prior $\mathbf{w}\sim\mathcal{N}(\mathbf{0},\tau^2\mathbf{I})$,
   show that minimizing the negative log-posterior is equivalent to minimizing
   $L(\mathbf{w},b)+\frac{\lambda}{2}\|\mathbf{w}\|^2$ with
   $\lambda=\sigma^2/(n\tau^2)$, then find the prior standard deviation $\tau$
   corresponding to the $\lambda=3$ used in this section's experiments.
   *Provenance:* original (kept from the book).
1. [conceptual] **Shrinkage equals a penalized loss.** Show that the
   multiplicative weight-decay update $\mathbf{w}\leftarrow(1-\eta\lambda)\mathbf{w}-\eta\nabla L$
   used in this section is algebraically identical to a plain gradient step on
   the penalized loss $L+\frac{\lambda}{2}\|\mathbf{w}\|^2$. Then explain, in
   one or two sentences, why this equivalence breaks for an adaptive optimizer
   like Adam — where in the update does the per-coordinate rescaling enter —
   connecting to this section's AdamW aside.
   *Provenance:* adapted from Prince, *Understanding Deep Learning*, ch. 9,
   Problem 9.5 (overlap high; cite on adoption) — extended with the AdamW
   follow-up question, which UDL's Problem 9.5 does not ask, to tie back to
   this section's own decoupled-weight-decay discussion.
