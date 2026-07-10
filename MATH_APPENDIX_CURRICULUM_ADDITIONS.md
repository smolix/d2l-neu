# Curriculum Additions for the Mathematics Appendix

Status: planning note, audited against the tree on 2026-07-10. An earlier
version of this note proposed a body of additions as future work; a review
found that most of them, including both "highest-priority" notebooks, already
ship in the committed appendix chapters. This revision records what exists,
with pointers, so the plan cannot be executed twice, and keeps only the
genuinely outstanding items as proposals. Do not implement the outstanding
items until the scope has been approved.

The appendix covers the mathematical foundations of deep learning in
considerable depth. Any further additions should not turn it into a survey of
every mathematical topic used in machine learning: each should either complete
an argument the book already begins, explain a method students will encounter
in modern training systems, or connect material currently presented in
separate chapters.

## Already implemented (verified in the tree, 2026-07-10)

### Dedicated notebooks

The two notebooks this note once listed as its highest-priority additions are
written, executed, and committed. Their content matches the original proposals
section for section; do not re-plan or re-write them.

- **Adaptive methods**:
  `chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md` (§24.2).
  Covers the Ghadimi--Lan $O(K^{-1/2})$ nonconvex SGD stationarity rate with
  explicit assumptions, AdaGrad as a diagonal preconditioner, RMSProp, Adam
  with the bias-correction factors derived, the Reddi et al. nonconvergence
  example ("When Adam Fails to Converge") and what AMSGrad changes, decoupled
  weight decay versus an $L_2$ penalty under a preconditioner, schedules and
  warmup with proven claims separated from conventions, and the
  preconditioning ladder to K-FAC, Shampoo, and Muon. The executable
  comparisons on an ill-conditioned quadratic and the derivation-style
  exercises proposed here are present.
- **Concentration and generalization**:
  `chapter_mdl-probability-statistics/mdl-concentration-generalization.md`
  (§25.5). Covers the Chernoff method, Hoeffding's lemma and inequality with
  proof, sub-Gaussian and sub-exponential variables, norm concentration and
  near-orthogonality in high dimension, the union bound and a complete
  finite-class generalization proof, Rademacher complexity with the linear
  class computed in closed form, and interpolation, double descent, and
  benign overfitting (including a random-features double-descent experiment,
  "Double Descent in Twenty-Six Lines", and the Chebyshev/Hoeffding/empirical
  tail comparison, "The Tail Race in Code"). The main generalization chapter
  (`chapter_multilayer-perceptrons/generalization-deep.md`) already points
  here for the substantial double-descent treatment.

### Within existing notebooks

- Implicit function theorem, with
  $\partial\mathbf x^\star/\partial\boldsymbol\theta =
  -(\partial\mathbf g/\partial\mathbf x)^{-1}\,
  \partial\mathbf g/\partial\boldsymbol\theta$ derived, numerically verified,
  and connected to bilevel optimization, deep equilibrium models, and the
  adjoint method: `chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md`.
- Leibniz rule with moving limits, and the Gamma function named where used:
  `chapter_mdl-calculus/mdl-integral-calculus.md` (Gamma also in
  `chapter_mdl-probability-statistics/mdl-distributions.md`).
- Rank--nullity (stated as a proposition), projection onto a subspace with
  $\mathbf Q\mathbf Q^\top$, the least-squares connection, and a QR /
  Gram--Schmidt orthogonality experiment:
  `chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md`.
- Marchenko--Pastur, well beyond a remark (noise baseline for spectra, bulk
  edges, rank selection): `chapter_mdl-linear-algebra/mdl-eigendecomposition.md`
  and `mdl-svd-low-rank.md`.
- Proximal operators as generalized projections, soft thresholding as the
  prox of the $L_1$ norm, and an executable ISTA example:
  `chapter_mdl-optimization/mdl-convexity.md`.
- Lagrangian duality as a saddle point, with the careful pointer to minimax
  training (GANs, adversarial training) and no convex--concave over-claim:
  `chapter_mdl-optimization/mdl-constrained-optimization-duality.md`.
- Summation error, pairwise and Kahan (compensated) summation, and stochastic
  rounding with its bias/variance trade-off, cross-referencing chapter 6 for
  floating-point formats:
  `chapter_mdl-optimization/mdl-numerical-stability-conditioning.md`.
- Entropy rate for a sequence source, the precise conditions for a language
  model to drive an arithmetic coder and which costs the idealized
  cross-entropy calculation excludes, cross-entropy scaling laws presented as
  observed regularities over stated regimes, and MDL / two-part codes in
  further reading: `chapter_mdl-information-theory/mdl-information-theory.md`
  and the chapter's index Resources list.
- Proper scoring rules (log score, Brier score, strict propriety):
  `chapter_mdl-information-theory/mdl-information-theory.md`.
- Rényi divergences compared with KL where divergence families are discussed:
  `chapter_mdl-information-theory/mdl-divergences-distances.md`.
- Calibration: the population conditions under which log loss is strictly
  proper, and why finite data, misspecification, regularization, optimization
  error, and shift can still yield miscalibration, in
  `mdl-information-theory.md`; a reliability diagram and a quantitative
  confidence-versus-accuracy audit in
  `chapter_mdl-probability-statistics/mdl-naive-bayes.md` ("Calibration").
- Bootstrap confidence interval for the naive Bayes accuracy (with the paired
  comparison caveat): `mdl-naive-bayes.md`.
- The parameterization dictionary for Gaussian paths (score, noise, $x_0$,
  velocity), the log-SNR clock, and which transformations are a change of
  time versus a change of objective:
  `chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md`.
- Gradient descent as forward Euler on the gradient flow, the learning-rate
  stability limit on a quadratic, and momentum as a discretization of the
  heavy-ball ODE: `chapter_mdl-dynamics/mdl-odes-solvers.md` (heavy-ball also
  in `chapter_mdl-optimization/mdl-gradient-based-optimization.md`).
- Attention mathematics, threaded through existing chapters as proposed: the
  softmax Jacobian with an attention-Jacobian exercise in
  `mdl-matrix-calculus-autodiff.md`, log-sum-exp / softmax conjugacy in
  `mdl-convexity.md`, and the interacting-particle outlook references in
  `chapter_mdl-dynamics`.
- The KSG mutual-information estimator, implemented with code rather than
  left as a reading item:
  `chapter_mdl-information-theory/mdl-mutual-information.md`.

Optimal transport has enough coverage in the appendix and does not need a new
section.

## Outstanding proposals

These are the items from the original plan that are genuinely absent from the
tree. All are small, independently reviewable edits to existing notebooks; no
new notebook is needed.

1. **Metropolis-adjusted Langevin (MALA).** Extend the Langevin /
   predictor--corrector discussion in
   `chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md` with the
   Metropolis--Hastings correction, showing how it restores the target
   stationary distribution after discretization. A general survey of MCMC
   still belongs in a later edition.
2. **Attention scaling from near-orthogonality.** Connect the
   near-orthogonality results in
   `chapter_mdl-probability-statistics/mdl-concentration-generalization.md`
   to the $1/\sqrt{d}$ scaling of dot-product attention, being precise about
   the distributional assumptions. A remark or short subsection, not a new
   section.
3. **Variational inference paragraph.** A short extension of the existing
   ELBO discussion (`mdl-mutual-information.md`, `mdl-divergences-distances.md`)
   with a pointer to the VAE chapter.
4. **Reading-list and exercise material only** (a short remark, exercise, or
   reference, no full treatment): $\mu$P, tensor programs, and
   width-dependent parameter scaling; neural tangent kernels; broader
   approximation theory; CUR decompositions and nuclear-norm methods; the
   delta method; fuller treatments of MCMC and variational inference.

## Editorial requirements

Any later implementation should follow these constraints:

1. Introduce every mathematical object before using it. State hypotheses next
   to the theorem or proposition that needs them.
2. Separate exact results, asymptotic results, modeling assumptions, and
   empirical observations. Avoid presenting a useful heuristic as a theorem.
3. Use executable cells to test central quantitative claims. A plot should
   answer a stated question rather than merely decorate the text.
4. Keep examples small enough to execute on CPU. New framework-specific code
   should support PyTorch, TensorFlow, JAX, and MXNet; MXNet remains a
   co-equal tab, and where a framework lacks a needed piece, a reduced
   variant with an honest tab note is acceptable.
5. Avoid repeating material from the main book. Replace duplicate explanations
   with a short recall and a precise cross-reference.
6. Cite primary sources for modern results and a stable textbook source for
   established theory. Scaling-law and optimizer claims need dates and scope.
7. Add exercises that require derivation, diagnosis, or interpretation, not
   only substitution into a displayed formula.
8. Update slides only after the notebook prose and executable examples have
   passed review.

## Suggested order of work

When curriculum work resumes:

1. Add the MALA and attention-scaling items as small, separately reviewed
   edits to their host notebooks.
2. Add the variational-inference paragraph.
3. Add reading-list items last, only after the substantive additions have
   settled.

This ordering does not authorize implementation. It records the remaining
scope so that the additions can be evaluated before they alter the textbook.
