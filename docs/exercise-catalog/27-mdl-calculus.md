# Chapter overview — chapter_mdl-calculus (4 files)

All four sections already carry unusually strong exercise sets (the prior style
review found zero defects in three of the four files, and only a cosmetic
spacing nit in the fourth). This is the rare chapter where the job is mostly
*addition and reorganization*, not repair. Best external match by far: MIT
18.S096 "Matrix Calculus for ML and Beyond" (Edelman & Johnson) — its two
problem sets are already pitched at this book's altitude and supply a
genuinely new identity (Hellman–Feynman) the section never proves. MIT
18.02 (Fall 2007) problem sets are the best source for multivariable-calculus,
which is the one file lacking any [short-code] exercise despite having code.
MIT 18.01SC supplies a classic Newton's-method failure case and a numerical-
integration comparison. CS231n's backprop notes and Boyd & Vandenberghe's
convex-optimization exercises each contribute one framing idea, not a
transplantable problem. Coverage gap, confirmed rather than assumed: none of
18.01/18.02/18.S096 touches stochastic gradient estimators (score-function/
pathwise), subgradients for training, or descent-lemma convergence analysis —
these are ML-native material with no classical-calculus counterpart. Totals:
37 current exercises across 4 sections, disposition keep 32 / rewrite 2 / drop
3 (some "keep" originals are merged pairs), 31 problems in the 4 proposed
sets (7–8 each).

## chapter_mdl-calculus/mdl-single-variable-calculus.md — Single Variable Calculus

**Topic:** The derivative as local linear model; differentiation rules;
gradient descent and the descent lemma; curvature, Taylor series, Newton's
method; nondifferentiable points and subgradients.

**Current exercises:** 10; disposition: keep 7, rewrite 2, drop 1 — the set is
already excellent (the style review found zero defects); we drop only ex8 (an
informal, un-cross-referenced restatement of material the vector-descent-lemma
derivation in ex5 already covers better) and rewrite two (ex9, ex10) to close
gaps the section's own Summary flags but never tests: autograd failing to
return a valid subgradient through a composition, and the error-halving rate
the section's own Lagrange-error code cell demonstrates but never checks
against a Taylor-series exercise.

**External sources found:**
- MIT OpenCourseWare, 18.01SC Single Variable Calculus (Fall 2010), Session 33
  problem "Cube Root of $x$" — shows Newton's method never converges to the
  root of $x^{1/3}$ from any nonzero starting point, the classic divergence
  counterexample to the section's own (convergent) Newton's-method demo —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/resources/mit18_01scf10_ex33prb/
- MIT OpenCourseWare, 18.01SC Single Variable Calculus, Unit 2 "Optimization,
  Related Rates, and Newton's Method," Sessions 29–30 — general context for
  how the classical course scaffolds optimization/Newton's-method problems
  (background only, no exercise adopted directly) —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/part-b-optimization-related-rates-and-newtons-method/session-29-optimization-problems/
- Stephen Boyd and Lieven Vandenberghe, *Additional Exercises for Convex
  Optimization* (used in Stanford EE364a, UCLA EE236b, MIT 6.975), Exercise
  9.1 "Gradient descent and nondifferentiable functions" — a convex,
  nondifferentiable $f(x_1,x_2)$ on which exact-line-search gradient descent
  provably converges to $(0,0)$, which is *not* a minimizer since $f$ is
  unbounded below —
  https://alexandreamice.github.io/teaching/convex_optimization/6.S098_homeworks/boyd_extra_exercises.pdf
  (PDF, Exercise 9.1, p. 128 of the January 2022 revision)
- This section's most distinctive material — the descent lemma /
  $L$-Lipschitz convergence analysis (ex5–7) and the ML framing of
  subgradients through ReLU/hinge losses (ex9) — has essentially NO external
  exercise tradition: these are inventions of modern optimization theory, not
  standard 18.01 fare. A genuine finding, not a search failure.

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **Derivative Rules Audit.** Compute the derivatives of
   $x^3-4x+1$ and $\log(1/x)$ directly from the rules table; separately,
   derive the quotient rule from the product and chain rules (writing
   $g/h = g\cdot h^{-1}$) and check it reproduces
   $\frac{d}{dx}\tan x = 1/\cos^2 x$. Deliverable: both derivations plus the
   numeric check on $\tan x$.
   *Provenance:* original (merges current ex1 and ex2).
2. [conceptual] **Stationary Points Aren't Enough.** Decide whether $f'(x)=0$
   implies a max or min (give a counterexample and the test that resolves it),
   then find the minimum of $f(x)=x\log x$ on $x\ge 0$. Deliverable: the
   counterexample, the distinguishing test, and the located minimum with
   justification.
   *Provenance:* original (merges current ex3 and ex4).
3. [conceptual] **Vector Descent Lemma.** Starting from the small-change
   identity, derive the first-order descent prediction
   $f(x-\eta f'(x)) \approx f(x) - \eta[f'(x)]^2$, name the discarded term,
   then extend the descent lemma :eqref:`eq_mdl-descent` to
   $f:\mathbb R^n\to\mathbb R$ with an $L$-Lipschitz gradient. Deliverable: the
   vector-case inequality with a complete Cauchy–Schwarz proof along the
   segment $\mathbf x + t\mathbf s$.
   *Provenance:* original (current ex5, unchanged).
4. [conceptual] **Step-Size Regimes for Quadratic Descent.** For $f(x)=x^2$,
   characterize every step size $\eta$ for which gradient descent from
   $x_0\ne0$ converges, find the single $\eta$ reaching the minimum in one
   step, and separately exhibit a step size for which one gradient step
   *increases* $f$, naming which hypothesis of the descent lemma fails.
   Deliverable: the convergence range, the one-step $\eta$, and the failure
   example with diagnosis.
   *Provenance:* original (merges current ex6 and ex7).
5. [short-code] **Newton's Method on a Cube Root.** Implement Newton's method
   with the section's own autograd (as in the $x^4/4-x$ example) applied to
   $f(x)=x^{1/3}$, run it from several nonzero starting points, and show
   numerically that the iterates diverge (oscillating with growing
   magnitude) rather than converging to the root at $0$. Deliverable: a
   short script/plot of iterate trajectories for at least 3 starting points,
   contrasted against the section's convergent example.
   *Provenance:* adapted from MIT 18.01SC "Cube Root of $x$" (overlap high on
   the underlying mathematical fact; the autograd implementation and the
   contrast with the book's own convergent example are new).
6. [conceptual] **Subgradients Through Composition.** Compute
   $\partial\,\mathrm{ReLU}(0)$ and $\partial|x|(0)$, identify which of
   $\{0, 0.5, 1\}$ are valid subgradients of $\mathrm{ReLU}$ at $0$, sketch
   the subdifferential of the hinge loss $\max(0,1-x)$, and then — using the
   Summary's own claim — construct a two-function composition where
   autograd's fixed return value $\mathrm{ReLU}'(0)=0$ fails to be a valid
   subgradient of the *composition* at the shared kink. Deliverable: the
   three subdifferentials/sketch plus one explicit composition
   counterexample.
   *Provenance:* adapted from d2l's own Summary bullet (overlap high on the
   original ex9 parts; the composition-failure counterexample is new,
   prompted by the caveat in Boyd & Vandenberghe Exercise 9.1 that nonsmooth
   descent needs extra structural assumptions).
7. [short-code] **Taylor Error Halving, Verified.** Use the degree-5 Taylor
   polynomial of $e^x$ at $x_0=0$ to estimate $e$ and compare to the true
   value; then, using the section's own Lagrange-error code cell, measure the
   degree-$n$ truncation error at window half-widths $h$ and $h/2$ for two
   different $n$, and confirm the error ratio is close to $2^{n+1}$ as
   predicted. Deliverable: the estimate of $e$, plus a small table of
   measured vs. predicted error ratios.
   *Provenance:* adapted from d2l (current ex10, extended with a numerical
   verification using the section's existing Lagrange-error code).

## chapter_mdl-calculus/mdl-multivariable-calculus.md — Multivariable Calculus

**Topic:** Partial derivatives and the gradient; steepest-descent geometry
and level sets; the multivariate chain rule as backpropagation; the Hessian
and the second-derivative test; Lagrange multipliers.

**Current exercises:** 8; disposition: keep 6, drop 2 — the set is good and
entirely conceptual (the only style defect anywhere in this chapter is a
missing blank line after this file's heading), but it is the one section
with NO [short-code] exercise despite a full worked forward/backward pass, a
4-framework autograd check, and a Hessian-eigenvalue classification cell. We
drop two thin/redundant items to make room for two code-based additions.

**External sources found:**
- MIT OpenCourseWare, 18.02 Multivariable Calculus (Fall 2007), Problem Set
  3, Problem 3 — a contour plot of
  $f(x,y)=x^3-xy^2-4x^2+3x+x^2y$; read the sign of $f_x,f_y$ off the
  contours, confirm numerically, and locate the points where
  $\nabla f = \mathbf 0$ —
  https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/resources/ps3/
- MIT OpenCourseWare, 18.02 Multivariable Calculus (Fall 2007), Problem Set
  4, Problem 2 — a triangle inscribed in the unit circle with vertices at
  polar angles $\theta_1,\theta_2$: find the critical points of the area
  function, classify them via the second-derivative test, and confirm
  against the max/min found by checking the boundary —
  https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/resources/ps4/
- MIT OpenCourseWare, 18.02 Multivariable Calculus (Fall 2007), Problem Set
  4, Problem 3 — derive the polar-to-rectangular change-of-variables formula
  for partial derivatives as an explicit $2\times2$ matrix chain-rule product
  $A$, derive the inverse matrix $B$ independently, and verify $AB=I$ —
  https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/resources/ps4/
- Stanford CS231n, "Backpropagation" notes (Karpathy) — decomposes a
  multi-stage scalar expression into a computational graph and states three
  reusable "patterns in backward flow" (add distributes the gradient, max
  routes it to the larger input, multiply swaps and scales), recommending
  small explicit hand examples before vectorizing —
  https://cs231n.github.io/optimization-2/

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Gradient of a Log-Sum-Exp.** For $L(x,y)=\log(e^x+e^y)$,
   compute $\nabla L$, verify its components always sum to $1$, and interpret
   what that invariant says about the directions in which $L$ grows fastest.
   Deliverable: the gradient formula and the interpretation.
   *Provenance:* original (current ex1, unchanged).
2. [short-code] **Reading a Contour Plot.** Using the section's own plotting
   tools, draw the contour plot of $f(x,y)=x^3-xy^2-4x^2+3x+x^2y$ over
   $[0,2]^2$, read the sign of $f_x,f_y$ at $(1,1.5)$ and $(1.2,0.6)$
   directly off the contours, then compute the partials analytically and
   confirm; finally locate the two points where $\nabla f=\mathbf 0$ and
   describe what the level curves do there. Deliverable: the contour plot,
   the sign predictions vs. computed values, and the two critical points.
   *Provenance:* adapted from MIT 18.02 PS3 Problem 3 (overlap high — same
   function; we add the plotting step since the book has its own tools where
   MIT's exercise used a course applet).
3. [conceptual] **Orthogonal to the Level Curve.** For $f(x,y)=x^2+2y^2$,
   compute $\nabla f$ and verify at a sample point on the ellipse $f=c$ that
   the gradient is orthogonal to the level curve. Deliverable: the gradient
   and the orthogonality check at one concrete point.
   *Provenance:* original (current ex2, unchanged).
4. [conceptual] **Classifying Critical Points.** Show the critical points of
   $f(x,y)=x^3-3x+y^2$ are $(\pm1,0)$ and classify each via the Hessian;
   separately, classify the critical point of $g(x,y)=x^2-y^2$ from its
   (constant) Hessian's eigenvalues and explain why it is a saddle.
   Deliverable: both classifications with eigenvalue justification.
   *Provenance:* original (merges current ex4 and ex5).
5. [conceptual] **A Silent Second-Derivative Test.** Construct a two-variable
   function whose Hessian at a critical point is positive semidefinite (one
   zero eigenvalue) even though the point is not a local minimum, and explain
   why the second-derivative test cannot resolve this case. Deliverable: the
   function, the Hessian, and the argument for why the test goes silent.
   *Provenance:* original (current ex6, unchanged).
6. [conceptual] **Inscribed-Triangle Optimization.** For a triangle inscribed
   in the unit circle with one vertex fixed at $(1,0)$ and the others at
   polar angles $\theta_1,\theta_2$, express its area $A(\theta_1,\theta_2)$,
   find the critical points, classify them with the second-derivative test,
   and separately check the boundary of the valid region to identify the
   true max and min. Deliverable: $A(\theta_1,\theta_2)$, the critical points
   with their classification, and a description of the triangle shapes at
   the max and min.
   *Provenance:* adapted from MIT 18.02 PS4 Problem 2 (overlap high on the
   setup and required steps; framed here purely with calculus rather than
   the original's Matlab data-fitting companion problem).
7. [short-code] **Backprop by Hand, Then by Autograd.** Using the section's
   own worked example (the forward/backward pass with tabbed autograd
   checks), hand-derive each local partial in the staged computation, predict
   how the gradient flows through the add/multiply/branch operations using
   the "add distributes, multiply swaps-and-scales" patterns, then run the
   section's own autograd cell to confirm every hand-derived partial matches.
   Deliverable: the hand-derived partials next to the autograd output, one
   row per intermediate variable.
   *Provenance:* inspired by CS231n backpropagation notes (overlap low — the
   specific computational graph is the book's own; only the "patterns in
   backward flow" framing is borrowed).
8. [conceptual] **Lagrange Multipliers and Shadow Prices.** Use the Lagrange
   condition :eqref:`eq_mdl-lagrange-condition` to maximize $f(x,y)=xy$
   subject to $g(x,y)=x+y=1$, verify $\nabla g\ne\mathbf 0$ first, solve by
   substitution as a check, then perturb the constraint to $x+y=1+\delta$ and
   show the optimal value changes by $\lambda\delta$ to first order.
   Deliverable: the maximizer, the substitution check, and the first-order
   sensitivity result.
   *Provenance:* original (current ex8, unchanged).

## chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md — Matrix Calculus and Automatic Differentiation

**Topic:** Jacobians as best linear approximations; the chain rule as
Jacobian composition; forward-mode AD via dual numbers; reverse-mode AD via
the tape (backprop); Hessian-vector products; implicit differentiation.

**Current exercises:** 9; disposition: keep 9, rewrite 0, drop 0 — this is the
single strongest exercise set in the chapter (already dense, code-integrated,
and citing the section's own framework snippets); we only reorganize via two
thematic merges to fit the 5–8-problem format and add one genuinely new item
for a gap the section's own body creates but never tests: the
implicit-function-theorem material ("Differentiating Through Equations") and
the eigenvalue/Hellman–Feynman identity that MIT's companion course tests
directly.

**External sources found:**
- MIT OpenCourseWare, 18.S096 Matrix Calculus for Machine Learning and
  Beyond (Edelman & Johnson, IAP 2023), Problem Set 1 — Jacobians of 2D
  image transforms (rotation, hyperbolic rotation, nonlinear shear), the
  derivative of $f(A)=A^\top$ and $\mathrm{tr}\,A$ as instances of a general
  linear-operator differentiation rule, and Jacobians/gradients of
  $\mathbf x^\top(\mathbf A+\mathrm{diag}(\mathbf x))^2\mathbf x$ and
  $(\mathbf A+\mathbf y\mathbf x^\top)^{-1}\mathbf b$ (the latter tied to the
  Sherman–Morrison low-rank-update formula) —
  https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset1.pdf
- MIT OpenCourseWare, 18.S096 Matrix Calculus for Machine Learning and
  Beyond, Problem Set 2 — Problem 1 asks for $\nabla_{\mathbf p} f$ of
  $f(\mathbf p)=(\mathbf c^\top A(\mathbf p)^{-1}\mathbf b)^2$ where
  $A(\mathbf p)$ is tridiagonal, computed via *two* linear solves (a primal
  solve and an "adjoint" solve) rather than by forming $A^{-1}$ — a direct,
  concrete instance of the section's own implicit-differentiation/adjoint
  material; Problem 3 derives the Hellman–Feynman theorem
  $d\lambda = \mathbf q^\top d\mathbf S\,\mathbf q$ for a simple eigenvalue
  of a symmetric matrix —
  https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf
- CMU 10-714 / dlsyscourse.org, "Deep Learning Systems," Homework 1
  "Automatic differentiation framework" — students incrementally build the
  reverse-mode autodiff engine of their own Needle library over a
  computational graph, the same tape-and-replay idea as the section's toy
  `Var`/`backprop` implementation —
  https://dlsyscourse.org/ (homework list confirmed; the per-offering
  assignment text itself sits behind a Colab/GitHub link we did not fetch,
  so no specific sub-problem is quoted)
- JAX, "The Autodiff Cookbook" — documents computing Hessian-vector products
  via forward-over-reverse for memory efficiency (matching the section's
  Pearlmutter's-trick recipe) and gradient checkpointing for trading
  recomputation against activation memory —
  https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
- The Matrix Cookbook (Petersen & Pedersen) — the standard desktop reference
  for exactly the identities this section derives from first principles
  ($\nabla\mathrm{tr}$, derivatives of determinants/inverses/eigenvalues); we
  cite it as the reference this section's "Key Matrix-Derivative Identities"
  subsection implicitly competes with, not as a source of new exercises,
  since it is a lookup table rather than a problem set —
  http://www.gatsby.ucl.ac.uk/~balaji/ml/files/matrix_cookbook.pdf

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Jacobian as Linear Approximation.** Compute the Jacobian of
   $\mathbf f(x,y)=(x^2y,\sin(x+y))$, verify the linear approximation against
   a finite difference at $(1,0)$, and report how much the error shrinks when
   the perturbation is halved. Deliverable: the Jacobian, the finite-
   difference check, and the observed error-halving rate.
   *Provenance:* original (current ex1, unchanged).
2. [conceptual] **Two Ways to a Quadratic Gradient.** Derive
   $\nabla_{\mathbf x}\|\mathbf A\mathbf x-\mathbf b\|_2^2=2\mathbf A^\top(\mathbf A\mathbf x-\mathbf b)$
   twice: once by an index/Einstein-notation expansion, and once by the
   scalar-collapse heuristic (guess the 1D form, then fix shapes with
   transposes). Deliverable: both derivations side by side.
   *Provenance:* original (current ex2, unchanged).
3. [conceptual] **Counting AD Passes.** For a scalar loss with $n=10^6$
   parameters, state how many passes forward mode needs to assemble the full
   gradient and how many reverse mode needs, explaining both counts in terms
   of JVPs vs. VJPs. Deliverable: the two pass counts with a one-paragraph
   justification.
   *Provenance:* original (current ex3, unchanged).
4. [short-code] **Extending Both Toy AD Engines.**
   1. Add a `__pow__` (or `log`) method to the section's forward-mode `Dual`
      class and use it to differentiate $f(x)=\log(1+x^2)$ at $x=2$, checking
      against the analytic derivative and naming which rule from the proof
      of :eqref:`eq_mdl-dual-eval` each new method encodes.
   1. Add a new primitive (e.g. $\exp$ or $\sin$) to the section's
      reverse-mode `Var`/`backprop` tape, supply its `_backward`, and verify
      the resulting gradient against a framework's autograd.

   Deliverable: both extended classes plus their numeric checks.
   *Provenance:* original (merges current ex4 and ex5).
5. [short-code] **Softmax and Attention Jacobians.**
   1. Derive the softmax Jacobian :eqref:`eq_mdl-softmax-jacobian` from the
      quotient rule and re-derive the logit gradient $\mathbf p-\mathbf y$,
      explaining why fusing softmax with cross-entropy is numerically
      preferable to composing them.
   1. In self-attention, show the Jacobian of row-wise softmax attention
      weights $\mathbf P$ with respect to the scores
      $\mathbf S=\mathbf Q\mathbf K^\top/\sqrt d$ is block-diagonal with
      per-row blocks $\mathrm{diag}(\mathbf p_i)-\mathbf p_i\mathbf p_i^\top$,
      extend the chain to $\partial\mathbf p_i/\partial\mathbf q_i$, discuss
      where the $1/\sqrt d$ ends up, and verify the block-diagonal structure
      numerically with a framework's `jacobian` on a $3\times3$ score matrix.

   Deliverable: both derivations plus the numerical Jacobian check.
   *Provenance:* original (merges current ex6 and ex9).
6. [short-code] **Hessian-Vector Product.** For
   $L(\mathbf x)=\tfrac12\mathbf x^\top\mathbf A\mathbf x$, show
   $\mathbf H\mathbf v=\mathbf A\mathbf v$ analytically, then implement the
   forward-over-reverse recipe of :eqref:`eq_mdl-hvp` with a framework's
   autograd and confirm it returns $\mathbf A\mathbf v$ without ever forming
   $\mathbf H$; contrast against the double-backward (reverse-over-reverse)
   alternative's cost. Deliverable: the analytic result, the
   forward-over-reverse implementation, and a brief cost comparison.
   *Provenance:* original (current ex7, unchanged).
7. [short-code] **Gradient of the Log-Determinant.** Show
   $\nabla_{\mathbf A}\log|\det\mathbf A|=\mathbf A^{-\top}$ using the
   perturbation/Jacobi's-formula argument, then verify it numerically for a
   random invertible $3\times3$ matrix by differentiating a
   `slogdet`-style routine with a framework's autograd. Deliverable: the
   derivation and the numeric check, with a one-line note on why this
   identity is what makes normalizing flows trainable by gradient descent.
   *Provenance:* original (current ex8, unchanged).
8. [conceptual] **Hellman–Feynman: Gradient of an Eigenvalue.** For a real
   symmetric matrix $\mathbf S$ with a simple eigenvalue $\lambda$ and unit
   eigenvector $\mathbf q$ ($\mathbf S\mathbf q=\lambda\mathbf q$), derive the
   Hellman–Feynman identity $d\lambda=\mathbf q^\top d\mathbf S\,\mathbf q$ by
   differentiating both $\lambda=\mathbf q^\top\mathbf S\mathbf q$ and
   $\mathbf q^\top\mathbf q=1$, then give the gradient $\nabla_{\mathbf S}\lambda$
   under the Frobenius inner product. Deliverable: the derivation and the
   gradient formula, plus one sentence on why this is cheaper than
   differentiating through an eigendecomposition solver.
   *Provenance:* adapted from MIT 18.S096 PSet 2, Problem 3 (overlap high —
   same theorem and derivation route; we add the framing connecting it to
   differentiating through a solver, echoing the section's own
   implicit-function-theorem material).

## chapter_mdl-calculus/mdl-integral-calculus.md — Integral Calculus

**Topic:** The definite integral and the FTC; improper integrals,
integration by parts, change of variables; multiple integrals and Fubini;
densities/expectations/Monte Carlo; differentiating under the integral sign
(score-function and pathwise gradient estimators).

**Current exercises:** 10; disposition: keep 10, rewrite 0, drop 0 — the
style review found this the most well-scaffolded set in the chapter (e.g.
ex8's Fubini "paradox" resolved with a concrete hint, ex10's
integration-by-parts-to-Gamma-function chain). We only merge thematically
adjacent pairs to fit the 5–8-problem format; no external source improves on
the two research-grade probability-gradient exercises (ex1, ex2), which we
keep verbatim.

**External sources found:**
- MIT OpenCourseWare, 18.01SC Single Variable Calculus (Fall 2010), Unit 3
  Exercises, Section 3G "Numerical Integration" — compares left-endpoint
  Riemann sums, the trapezoidal rule, and Simpson's rule against exact
  values on the same integrals, and asks which hypotheses on $f$ make the
  trapezoidal estimate systematically too high or too low —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/eaabc6c74a11402341dbc120fffa5ea5_MIT18_01SC_pset3prb.pdf
- MIT OpenCourseWare, 18.01SC Single Variable Calculus, Unit 3 Exercises,
  Problem 3C-4 — evaluate $\lim_{b\to\infty}\int_1^b x^{-10}\,dx$ and state
  what area it describes, the same improper-integral-as-a-limit idea as the
  section's own $p$-integral exercise —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/eaabc6c74a11402341dbc120fffa5ea5_MIT18_01SC_pset3prb.pdf
- MIT OpenCourseWare, 18.01SC Single Variable Calculus, Unit 3 Exercises,
  Problems 3E-6/3E-7 — establish integral inequalities (e.g.
  $\int_0^1 dx/(1+x^3) > 0.65$) by comparison with an easier integral rather
  than exact evaluation — a bounding technique the section's own
  Monte-Carlo-vs-quadrature exercises implicitly rely on —
  https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/eaabc6c74a11402341dbc120fffa5ea5_MIT18_01SC_pset3prb.pdf
- Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih, "Monte
  Carlo Gradient Estimation in Machine Learning" (JMLR 2020 / arXiv:1906.10652)
  — the standard survey unifying the score-function (REINFORCE) and pathwise
  estimators and their variance/baseline theory that this section's own
  exercises (ex1, ex2) already implement in miniature; cited to confirm those
  two exercises are already at the research frontier, not to supply a new
  problem — https://arxiv.org/abs/1906.10652
- This section's probability-of-gradients material (ex1, ex2: score-function
  vs. pathwise estimators, optimal baselines) has NO counterpart in either
  18.01 or 18.02 — a genuine finding: classical calculus courses integrate
  deterministic functions and never differentiate an expectation.

**Proposed problem set** (8 problems, our reference format):
1. [short-code] **Score-Function vs. Pathwise Gradients.** For
   $X\sim\mathcal N(\theta,1)$ and $g(X)=X^4$, derive both the pathwise and
   score-function gradient estimators of $\frac{d}{d\theta}\mathbb E[g(X)]$,
   verify analytically that both have mean $4\theta^3+12\theta$, and compare
   their Monte Carlo variances in NumPy. Deliverable: both derivations and a
   variance comparison plot or table.
   *Provenance:* original (current ex1, unchanged).
2. [conceptual] **The Baseline Trick.** Show that replacing $g(X)$ with
   $g(X)-b$ in the score-function estimator
   :eqref:`eq_mdl-score-function-gradient` leaves the expected gradient
   unchanged for any $b$ independent of $X$, then derive the scalar baseline
   minimizing the estimator's second moment. Deliverable: the invariance
   proof and the optimal-baseline formula.
   *Provenance:* original (current ex2, unchanged).
3. [short-code] **Three Ways to Estimate an Integral.**
   1. Evaluate $\int_1^2\frac{1}{x}dx$ via an antiderivative, then confirm
      with a Riemann sum.
   1. Estimate $\int_0^1 e^{-x^2}dx$ by Monte Carlo (averaging $e^{-x_i^2}$
      over uniform samples) and separately by a Riemann sum, and compare how
      each estimator's error shrinks as the sample/interval count grows.

   Deliverable: the exact value from part 1, and from part 2 an
   error-vs-$n$ comparison between the two estimators.
   *Provenance:* original (merges current ex3 and ex9).
4. [conceptual] **Change of Variables, One and Two Dimensions.** Use the
   change-of-variables formula to evaluate $\int_0^{\sqrt\pi} x\sin(x^2)\,dx$;
   separately, evaluate $\int_{[0,1]^2}xy\,dx\,dy$ by Fubini's theorem.
   Deliverable: both evaluated integrals with the substitution/order of
   integration shown explicitly.
   *Provenance:* original (merges current ex4 and ex5).
5. [short-code] **Convergence Boundary of an Improper Integral.** Determine
   for which $p$ the integral $\int_1^\infty x^{-p}\,dx$ converges, then
   verify the boundary case $p=1$ numerically by tracking the partial
   integrals $\int_1^B x^{-1}dx$ as $B$ grows and showing they diverge
   logarithmically rather than converging. Deliverable: the convergence
   condition and a plot/table of the diverging partial integrals at $p=1$.
   *Provenance:* original (current ex6, unchanged).
6. [conceptual] **Normalizing a Gaussian-Shaped Density.** Find the constant
   $c$ making $c\,e^{-x^2}$ a probability density on $\mathbb R$, then compute
   its mean and $\mathbb E[X^2]$. Deliverable: $c$, the mean, and
   $\mathbb E[X^2]$.
   *Provenance:* original (current ex7, unchanged).
7. [conceptual] **A Fubini Paradox.** For
   $f(x,y)=(x^2-y^2)/(x^2+y^2)^2$ on $[0,1]^2$, compute both iterated
   integrals and show they equal $-\pi/4$ and $+\pi/4$ respectively; explain,
   using the hint that $\partial_x\frac{-x}{x^2+y^2}=f(x,y)$, why this does
   not contradict Fubini's theorem. Deliverable: both iterated integrals and
   the absolute-integrability argument that resolves the apparent
   contradiction.
   *Provenance:* original (current ex8, unchanged).
8. [conceptual] **Gamma Function by Parts.** Use integration by parts
   :eqref:`eq_mdl-parts` twice to evaluate $\int_0^\infty u^2e^{-u}du=\Gamma(3)$,
   then run the same integration-by-parts step at general $t$ to derive the
   recursion $\Gamma(t+1)=t\,\Gamma(t)$ and hence $\Gamma(n+1)=n!$.
   Deliverable: the evaluated integral and the general recursion with its
   factorial corollary.
   *Provenance:* original (current ex10, unchanged).
