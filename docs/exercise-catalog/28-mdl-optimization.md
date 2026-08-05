# Chapter Overview: chapter_mdl-optimization

This is the strongest-specified chapter surveyed so far: the prior style review
found zero clarity flags across all 44 exercises in the group's 5 files, and
every exercise already closes with an explicit derivation-plus-numerical-check
structure. The disposition below is therefore keep-heavy, per the rubric's
guidance for already-excellent chapters. Best external match overall is Boyd &
Vandenberghe's own **"Additional Exercises for Convex Optimization"** (2010):
its unconstrained-minimization chapter (§8) and duality chapter (§4) read like
an alternate draft of this chapter's own Newton/quasi-Newton and KKT/duality
exercises, down to the Kantorovich inequality via a simplex-constrained KKT
derivation (Exercise 4.14) — a near-perfect extension of the chapter's own
eigenvalue-conditioning theme. **CMU 10-725** (Fall 2018, Tibshirani) is the
best homework-format match for proximal operators, the $O(1/k)$ rate proof,
and SVM duality. **MIT 18.335** (Johnson) Problem Set 1 directly assigns
Trefethen & Bau's floating-point exercise 13.2 alongside a pairwise-summation
error-bound proof that upgrades this chapter's own pairwise-merge exercise.
**Stanford CS231n** Assignment 2 is the best match for adaptive-stochastic
methods, but in a different register — implement-and-race optimizers from
scratch — complementing rather than duplicating this chapter's derivation-heavy
exercises. Real gaps with **no external exercise tradition found**: K-FAC/
Shampoo/Muon-style structured preconditioners (too recent for problem sets),
Higham's and Nesterov's own textbook exercises (used by the section as proof
references, not posted online as assignments), and a specific Nocedal & Wright
exercise for BFGS/trust-region (the book's own ex10/ex11 already read as
faithful restatements of its Ch. 4/6, so no numbered source is cited). Because
existing exercises cross-reference each other by number across sister files,
every addition below is appended at the end rather than interleaved, and
gradient-based-optimization.md — already the group's largest set at 11 items —
gets no addition at all.

---

## chapter_mdl-optimization/mdl-gradient-based-optimization.md — Gradient-Based Optimization

**Topic:** descent directions and the descent lemma; the condition number and
per-mode contraction on quadratics; momentum and the $\sqrt\kappa$ law;
stochastic gradients and the noise ball; Newton's method, BFGS, and trust
regions.

**Current exercises:** 11; disposition: keep 10, rewrite 1, drop 0 — every item
pairs a derivation with a concrete numerical or structural check and the
review found no clarity issues anywhere in the file; the one rewrite is a pure
formatting fix (a stray blank line isolating Exercise 10) already flagged by
the prior style review.

**External sources found:**
- Boyd & Vandenberghe, "Additional Exercises for Convex Optimization" (Apr. 2010), §8 "Unconstrained and equality constrained minimization," Ex. 8.1–8.2 — generate a random least-squares instance and compare gradient descent, steepest descent, and Newton's method by flop count rather than iteration count, recasting this section's own $\kappa$-vs-iterations argument as a wall-clock/flop argument — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- Same source, Ex. 8.4 "Newton method for approximate total variation de-noising" — implement Newton's method with backtracking line search on a smoothed TV objective and verify quadratic convergence by plotting the Newton decrement against iteration, a concrete non-quadratic test case this section's own Newton exercise (ex8) does not supply — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- Same source, Ex. 8.3 and 8.5 — exploit a banded or low-rank Hessian to solve the Newton system in far less than $O(n^3)$, the same "exploit Hessian structure" argument as this section's ex8 (dense vs. truncated-CG Newton), via a different structural exploit — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- CMU, 10-725 "Convex Optimization" (Fall 2018, R. Tibshirani), Homework 2, Q3 "Convergence Rate for Proximal Gradient Descent" — proves the $O(1/k)$ descent-lemma rate for gradient descent step by step (monotone decrease, then telescoping distance) — the field's standard derivation, confirming rather than upgrading this section's own ex1/ex6 — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework2.pdf
- **No specific numbered exercise verified** for BFGS/secant-equation or trust-region content in Nocedal & Wright's *Numerical Optimization*, the book most naturally associated with ex10/ex11: course pages and a solution-manual listing were found but no exercise text could be fetched and confirmed, so none is cited — a finding, not a failure, since ex10/ex11 already read as faithful, self-contained restatements of the textbook's own quasi-Newton and trust-region chapters.

**Proposed problem set** (11 problems — already the group's largest set; every
existing item is well-formed and cross-referenced by number from sister files,
so no addition is proposed):
1. [conceptual] **Descent lemma at the optimal step.** Prove the descent lemma from $L$-smoothness, show $\eta=1/L$ maximizes the guaranteed per-step decrease, and explain what breaks at $\eta=2/L$.
   *Provenance:* original (existing ex1, kept; CMU 10-725 HW2 Q3's proximal-gradient proof is the same argument in the field's standard form).
1. [conceptual] **Descent directions under a metric.** Show $\mathbf{d}=-B\nabla f$ is a descent direction for any $B\succ0$; find the steepest direction under $\|\cdot\|_A$ and recognize Newton's direction as the $A=\nabla^2f$ case.
   *Provenance:* original (existing ex2, kept).
1. [conceptual] **Optimal step from the spectrum.** Derive $\eta^\star=2/(\lambda_{\min}+\lambda_{\max})$ and the $(\kappa-1)/(\kappa+1)$ contraction; explain geometrically why $\kappa\to1$ converges in one step.
   *Provenance:* original (existing ex3, kept).
1. [short-code] **A step size that diverges one coordinate.** On a 2-D quadratic, pick $\eta$ between $2/\lambda_2$ and $2/\lambda_1$, run 20 iterations, and reconcile the one-coordinate divergence with the spectral-radius criterion.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Heavy ball's complex eigenvalues.** Write heavy-ball on a 1-D quadratic as a linear map on $(x_k,x_{k-1})$, find when its iteration matrix's eigenvalues go complex, and derive the $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ rate.
   *Provenance:* original (existing ex5, kept).
1. [conceptual] **Minibatch variance with and without replacement.** Prove unbiasedness and $1/b$ variance for sampling with replacement, then show without-replacement sampling stays unbiased and gains a finite-population variance factor.
   *Provenance:* original (existing ex6, kept).
1. [short-code] **The noise ball and the batch/step tradeoff.** Show no constant step converges under nonzero gradient noise, then use the noise-ball scaling to argue when doubling the batch size is worth it, checking the prediction against the schedule-comparison cell.
   *Provenance:* original (existing ex7, kept).
1. [conceptual] **Newton: one step, affine invariance, and its cost.** Show Newton reaches the minimizer of any strictly convex quadratic in one step and is affine-invariant, then compare dense-factorization cost against truncated-CG.
   *Provenance:* original (existing ex8, kept).
1. [conceptual] **Implicit bias of gradient descent.** For underdetermined least squares, show GD from zero stays in the row space and converges to the minimum-norm interpolator.
   *Provenance:* original (existing ex9, kept; its numerical companion is convexity.md ex8).
1. [conceptual] **BFGS and the secant equation.** Verify the BFGS update is symmetric, satisfies the secant equation, and stays positive definite under $\mathbf y_k^\top\mathbf s_k>0$.
   *Provenance:* original (existing ex10, kept — fix the stray blank line separating it from ex9; content unchanged).
1. [short-code] **Trust-region acceptance on a quartic.** Solve the 1-D trust-region subproblem at several radii, compute the improvement ratio, and determine which steps are accepted where unrestricted Newton fails.
   *Provenance:* original (existing ex11, kept).

---

## chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md — Stochastic and Adaptive Methods

**Topic:** the Ghadimi–Lan nonconvex SGD rate; per-coordinate scaling
(AdaGrad, RMSProp, Adam) and Adam's bias correction and non-convergence
counterexample; decoupled weight decay; schedules and warmup; SVRG/SAGA
variance reduction; preconditioners beyond diagonal scaling.

**Current exercises:** 8; disposition: keep 6, rewrite 2, drop 0 — every item
already carries a bold descriptive name and an explicit, checkable deliverable
(the review found no clarity issues); the two rewrites are pure formatting
fixes already flagged by the prior style review (ex4's crammed inline
"(a)...(b)..." and a stray blank line isolating ex8).

**External sources found:**
- Stanford CS231n, Assignment 2 (`FullyConnectedNets.ipynb` / `optim.py`) — students implement SGD-with-momentum, RMSProp, and Adam directly from their update equations and compare optimization trajectories on a small fully connected network trained on real image data; a hands-on, implement-and-race register this section's derivation-heavy exercises do not otherwise offer — https://cs231n.github.io/assignments2024/assignment2/
- Sashank Reddi, Satyen Kale & Sanjiv Kumar, "On the Convergence of Adam and Beyond" (ICLR 2018) — the source of the periodic non-convergence counterexample this section's own ex3/ex7 already exercise directly; already cited as the origin of :eqref:`eq_mdl-opt-reddi`, so no new problem is proposed from it — https://arxiv.org/abs/1904.09237
- **No good external exercise tradition found** for K-FAC/Shampoo/Muon-style structured preconditioners: these are 2015–2024 research methods with lecture coverage (e.g., CMU 10-725's stochastic-gradient lectures) but no course homework problem could be located that derives or implements them.
- **No good external exercise tradition found** specifically for SVRG/SAGA as a *homework* problem (as opposed to lecture material): CMU 10-725's lecture notes cover variance reduction, but no assignment deriving SVRG's unbiasedness or comparing it against SGD/SAGA could be located and verified.

**Proposed problem set** (9 problems — exceeds the guideline range because the
existing 8-item set is uniformly strong and none merit dropping; one
project-scale addition is appended):
1. [conceptual] **AdaGrad from the metric view.** Set $A_t=\mathrm{diag}(\sqrt{\mathbf s_t})+\epsilon I$ in the steepest-descent-under-a-metric result of gradient-based-optimization.md ex2 to recover AdaGrad, and explain why $A_t$ is built from $\sqrt{\mathbf s_t}$ rather than $\mathbf s_t$.
   *Provenance:* original (existing ex1, kept).
1. [conceptual] **Bias correction for the first moment.** Repeat the second-moment bias-correction computation for $\mathbf m_t$, show uncorrected Adam's first step has magnitude $\approx3.16\times$ the true gradient at default hyperparameters, and find where the inflation factor peaks.
   *Provenance:* original (existing ex2, kept).
1. [conceptual] **The Reddi example by hand.** For $C=4$, $\beta_1=0$, $\beta_2=1/17$, compute the three Adam updates of one period of the Reddi–Kale–Kumar counterexample and show the net displacement is toward the wrong endpoint.
   *Provenance:* original (existing ex3, kept; the counterexample itself is Reddi, Kale & Kumar, ICLR 2018).
1. [conceptual] **Coupled versus decoupled fixed points.** Write the stationarity conditions of (a) Adam with an $\ell_2$ penalty and (b) AdamW, show they characterize different points, and explain which applies the uniform per-step shrinkage that matters during training.
   *Provenance:* original (existing ex4, kept — reformat the inline "(a) ... (b) ..." into a nested list; content unchanged).
1. [short-code] **Where $\epsilon$ matters.** On the $\kappa=10^3$ quadratic, sweep $\epsilon$ across five orders of magnitude and record iterations to convergence, explaining the two regimes ($\epsilon$ invisible vs. $\epsilon$-dominated).
   *Provenance:* original (existing ex5, kept).
1. [short-code] **The optimal constant step, exactly.** Derive the exact finite-horizon loss under a fixed step, show the minimizing step scales like $\Theta(\log K/K)$ rather than $1/\sqrt K$, and verify against a grid search at $K=2000$.
   *Provenance:* original (existing ex6, kept).
1. [short-code] **AMSGrad's monotone steps.** Show the effective per-coordinate step is nonincreasing under a running maximum, then modify the Reddi-counterexample cell to interpolate between AdamW and AMSGrad and find the largest forgetting factor at which drift reappears.
   *Provenance:* original (existing ex7, kept).
1. [short-code] **SVRG unbiasedness and cost.** Prove the conditional unbiasedness of the SVRG estimator, count component-gradient evaluations per epoch, and vary the inner-loop length to explore the snapshot-amortization/staleness tradeoff.
   *Provenance:* original (existing ex8, kept — fix the stray blank line isolating it; content unchanged).
1. [extended] **Build and race three optimizers.** Implement SGD-with-momentum, RMSProp, and Adam from their update equations against a small multi-layer network trained on a real dataset, independently tune each optimizer's learning rate, and report which reaches a fixed loss threshold first at matched compute, relating the ranking to this section's coordinate-scaling argument.
   *Provenance:* adapted from Stanford CS231n Assignment 2 (overlap medium — the "implement the three updates and race them on a real net" structure is adopted; CS231n's own numerical-precision scaffolding is dropped since this section's NumPy cells already provide that) — https://cs231n.github.io/assignments2024/assignment2/

---

## chapter_mdl-optimization/mdl-convexity.md — Convex Sets and Convex Functions

**Topic:** three characterizations of convex functions (chords, tangents,
Hessian) plus subgradients; Jensen's inequality; why convexity makes local
minima global; a calculus for recognizing convexity, conjugates, proximal
operators, and coordinate descent; what survives for nonconvex networks.

**Current exercises:** 9; disposition: keep 6, rewrite 3, drop 0 — the review
found no clarity issues (ex5 and ex8/ex9 are singled out as particularly
well-scaffolded); the three rewrites are pure formatting fixes already flagged
by the prior style review (ex2 and ex8's crammed inline lettering, and a
stray blank line isolating ex9).

**External sources found:**
- CMU, 10-725 "Convex Optimization" (Fall 2018), Homework 2, Q1 "Subgradients and Proximal Operators" — derives Hölder's inequality, the $\ell_p$-norm subdifferential from it, and closed-form proximal operators for a quadratic, negative entropy ($z\log z$, via the Lambert $W$ function), the $\ell_2$ norm, and $\|\cdot\|_0$ — a broader "proximal operator zoo" than this section's own single ($\ell_1$/soft-thresholding) example — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework2.pdf
- Boyd & Vandenberghe, "Additional Exercises for Convex Optimization," §2 "Convex functions," Ex. 2.13 "Reverse Jensen inequality" and Ex. 2.16 "Infimal convolution" — close cousins of this section's own Jensen and conjugate-function exercises (ex4, the conjugate subsection), high conceptual overlap with what is already kept, so no new problem is drawn from them — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- CMU 10-725 Homework 2, Q3 "Convergence Rate for Proximal Gradient Descent" — the same $O(1/k)$ proof this section's own ex6 exercises numerically, confirming rather than upgrading it — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework2.pdf

**Proposed problem set** (10 problems — exceeds the guideline range because
the existing 9-item set is uniformly strong and well cross-referenced; one
narrowly-scoped addition is appended to broaden proximal-operator coverage):
1. [conceptual] **Convex sets from the triangle inequality.** Prove every norm ball is convex, exhibit two convex sets whose union is not, and show the set of strictly positive definite matrices is convex with the PSD cone as its closure.
   *Provenance:* original (existing ex1, kept).
1. [conceptual] **Certify or refute, one line each.** Certify $x\log x$, the hinge loss, and $\|Ax-b\|_1$ by naming the calculus rule, then decide whether the pointwise minimum of two convex functions is convex.
   *Provenance:* original (existing ex2, kept — reformat the inline "(a)...(d)" into a nested list; content unchanged).
1. [conceptual] **The subdifferential of $|x|$ and soft-thresholding.** Compute $\partial f(x)$ for $f(x)=|x|$, verify the optimality criterion picks out $x^\star=0$, and derive the soft-thresholding solution for $\ell_1$-penalized least squares.
   *Provenance:* original (existing ex3, kept).
1. [conceptual] **Jensen by induction, and the AM–GM chain.** Prove Jensen's finite form by induction from the chord inequality, then use it with $-\log$ to derive $\mathrm{HM}\le\mathrm{GM}\le\mathrm{AM}$.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Quasiconvexity is weaker than convexity.** Show every convex function is quasiconvex and that $\sqrt{|x|}$ is quasiconvex but not convex, then identify which step of the local-equals-global proof survives.
   *Provenance:* original (existing ex5, kept).
1. [short-code] **Rate contraction at the aggressive step.** Verify numerically that $\|\mathbf x_t-\mathbf x^\star\|$ is nonincreasing for gradient descent on a least-squares objective, and predict then verify the value-gap contraction at $\eta=2/(\mu+L)$.
   *Provenance:* original (existing ex6, kept).
1. [conceptual] **The PL condition, proved and tested.** Show the PL condition implies every stationary point is global and that strong convexity implies PL, then show PL does not imply convexity via $f(x)=x^2+3\sin^2x$.
   *Provenance:* original (existing ex7, kept).
1. [short-code] **Implicit bias, verified.** On a random $4\times10$ underdetermined system, compare the gradient-descent limit against the pseudoinverse, confirm the off-row-space component of the iterates stays exactly zero, then restart off the row space and observe what changes.
   *Provenance:* original (existing ex8, kept — reformat the inline "(a)(b)(c)" into a nested list; content unchanged; numerical companion to gradient-based-optimization.md ex9).
1. [short-code] **Coordinate descent as Gauss–Seidel.** Derive the coordinate-update formula, show one cyclic sweep is Gauss–Seidel on $A\mathbf x=\mathbf b$, then compare cyclic against randomized coordinate order and construct a matrix where cyclic is substantially worse.
   *Provenance:* original (existing ex9, kept — fix the stray blank line separating it from ex8; content unchanged).
1. [conceptual] **A proximal-operator zoo.** Derive closed-form proximal operators for $h(z)=\tfrac12z^\top Az+b^\top z+c$ ($A\succeq0$) and $h(z)=\sum_iz_i\log z_i$ (via the Lambert $W$ function); verify each numerically against a generic proximal solver on one instance, and contrast the $\ell_2$-norm case's shrink-toward-zero behavior with $\ell_1$'s soft-thresholding from ex3.
   *Provenance:* adapted from CMU 10-725 Homework 2, Q1(b) (overlap medium — the quadratic and negative-entropy proximal derivations are adopted as targets; the source's own $\ell_1$ item is dropped since ex3 already covers soft-thresholding) — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework2.pdf

---

## chapter_mdl-optimization/mdl-constrained-optimization-duality.md — Constrained Optimization and Duality

**Topic:** equality/inequality constraints and the KKT conditions; projections
and projected gradient descent; the Lagrange dual, weak/strong duality and
Slater's condition; duality as a saddle point and shadow prices; worked duals
(SVM, water-filling).

**Current exercises:** 8; disposition: keep 8, rewrite 0, drop 0 — the review
found no defects or clarity issues in this file's exercises at all (the only
chapter-wide defect noted is the Discussions block's prose-instead-of-tabs
convention, which is outside the exercises themselves).

**External sources found:**
- Boyd & Vandenberghe, "Additional Exercises for Convex Optimization," §4 "Duality," Ex. 4.14 "Kantorovich inequality" — derive the KKT conditions of a simplex-constrained log-sum problem, show a specific point is optimal, then apply it with $a_k=\lambda_k(A)$ to prove the classical Kantorovich inequality bounding $(u^\top Au)(u^\top A^{-1}u)$ by a function of $\lambda_1/\lambda_n$ — a genuine extension of this section's KKT toolkit into the eigenvalue-conditioning language numerical-stability-conditioning.md also uses — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- Same source, Ex. 4.7 "Connection between perturbed optimal cost and Lagrange dual functions" and Ex. 4.18 "An exact penalty function" — sensitivity/shadow-price and penalty-method material closely paralleling this section's own shadow-price subsection and duality-as-saddle-point discussion; high conceptual overlap with what is already kept, so no new problem is drawn from them — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf
- CMU, 10-725, Homework 3 (Fall 2018), Q4 "Support Vector Machines and Duality" — derives the SVM primal KKT conditions, the dual QP, and the RBF-kernel Gram-matrix argument, then has students solve the dual with a QP solver and identify support vectors by their $\alpha_i$ range — the same derivation this section's own SVM-dual worked example and ex6 already carry out; confirms rather than upgrades the exercise, so no new problem is drawn from it — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework3.pdf
- CMU 10-725 Homework 3, Q1–Q2 — LP duality derivations and a log-barrier-vs-LP KKT comparison; standard material this section's own ex1/ex3/ex5 already cover in different problem instances — https://www.stat.cmu.edu/~ryantibs/convexopt-F18/homework/homework3.pdf

**Proposed problem set** (9 problems — exceeds the guideline range because
the existing 8-item set is complete with no defects; one addition is
appended that extends the KKT toolkit toward the chapter's own conditioning
theme):
1. [conceptual] **Two equality constraints.** Derive the Lagrange condition when two constraint gradients are linearly independent, then revisit the degenerate counterexample and identify which proof step fails.
   *Provenance:* original (existing ex1, kept).
1. [conceptual] **Maximum entropy on the simplex.** Maximize entropy over the simplex with one multiplier, show the optimum is uniform, and explain where the positivity constraints' inactivity was used.
   *Provenance:* original (existing ex2, kept).
1. [conceptual] **KKT for a norm ball, and a non-convex trap.** Write all four KKT conditions for projection onto a Euclidean ball, verify the clipping solution, then exhibit a 1-D nonconvex problem with a KKT point that is a local maximum.
   *Provenance:* original (existing ex3, kept).
1. [conceptual] **Projected-gradient fixed points are KKT points.** Show $\mathbf x^\star$ is a fixed point of projected gradient descent onto the nonnegative orthant iff it satisfies complementary-slackness-style conditions, and check this is exactly KKT.
   *Provenance:* original (existing ex4, kept).
1. [short-code] **Weak duality, and an exact dual solve.** Reproduce the concavity and weak-duality proofs from the definitions, then compute the dual of an equality-constrained QP in closed form and verify $d^\star=p^\star$ numerically on a $2\times1$ instance.
   *Provenance:* original (existing ex5, kept).
1. [conceptual] **The SVM dual's norm identity.** Show that at the SVM dual optimum $\sum_i\alpha_i^\star=\|\tilde{\mathbf w}^\star\|^2$ from complementary slackness, and conclude the dual optimal value equals $\tfrac12\|\tilde{\mathbf w}^\star\|^2$.
   *Provenance:* original (existing ex6, kept; CMU 10-725 HW3 Q4 derives the same SVM dual independently, confirming the exercise's standard form).
1. [short-code] **Water-filling's diminishing returns.** Show the optimal rate is concave and piecewise smooth in the power budget, the water level is piecewise linear and increasing, and identify at which budget values the water-filling cell's slope changes.
   *Provenance:* original (existing ex7, kept).
1. [short-code] **A duality gap flipped to zero.** Rerun the nonconvex duality-gap demo with $f_0(x)=+x^2$, compute $p^\star$ and $d^\star$ by hand and numerically, and explain via Slater's condition why the gap now closes.
   *Provenance:* original (existing ex8, kept).
1. [conceptual] **Kantorovich inequality via KKT.** For $a_1\ge\cdots\ge a_n>0$ and $b_k=1/a_k$, derive the KKT conditions of $\min\,-\log(a^\top x)-\log(b^\top x)$ s.t. $x\succeq0,\mathbf 1^\top x=1$, show $x=(1/2,0,\dots,0,1/2)$ is optimal, then apply this with $a_k=\lambda_k(A)$ for $A\succ0$ to prove the Kantorovich inequality $2(u^\top Au)^{1/2}(u^\top A^{-1}u)^{1/2}\le\sqrt{\lambda_1/\lambda_n}+\sqrt{\lambda_n/\lambda_1}$.
   *Provenance:* adapted from Boyd & Vandenberghe, "Additional Exercises for Convex Optimization," Exercise 4.14 (overlap high — statement and structure adopted directly; cite on adoption) — https://web.mit.edu/~jadbabai/www/EE605/additional_exercises.pdf

---

## chapter_mdl-optimization/mdl-numerical-stability-conditioning.md — Numerical Stability and Conditioning

**Topic:** floating-point representation and range; numerically safe softmax,
log-sum-exp, and cross-entropy; catastrophic cancellation and Welford's
recursion; conditioning, backward/forward error, the normal equations, and
ridge regularization as preconditioning.

**Current exercises:** 8; disposition: keep 8, rewrite 0, drop 0 — the review
found no defects and no clarity issues anywhere in this file; every item is a
precise, numerically checkable task.

**External sources found:**
- MIT, 18.335 "Introduction to Numerical Methods" (S. Johnson), Problem Set 1, Problem 1 "Floating point" — assigns Trefethen & Bau's Exercise 13.2 directly, confirming this section's own machine-epsilon exercise (ex1) sits squarely in the field's standard curriculum — https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/6e6730885b20bd046cbde805d15a4835_MIT18_335JS19_pset1.pdf
- Same source, Problem 2 "Funny functions" — write an $L_4$-norm function and a $\cot(x)-\cot(x+y)$ function that are each accurate near their respective cancellation regimes without arbitrary-precision arithmetic — the same "rewrite to avoid cancellation" task as this section's own ex6, with two different worked functions — https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/6e6730885b20bd046cbde805d15a4835_MIT18_335JS19_pset1.pdf
- Same source, Problem 4 "Addition, another way" — prove that recursive pairwise (divide-and-conquer) summation of $n=2^k$ floats has worst-case error $O(u\log_2 n)$ rather than naive summation's $O(un)$, then implement it efficiently and confirm the bound empirically — a direct upgrade of this section's own pairwise-merge exercise (ex4) from a correctness check into a formal error-bound proof — https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/6e6730885b20bd046cbde805d15a4835_MIT18_335JS19_pset1.pdf
- **No specific exercise verified** from Higham's *Accuracy and Stability of Numerical Algorithms*, despite the section's own Discussions citing it (`:citet:`Higham.2002``) as its standard reference: the book is taught from directly rather than posted online as a standalone problem set, so no numbered exercise could be fetched and confirmed — a finding, not a failure.

**Proposed problem set** (9 problems — exceeds the guideline range because the
existing 8-item set is complete with no defects; one addition is appended
that formalizes the error-bound direction of the existing pairwise-merge
exercise):
1. [short-code] **Machine epsilon by halving.** Compute $\varepsilon_{\text{mach}}$ for float32 by a halving loop, explain why it exits at $2^{-24}$ rather than $2^{-23}$, and repeat for bfloat16 to confirm $2^{-7}$.
   *Provenance:* original (existing ex1, kept; MIT 18.335 PSet 1 assigns Trefethen & Bau Ex. 13.2 as the same task).
1. [conceptual] **Where fp16 overflows but fp32 doesn't.** Find all integer logits for which $e^x$ overflows in fp16 but not fp32, and explain why a network with activations $\approx30$ fails in fp16 without loss scaling despite representable softmax probabilities.
   *Provenance:* original (existing ex2, kept).
1. [conceptual] **Log-sum-exp's gradient is softmax.** Prove $\nabla\,\mathrm{lse}(\mathbf z)=\mathrm{softmax}(\mathbf z)$, use convexity of lse to conclude cross-entropy's gradient is $\mathrm{softmax}(\mathbf z)-\mathbf e_y$, another reason to compute the loss from logits.
   *Provenance:* original (existing ex3, kept).
1. [short-code] **When naive variance goes negative.** Construct a three-number dataset where the naive variance formula returns a negative value in float64, verify Welford's recursion gets it right, and derive the pairwise-merge rule for parallelizing across devices.
   *Provenance:* original (existing ex4, kept).
1. [conceptual] **Sterbenz's lemma.** Show the relative error of $a-b$ can be as large as $(|a|+|b|)u/|a-b|$, then prove subtraction is exact when $a/2\le b\le2a$.
   *Provenance:* original (existing ex5, kept).
1. [short-code] **Rewriting to avoid cancellation.** Rewrite $\sqrt{x+1}-\sqrt x$, $1-\cos x$, and the small root of a quadratic to avoid cancellation, and check one numerically in float32.
   *Provenance:* original (existing ex6, kept; MIT 18.335 PSet 1 Problem 2 poses the same task for the $L_4$ norm and $\cot(x)-\cot(x+y)$).
1. [short-code] **Backward error on the Hilbert matrix.** Compute the backward error of each Hilbert-system solve with respect to the right-hand side, and verify the forward-error bound is tightest where the condition number is smallest.
   *Provenance:* original (existing ex7, kept).
1. [conceptual] **Ridge as a conditioning dial.** Compute the condition number and predicted gradient-descent iteration count for three ridge penalties, then explain via constrained-optimization-duality.md which constrained problem each penalty implicitly solves.
   *Provenance:* original (existing ex8, kept).
1. [short-code] **Pairwise summation's logarithmic error bound.** For $n=2^k$, prove that pairwise (divide-and-conquer) summation satisfies $|\tilde f(x)-f(x)|\le u\log_2(n)\sum_i|x_i|+O(u^2)$, in contrast to naive summation's $O(nu)$; implement it, sweep $n$ from $2^4$ to $2^{20}$ on random inputs, and confirm the measured error grows logarithmically rather than linearly.
   *Provenance:* adapted from MIT 18.335 (S. Johnson), Problem Set 1, Problem 4 "Addition, another way" (overlap medium — the error-bound proof and the log-vs-linear empirical check are adopted; the source's recursion-vs-loop-overhead engineering aside is dropped as out of scope) — https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/6e6730885b20bd046cbde805d15a4835_MIT18_335JS19_pset1.pdf
