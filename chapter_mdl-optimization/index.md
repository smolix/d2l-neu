# Optimization
:label:`chap_mdl-optimization`

Every neural network in this book is trained by the same basic loop: compute
the gradient of the loss, take a step in the opposite direction, and repeat.
In :numref:`chap_optimization`, we saw empirically what momentum, adaptive
step sizes, and learning-rate schedules do for that loop on real networks.
Those experiments left three natural questions open. Why is there a hard
ceiling on the learning rate? Why can a step safely below that ceiling still
make painfully slow progress? And why does gradient noise keep a training run
from settling down?

Optimization theory ties each of these questions to a simple geometric
property of the objective. The steepest curvature sets the largest safe step
size; the ratio of the steepest curvature to the flattest sets how many steps
we need; and convexity decides whether the point where our run stops is
actually the best one available. Throughout this chapter, we will return to a
single running example: the narrow bowl
$\tfrac12(x_1^2 + 1000\,x_2^2)$, which is a thousand times steeper in one
direction than in the other. A step size that stays safe along the steep wall
barely nudges the iterate along the flat valley, forcing plain gradient
descent to take thousands of steps.

The two optimization chapters divide the work cleanly:
:numref:`chap_optimization` focuses on practice and measurement, while this
chapter supplies the proofs and underlying geometry. A few compact results,
such as the $2/L$ learning-rate ceiling or the Adam update, appear in both
places so each chapter stands on its own. Full derivations live only here,
and you can read the main text of the book without them.

The chapter unfolds in three parts. The first three sections assume exact
gradients. :numref:`sec_mdl-gradient-based-optimization` proves the
step-size ceiling and iteration count, and shows how momentum replaces the
curvature ratio with its square root. :numref:`sec_mdl-convexity` establishes
when a stationary point is guaranteed to be a global minimum. If you read
only two sections of this chapter, make them these two. Between them,
:numref:`sec_mdl-curvature-methods` covers Newton's method and its relatives,
which actively model curvature rather than merely respecting it; because
deep learning uses them less often, you can safely save that section for a
second reading. The next two sections bring in gradient noise:
:numref:`sec_mdl-stochastic-gradient-methods` derives the third result to
remember, namely that a fixed step stalls at a noise floor proportional to
the step size, and :numref:`sec_mdl-adaptive-stochastic-methods` pits Adam
against a tuned heavy-ball method to see where per-coordinate scaling helps
and where it falls short. The remaining sections are best read on demand: constraints
and duality (:numref:`sec_mdl-constrained-optimization-duality`,
:numref:`sec_mdl-lagrangian-duality`) underpin the norm-budget view of weight
decay in :numref:`sec_weight_decay`, and
:numref:`sec_mdl-numerical-stability-conditioning` explains why the
low-precision arithmetic of :numref:`sec_numerics` breaks down when it does.

We assume familiarity with gradients and Hessians
(:numref:`sec_mdl-multivariable_calculus`) as well as eigenvalues
(:numref:`sec_mdl-eigendecompositions`). Our proofs assume smooth or convex
objectives. Deep-network losses are neither, so for neural networks these
theorems serve as local models rather than literal guarantees, and each
section notes where its model stops applying. Classical topics such as
interior-point and operator-splitting methods are left out of the main text
and summarized in the further-reading section at the end.

```toc
:maxdepth: 2

mdl-gradient-based-optimization
mdl-curvature-methods
mdl-convexity
mdl-stochastic-gradient-methods
mdl-adaptive-stochastic-methods
mdl-constrained-optimization-duality
mdl-lagrangian-duality
mdl-numerical-stability-conditioning
further-reading
```

## Notation {.unnumbered}

Throughout this chapter, we follow the conventions summarized in
:numref:`tab_mdl-opt-notation`. Each section restates the symbols it uses
most, and notation specific to a single section is introduced where it first
appears.

:Symbols of the optimization chapter. A star $\star$ marks an optimum and an asterisk $*$ a convex conjugate. Matrices are bold.
:label:`tab_mdl-opt-notation`

| Symbol | Meaning |
|:--|:--|
| $f$, $\ell(\mathbf{w})$ | objective; training loss as a function of the weights |
| $\mathbf{x}^\star$, $f^\star$ | a minimizer and the minimum value |
| $\nabla f$, $\nabla^2 f$, $\mathbf{g}_k$ | gradient, Hessian; the gradient (or a stochastic gradient) used at step $k$ |
| $L$, $\mu$ | smoothness and strong-convexity constants, $\mu\mathbf{I} \preceq \nabla^2 f \preceq L\mathbf{I}$ |
| $\kappa = L/\mu$; $\kappa(\mathbf{A})$ | condition number of the problem; of a matrix, $\sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$ |
| $\lambda_i$, $\lambda_{\max}(\mathbf{A})$ | eigenvalues of a Hessian or of a quadratic's matrix $\mathbf{A}$ |
| $\eta$, $\beta$ | step size (learning rate); momentum coefficient |
| $k$, $t$ | iteration index; optimizer step in the adaptive methods |
| $\epsilon$ | target accuracy; in Adam, the constant that keeps a division finite |
| $n$, $d$, $\mathcal{B}$, $b$ | number of examples; number of variables; a minibatch and its size |
| $\boldsymbol{\Sigma}$ | covariance of a per-example gradient |
| $\mathbf{m}_t$, $\mathbf{v}_t$; $\beta_1$, $\beta_2$ | Adam's first- and second-moment estimates; their decay rates |
| $\mathbf{P}$ | preconditioner: the step is $-\eta\,\mathbf{P}^{-1}\mathbf{g}$ |
| $g_i$, $h_j$ | inequality constraints $g_i(\mathbf{x}) \le 0$; equality constraints $h_j(\mathbf{x}) = 0$ |
| $\lambda$, $\nu$ | multipliers of inequality and equality constraints; $\lambda$ also weights the decay term $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ |
| $\mathcal{L}$, $q$ | Lagrangian; dual function |
| $p^\star$, $d^\star$ | optimal values of the primal and the dual problem |
| $f^*$ | convex conjugate of $f$ |
| $\Pi_C$, $\mathrm{prox}_\psi$ | Euclidean projection onto a set $C$; proximal operator of $\psi$ |
| $u$, $\varepsilon_{\mathrm{mach}}$ | unit roundoff and machine epsilon of a floating-point format, $u = \varepsilon_{\mathrm{mach}}/2$ |

## Resources and Further Reading {.unnumbered}

The books, courses, and surveys below accompany the material taught in this
chapter, with each entry pointing to the sections it pairs with. Topics that
lie outside the main text, including optimal methods on quadratics, mirror
descent, interior-point methods, operator splitting, and probabilistic
rounding-error analysis, are collected in
:numref:`sec_mdl-optimization-further-reading`.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — convex sets and functions, duality, and interior-point methods, with a free [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf); the reference for :numref:`sec_mdl-convexity`, :numref:`sec_mdl-constrained-optimization-duality`, and :numref:`sec_mdl-lagrangian-duality`.
- [Numerical Optimization — Nocedal & Wright (2nd ed., Springer)](https://doi.org/10.1007/978-0-387-40065-5) — line search, trust regions, quasi-Newton and conjugate-gradient methods, and constrained optimization, in more depth than :numref:`sec_mdl-gradient-based-optimization`, :numref:`sec_mdl-curvature-methods`, and :numref:`sec_mdl-constrained-optimization-duality`.
- [Lectures on Convex Optimization — Nesterov (2nd ed., Springer)](https://doi.org/10.1007/978-3-319-91578-4) — complexity bounds and accelerated first-order methods, with the general proofs that :numref:`sec_mdl-gradient-based-optimization` and :numref:`sec_mdl-convexity` cite.
- [Optimization for Data Analysis — Wright & Recht (Cambridge, 2022)](https://www.cambridge.org/core/books/optimization-for-data-analysis/C02C3708905D236AA354D1CE1739A6A2) — gradient, accelerated, stochastic, and coordinate methods for data analysis, a companion to :numref:`sec_mdl-gradient-based-optimization` through :numref:`sec_mdl-adaptive-stochastic-methods`.
- [Accuracy and Stability of Numerical Algorithms — Higham (2nd ed., SIAM)](https://doi.org/10.1137/1.9780898718027) — rounding-error analysis, summation, and the conditioning of linear systems and least squares, the material of :numref:`sec_mdl-numerical-stability-conditioning`.
- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — a free, self-contained chapter on gradient descent, constrained optimization, and convexity, at the level of :numref:`sec_mdl-gradient-based-optimization` and :numref:`sec_mdl-constrained-optimization-duality`.

**Courses and video lectures**

- [Stanford EE364a: Convex Optimization I — Boyd](https://web.stanford.edu/class/ee364a/) — slides, homework, and software for the Boyd & Vandenberghe text, with [2023 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h); it follows :numref:`sec_mdl-convexity` and :numref:`sec_mdl-lagrangian-duality`.
- [CMU 10-725: Convex Optimization — Tibshirani](https://www.stat.cmu.edu/~ryantibs/convexopt/) — ML-oriented slides and scribed notes on subgradients, proximal and stochastic methods, duality, and ADMM, with [Fall 2016 lecture videos](https://www.youtube.com/playlist?list=PLjbUi5mgii6AVdvImLB9-Hako68p9MpIC); it accompanies :numref:`sec_mdl-stochastic-gradient-methods`, :numref:`sec_mdl-constrained-optimization-duality`, and :numref:`sec_mdl-lagrangian-duality`.

**Tutorials, notes, and surveys**

- [Why Momentum Really Works — Goh (Distill)](https://distill.pub/2017/momentum/) — interactive figures of momentum on convex quadratics, the analysis of :numref:`subsec_mdl-momentum-acceleration`.
- [Optimization Methods for Large-Scale Machine Learning — Bottou, Curtis & Nocedal](https://arxiv.org/abs/1606.04838) — the theory of stochastic-gradient methods for large-scale learning, the survey behind :numref:`sec_mdl-stochastic-gradient-methods`.
- [An overview of gradient descent optimization algorithms — Ruder](https://www.ruder.io/optimizing-gradient-descent/) — momentum, Nesterov acceleration, AdaGrad, RMSProp, and Adam side by side, the methods of :numref:`sec_mdl-adaptive-stochastic-methods`.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic — Goldberg](https://doi.org/10.1145/103162.103163) — the IEEE format, rounding, and cancellation, the opening material of :numref:`sec_mdl-numerical-stability-conditioning`.
- [Optimization (Chapter 5), Patterns, Predictions, and Actions — Hardt & Recht](https://mlstory.org/optimization.html) — a concise account of optimization in supervised learning, a short alternative route through :numref:`sec_mdl-gradient-based-optimization` and :numref:`sec_mdl-stochastic-gradient-methods`.
