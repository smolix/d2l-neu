# Optimization
:label:`chap_mdl-optimization`

Every network in this book was trained by the same loop: compute the
gradient of the loss, step against it, repeat. The size of the step is the
learning rate. Too high a rate makes the loss overflow, and a lower one is
stable but can be slow. :numref:`chap_optimization` showed what momentum,
adaptive methods and schedules do about this on real networks, and left the
proofs to this chapter. Why does the learning rate have a ceiling? Why can a
step below it still make almost no progress? Why do noisy gradients keep a
run from settling?

In 1847 Augustin-Louis Cauchy, who wanted to compute the orbits of celestial
bodies, proposed to solve a system of equations by moving downhill on the
sum of their squared errors, the method we now call gradient descent. Its
theory ties each question to a property of the function. The steepest
curvature sets the largest safe step, the ratio of the steepest to the
flattest curvature sets the number of steps, and convexity decides whether
the point where a run stops is the best one. The bowl
$\tfrac12(x_1^2 + 1000\,x_2^2)$, the running example of every section, is a
thousand times steeper in one direction than in the other, so a step that
is safe in the steep direction barely moves along the flat one, and gradient
descent needs thousands of steps. The same theory explains why momentum and Adam help, why
clipping and weight decay act as constraints, and why a loss can turn into
`NaN`.

:numref:`sec_mdl-gradient-based-optimization` asks how far a gradient step
can go and how curvature sets the number of steps.
:numref:`sec_mdl-convexity` asks when a point that looks optimal locally is
optimal globally, and how fast gradient descent reaches it.
:numref:`sec_mdl-stochastic-gradient-methods` asks what noisy gradients cost
and how to remove that cost.
:numref:`sec_mdl-adaptive-stochastic-methods` asks how an optimizer can learn
one step size per coordinate from gradients alone, and when that fails.
:numref:`sec_mdl-constrained-optimization-duality` asks how to find the best
point when the variables must stay in a set.
:numref:`sec_mdl-lagrangian-duality` asks how to prove that no feasible point
is better, and what the multipliers measure.
:numref:`sec_mdl-numerical-stability-conditioning` asks when finite-precision
arithmetic, rather than the algorithm, decides the answer.

We assume gradients, Hessians and eigenvalues. The proofs assume smooth or
convex functions. Deep-network losses are neither, so for them the results
are models, and each section says where its model stops. Interior-point and
operator-splitting methods are left out.

```toc
:maxdepth: 2

mdl-gradient-based-optimization
mdl-convexity
mdl-stochastic-gradient-methods
mdl-adaptive-stochastic-methods
mdl-constrained-optimization-duality
mdl-lagrangian-duality
mdl-numerical-stability-conditioning
further-reading
```

## Notation {.unnumbered}

The chapter uses the symbols of :numref:`tab_mdl-opt-notation`. Each section
restates the ones it needs, and symbols used in only one section are defined
there.

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

The resources below accompany the sections of this chapter; each names the
section it goes with. Topics that the chapter does not teach, such as optimal
methods on quadratics, mirror descent, interior-point methods,
operator splitting, and probabilistic rounding-error analysis, are collected in
:numref:`sec_mdl-optimization-further-reading`.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — convex sets and functions, duality, and interior-point methods, with a free [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf); the reference for :numref:`sec_mdl-convexity`, :numref:`sec_mdl-constrained-optimization-duality`, and :numref:`sec_mdl-lagrangian-duality`.
- [Numerical Optimization — Nocedal & Wright (2nd ed., Springer)](https://doi.org/10.1007/978-0-387-40065-5) — line search, trust regions, quasi-Newton and conjugate-gradient methods, and constrained optimization, in more depth than :numref:`sec_mdl-gradient-based-optimization` and :numref:`sec_mdl-constrained-optimization-duality`.
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
