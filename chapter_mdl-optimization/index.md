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
