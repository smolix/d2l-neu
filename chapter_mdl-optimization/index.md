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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35]{.kicker}

Mathematics for Deep Learning: Optimization<br>**smoothness · condition number · momentum · curvature · convexity · stochastic noise · preconditioning · duality · numerics**
:::
:::

::: {.slide title="Three Questions Every Training Loop Leaves Open"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why is there a hard step-size ceiling?** Stepping past $2/L$ along the steepest direction overshoots the valley wall and increases the loss.
- **Why do safe steps crawl?** When curvatures differ by $\kappa = L/\mu$, the steepest wall caps $\eta$ while the flattest direction stalls.
- **Why does minibatch noise stop above $f^\star$?** Constant-step SGD settles into a stationary noise ball of loss $\propto \eta\sigma^2/b$.
:::

::: {.col .fig .big}
@fig:ch35-opt-three-questions
:::
:::
:::

::: {.slide title="The Running Benchmark: An Ill-Conditioned Valley"}
[Unifying Thread]{.kicker}

::: {.cols .vc}
::: {.col}
We test every theorem against $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ ($\mu = 1, L = 1000, \kappa = 1000$):

- **First-order scaling:** Fixed-step GD takes $6{,}908$ steps ($O(\kappa)$), while Heavy-Ball and Nesterov momentum take $315$–$508$ steps ($O(\sqrt{\kappa})$).
- **Preconditioning vs. rotation:** Exact Newton finishes in $1$ step and aligned Adam in $22$–$48$ steps, but $45^\circ$ rotation breaks diagonal scaling.
- **Stochastic noise floor:** Constant-step SGD ($\eta = 10^{-3}, \sigma^2 = 1$) stalls at loss $7.5 \times 10^{-4}$.
:::

::: {.col .fig .big}
@fig:ch35-opt-benchmark-valley
:::
:::
:::

::: {.slide title="Architecture of Chapter 35: Three Core Parts"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **Part I — Exact Gradients & Geometry (§§35.1–35.3):** $L$-smoothness descent lemma, $\kappa$ vs. $\sqrt{\kappa}$ momentum law, Newton/BFGS/trust regions, and convexity guarantees.
- **Part II — Stochastic & Adaptive Methods (§§35.4–35.5):** The $1/b$ variance law, stationary noise floors, Robbins–Monro schedules, AdamW, and matrix preconditioning (K-FAC, Muon).
- **Part III — Constraints, Duality & Numerics (§§35.6–35.8):** KKT conditions, projections, proximal ISTA, Fenchel duality, and floating-point conditioning.
:::

::: {.col .fig .big}
@fig:ch35-opt-chapter-architecture
:::
:::
:::

::: {.slide title="Core Mathematical Quantities and Notation"}
[Notation Reference]{.kicker}

::: {.cols}
::: {.col}
- **Curvature & Conditioning:**
  - $L$, $\mu$: Smoothness ceiling and strong-convexity floor, $\mu\mathbf{I} \preceq \nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$.
  - $\kappa = L/\mu$: Problem condition number; for a matrix $\mathbf{A}$, $\kappa(\mathbf{A}) = \sigma_{\max}(\mathbf{A})/\sigma_{\min}(\mathbf{A})$.
  - $\mathbf{P}$: Preconditioner matrix rescaling the step to $-\eta \mathbf{P}^{-1}\mathbf{g}_k$.
- **Stochastic & Adaptive State:**
  - $\boldsymbol{\Sigma}$, $b$: Per-example gradient covariance and minibatch size (noise power $\operatorname{tr}\boldsymbol{\Sigma}/b$).
  - $\mathbf{m}_t, \mathbf{v}_t$: First- and second-moment moving averages with decay factors $\beta_1, \beta_2$.
:::

::: {.col}
- **Constraints, Duality & Operators:**
  - $g_i(\mathbf{x}) \le 0$, $h_j(\mathbf{x}) = 0$: Inequality and equality constraints with multipliers $\lambda_i \ge 0$ and $\nu_j$.
  - $\mathcal{L}$, $q$, $p^\star$, $d^\star$: Lagrangian, dual function, primal optimum, and dual optimum ($d^\star \le p^\star$).
  - $f^*$, $\Pi_C$, $\mathrm{prox}_\psi$: Convex conjugate, Euclidean projection onto $C$, and proximal operator.
- **Floating-Point Precision:**
  - $\varepsilon_{\mathrm{mach}} = 2^{-p}$, $u = 2^{-p-1}$: Machine epsilon and unit roundoff for $p$ mantissa bits.
:::
:::
:::

::: {.slide title="How to Navigate This Chapter"}
[Study Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **Essential Core (§35.1 & §35.3):** Start here for the $2/L$ step ceiling, $\sqrt{\kappa}$ momentum speedup, tangent lower bounds, Jensen's inequality, and global convergence rates.
- **Deep Learning Workhorses (§35.4 & §35.5):** Read next for SGD noise floors, cosine/WSD schedules, Adam scale invariance, decoupled AdamW, and Muon.
- **On-Demand Deep Dives (§35.2, §§35.6–35.8):** Consult for L-BFGS/trust regions, KKT/projections/ISTA, Fenchel duality, and `bf16` numerical stability.
:::

::: {.col .fig .big}
@fig:ch35-opt-study-roadmap
:::
:::
:::

