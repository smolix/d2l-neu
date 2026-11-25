# Calculus and Automatic Differentiation
:label:`chap_mdl-calculus`

Throughout this book, training a neural network follows the same rhythm:
evaluate a loss, ask the framework for its gradient, and nudge every weight a
small step downhill. We have relied on that loop without pausing to ask two
natural questions about how it works. First, suppose a training run suddenly
diverges after we double the learning rate. The gradient itself is still
correct, yet the step has carried us past the small neighborhood where a local
slope reliably predicts the loss. How far can that linear approximation be
trusted? Second, notice how remarkably cheap the gradient is. Estimating every
slope by perturbing a hundred million weights one at a time would require a
hundred million forward passes, yet a single backward call returns all of them
for roughly the cost of a few forward passes. How is that possible?

Both questions belong to calculus, a subject born in the seventeenth century
when Newton and Leibniz asked how to measure continuous change and how to add
up infinitely many tiny pieces. Their two tools, the derivative and the
integral, turn out to be inverses of each other. In deep learning, the
derivative supplies the best linear model of a loss surface near our current
weights, and aside from the loss value itself, it is the only information an
optimizer ever sees. The integral enters as soon as data or weights are
random: a probability or an expected loss is an area under a curve, and
methods such as policy gradients (:numref:`sec_policygradient`) must
differentiate that integral with respect to the very distribution they sample
from.

Earlier in the book, we stated the basic rules of differentiation
(:numref:`sec_calculus`) and treated automatic differentiation as a ready-made
tool (:numref:`sec_autograd`). The main chapters require nothing more, so this
appendix chapter is optional reading for readers who want to see why the
machinery works and where its guarantees come from. Three results do most of
the heavy lifting: the descent lemma, which converts a bound on curvature into
the safe step-size rule quoted in :numref:`sec_gd`; the chain rule written as
a product of Jacobians, which becomes far cheaper when multiplied from the
loss backward to give backpropagation (:numref:`sec_backprop`); and the two
complementary estimators for the gradient of an expectation, which trade
generality against variance.

The chapter unfolds in three parts. We first build linear and quadratic
approximations of a function, in one variable
(:numref:`sec_mdl-single_variable_calculus`) and then in many
(:numref:`sec_mdl-multivariable_calculus`), so we can bound safe step sizes
and tell a minimum from a saddle point. Next, we follow the chain rule from a
sum over paths in a computation graph (:numref:`sec_mdl-chain-rule-graphs`) to
products of layer Jacobians (:numref:`sec_mdl-matrix-calculus`), and finally
to the algorithms that multiply them automatically
(:numref:`sec_mdl-matrix-calculus-autodiff`). The closing part introduces
integration (:numref:`sec_mdl-integral_calculus`) and brings sampling and
differentiation together to estimate gradients of expectations
(:numref:`sec_mdl-expectation-gradients`).

If you are already comfortable with introductory calculus, feel free to skim
the warm-up pages that open several sections. We omit vector calculus beyond
the gradient (such as divergence and Stokes' theorem) as well as measure
theory; for differential equations, see :numref:`chap_mdl-dynamics`.

```toc
:maxdepth: 2

mdl-single-variable-calculus
mdl-multivariable-calculus
mdl-chain-rule-graphs
mdl-matrix-calculus
mdl-matrix-calculus-autodiff
mdl-integral-calculus
mdl-expectation-gradients
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §34]{.kicker}

Calculus and Automatic Differentiation: From Local Models to Backpropagation<br>**Taylor approximations · descent lemma · gradients & Hessians · computation graphs · matrix calculus · autodiff · expectation gradients**
:::
:::

::: {.slide title="Two Core Questions Behind Every Training Step"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. How far can a local slope be trusted?** The gradient $\nabla\ell(\mathbf{w})$ is a first-order linear model; how rapidly curvature bends the loss away determines which learning rates $\eta$ are safe.
- **2. Why is the gradient of $10^8$ weights so cheap?** Perturbing $10^8$ weights one by one would take ~12 days, whereas one backward pass returns all $10^8$ exact partial derivatives in ~30 ms.
:::

::: {.col .fig .big}
@fig:ch34-calc-two-questions
:::
:::
:::

::: {.slide title="Three Pillars Carrying Deep Learning Calculus"}
[Unifying Principles]{.kicker}

::: {.cols .vc}
::: {.col}
- **Pillar 1: The Descent Lemma (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-multivariable_calculus}):** Taylor's Lagrange remainder converts a curvature bound $|f''| \le L$ into safe step sizes $0 < \eta < 2/L$ ($\eta^\star = 1/L$).
- **Pillar 2: Reverse-Mode Jacobian Products (§§{sec_mdl-chain-rule-graphs}–§{num:sec_mdl-matrix-calculus-autodiff}):** Associating $((\mathbf{u}^\top\mathbf{J}_K)\mathbf{J}_{K-1})\cdots\mathbf{J}_1$ from the output backward computes all gradients in $O(1)$ sweeps.
- **Pillar 3: Differentiating Expectations (§§{sec_mdl-integral_calculus}–§{num:sec_mdl-expectation-gradients}):** Score-function (REINFORCE) vs. pathwise reparameterization trade generality against Monte Carlo variance.
:::

::: {.col .fig .big}
@fig:ch34-calc-three-pillars
:::
:::
:::

::: {.slide title="Chapter §{num:chap_mdl-calculus} Roadmap: From 1D Slices to Sampling"}
[Architectural Progression]{.kicker}

::: {.cols .vc}
::: {.col}
- **Part I: Local Models (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-multivariable_calculus}):** Taylor polynomials, Lagrange remainders, descent lemma, gradients $\nabla f \perp$ level sets, and symmetric Hessians $\mathbf{H}_f$.
- **Part II: Chain Rule & Autodiff (§§{sec_mdl-chain-rule-graphs}–§{num:sec_mdl-matrix-calculus-autodiff}):** Path sums on DAGs, differential method $df = \langle \nabla f, d\mathbf{X}\rangle$, dual numbers, Wengert tape, $2\sqrt{T}$ checkpointing, HVPs, and implicit diff.
- **Part III: Integrals & Expectations (§§{sec_mdl-integral_calculus}–§{num:sec_mdl-expectation-gradients}):** Riemann sums, FTC, Fubini, $|\det\mathbf{J}_{\boldsymbol{\phi}}|$ in flows, Leibniz rule, REINFORCE, and pathwise gradients.
:::

::: {.col .fig .big}
@fig:ch34-calc-roadmap-three-parts
:::
:::
:::

::: {.slide title="Running Benchmarks Anchoring Each Section"}
[Concrete Pedagogical Thread]{.kicker}

::: {.cols .vc}
::: {.col}
- **Local & Matrix Models (§§{sec_mdl-single_variable_calculus}–§{num:sec_mdl-matrix-calculus}):** Logistic loss $\ell(w) = \log(1+e^{-w})$ ($L = 1/4$), log-sum-exp at $(0, \log 2)$ ($\nabla f = (\tfrac{1}{3}, \tfrac{2}{3})^\top$), 4-input integer DAG ($\bar{w} = 73{,}472$), and 2-2-1 ReLU MLP.
- **Autodiff & Stochastic Calculus (§§{sec_mdl-matrix-calculus-autodiff}–§{num:sec_mdl-expectation-gradients}):** From-scratch `Dual`/`Var` engines ($802$ vs. $80{,}200$ ops), polar Gaussian integral $\sqrt{2\pi}$, and $\mathbb{E}[X^2]$ variance ($\sigma = 2.0$ vs. $7.18$).
:::

::: {.col .fig .big}
@fig:ch34-calc-running-benchmarks
:::
:::
:::

::: {.slide title="How to Navigate Chapter §{num:chap_mdl-calculus}"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **If you know introductory calculus:** Skim the differentiation table in §{sec_mdl-single_variable_calculus} and the Riemann sum / antiderivative review in §{sec_mdl-integral_calculus}, and jump directly to the **Mean Value Theorem, Lagrange remainder, and descent lemma** (§{sec_mdl-single_variable_calculus}) and **change of variables in $\mathbb{R}^n$** (§{sec_mdl-integral_calculus}).
- **Essential for deep learning theory & derivations:** Master the **differential method** $df = \langle \nabla f, d\mathbf{X}\rangle$ and trace move rule in §{sec_mdl-matrix-calculus}—it replaces hundred-entry matrix derivative tables and prevents silent transpose bugs on square matrices.
:::

::: {.col}
- **Essential for systems & modern ML:** Study **§{sec_mdl-matrix-calculus-autodiff} (Automatic Differentiation)** for why reverse mode costs ~3 forward passes, how activation checkpointing shrinks memory from $O(T)$ to $2\sqrt{T}$, how forward-over-reverse computes $\mathbf{H}\mathbf{v}$ without a 4 TB Hessian, and **§{sec_mdl-expectation-gradients}** for Monte Carlo gradient variance.
:::
:::

::: {.d2l-note}
**Up next (§{sec_mdl-single_variable_calculus} Single-Variable Calculus):** How first and second derivatives build local linear and quadratic models, bound approximation error, and certify safe step sizes for gradient descent.
:::
:::

