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

Every deep learning training loop evaluates a scalar loss $\ell(\mathbf{w})$, calls `backward()` for its gradient, and updates $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla \ell(\mathbf{w})$—raising two fundamental mathematical questions:

::: {.cols}
::: {.col}
- **1. How far can a local slope be trusted?** Doubling the learning rate $\eta$ can make training diverge even though $\nabla \ell(\mathbf{w})$ is exact. The gradient is only a first-order linear model; how rapidly curvature bends the loss away from that tangent plane determines which step sizes are safe.
:::

::: {.col}
- **2. Why is the gradient of $10^8$ weights so cheap?** Perturbing $10^8$ parameters one at a time via finite differences would require $10^8$ forward passes (~12 days at 10 ms/pass). A single backward pass returns all $10^8$ exact partial derivatives in roughly three forward-pass times (~30 ms).
:::
:::

::: {.d2l-note}
Differential calculus answers both questions, while integral calculus enters as soon as models sample from parameterized probability distributions (VAEs, normalizing flows, and policy gradients).
:::
:::

::: {.slide title="Three Pillars Carrying Deep Learning Calculus"}
[Unifying Principles]{.kicker}

Three mathematical results connect classical analysis directly to modern deep learning optimization and software systems:

::: {.cols}
::: {.col}
- **Pillar 1 — The Descent Lemma (§34.1–34.2):** Taylor's theorem with Lagrange remainder converts a bound $L$ on curvature ($|f''| \le L$) into a quadratic upper ceiling, proving that any step size $0 < \eta < 2/L$ strictly decreases the loss and $\eta = 1/L$ maximizes the guaranteed drop.
- **Pillar 2 — Reverse-Mode Jacobian Products (§34.3–34.5):** A $K$-layer network has Jacobian $\mathbf{J} = \mathbf{J}_K \cdots \mathbf{J}_1$. Associating from the scalar loss backward, $((\mathbf{u}^\top \mathbf{J}_K)\mathbf{J}_{K-1})\cdots \mathbf{J}_1$, keeps every step a matrix–vector product and computes all parameter gradients in $O(1)$ sweeps.
:::

::: {.col}
- **Pillar 3 — Differentiating Expectations (§34.6–34.7):** To differentiate $\mathbb{E}_{X \sim p_{\boldsymbol{\theta}}}[g(X)]$, we pass $\nabla_{\boldsymbol{\theta}}$ inside the integral and either differentiate the log-density (**score-function / REINFORCE**, general but high variance) or differentiate a reparameterized sample $X = T_{\boldsymbol{\theta}}(\mathbf{z})$ (**pathwise trick**, low variance when $g$ is smooth).
:::
:::
:::

::: {.slide title="Chapter 34 Roadmap: From 1D Slices to Sampling"}
[Architectural Progression]{.kicker}

The chapter develops in three parts—local polynomial approximations, chain rule and automatic differentiation across layers, and integration with stochastic gradient estimation:

| Section | Core Mathematical Object | Role in Deep Learning |
| :--- | :--- | :--- |
| **§34.1 Single-Variable Calculus** | $f'(x)$, Taylor polynomial $P_n$, Lagrange remainder $R_n$, $\partial f(x)$ | Safe step sizes $\eta < 2/L$, Newton's method, subgradients at $\mathrm{ReLU}$ kinks |
| **§34.2 Multivariable Calculus** | Gradient $\nabla f \in \mathbb{R}^n$, symmetric Hessian $\mathbf{H}_f \in \mathbb{R}^{n \times n}$ | Steepest descent, contour orthogonality, saddle vs. minimum classification |
| **§34.3 Chain Rule on Graphs** | Edge partials $\partial c/\partial n$ vs. total adjoints $\bar{n} = \mathrm{d}f/\mathrm{d}n$ | Path-sum chain rule on DAGs, skip connections, reverse topological sweep |
| **§34.4 Matrix Calculus** | Jacobian $\mathbf{J}_{\mathbf{f}} \in \mathbb{R}^{m \times n}$, differential $df = \langle \nabla f, d\mathbf{X}\rangle$ | Layer backward rules ($\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^\top$, $\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Y}}$), softmax-CE $\mathbf{p} - \mathbf{y}$ |
| **§34.5 Automatic Differentiation** | Dual numbers $a + b\varepsilon$ ($\varepsilon^2=0$), Wengert tape, JVPs & VJPs | Forward vs. reverse cost, $2\sqrt{T}$ checkpointing, $\mathbf{H}\mathbf{v}$ products, implicit diff |
| **§34.6 Integral Calculus** | Riemann sums, Fundamental Theorem, Fubini, $\|\det \mathbf{J}_{\boldsymbol{\phi}}\|$ | Probability densities, normalizing flows volume factor, Gaussian normalizer $\sqrt{2\pi}$ |
| **§34.7 Expectation Gradients** | Leibniz boundary rule, score $\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$, pathwise map $T_{\boldsymbol{\theta}}(\mathbf{z})$ | REINFORCE with baselines vs. VAE reparameterization variance trade-offs |
:::

::: {.slide title="Running Benchmarks Anchoring Each Section"}
[Concrete Pedagogical Thread]{.kicker}

Every theorem and algorithm in this chapter is verified numerically on a concrete analytical benchmark:

| Section | Running Mathematical Benchmark | Key Quantitative Result Verified in Code |
| :--- | :--- | :--- |
| **§34.1** | Softplus $s(x)=\log(1+e^x)$ & logistic loss $\ell(w)=s(-w)$ | $\ell(0.1)$ linear gap $0.00125 \approx \frac{1}{2}\ell''(0)(0.1)^2$; $L=\frac{1}{4} \Rightarrow \eta < 8$ safe |
| **§34.2** | Log-sum-exp $f(x,y)=\log(e^x+e^y)$ at $(0, \log 2)$ | $\nabla f = (\frac{1}{3}, \frac{2}{3})$ (softmax); $\mathbf{H}_f$ eigenvalues $0$ (along $\mathbf{1}$) and $\frac{4}{9}$ |
| **§34.3** | 4-input squaring DAG & 5-node skip-edge DAG | Exact integer adjoints $\bar{w}=73472, \bar{y}=-68224$ in $12$ multiplies vs. $24$ |
| **§34.4** | 2-2-1 $\mathrm{ReLU}$ network & curved map $(x+\sin y, y+x^2/2)$ | Left-to-right VJP costs $3\times 10^6$ multiplies vs. $2\times 10^9$ right-to-left |
| **§34.5** | `Dual` & `Var` engines on $(\sum_{j=1}^{200} x_j^2)^2$ and $y=(uv+u)^2$ | Reverse sweep takes $802$ ops vs. $80{,}200$ in forward mode ($100\times$ speedup) |
| **§34.6–34.7** | Gaussian $\mathcal{N}(0,1)$ polar integral & $\ell(\theta)=\mathbb{E}_{X\sim\mathcal{N}(\theta,1)}[X^2]$ | $\int e^{-z^2/2}dz = \sqrt{2\pi}$; pathwise SD $2.0$ vs. score-function SD $7.18$ at $\theta=1.5$ |
:::

::: {.slide title="How to Navigate Chapter 34"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **If you know introductory calculus:** Skim the differentiation table in §34.1 and the Riemann sum / antiderivative review in §34.6, and jump directly to the **Mean Value Theorem, Lagrange remainder, and descent lemma** (§34.1) and **change of variables in $\mathbb{R}^n$** (§34.6).
- **Essential for deep learning theory & derivations:** Master the **differential method** $df = \langle \nabla f, d\mathbf{X}\rangle$ and trace move rule in §34.4—it replaces hundred-entry matrix derivative tables and prevents silent transpose bugs on square matrices.
:::

::: {.col}
- **Essential for systems & modern ML:** Study **§34.5 (Automatic Differentiation)** for why reverse mode costs ~3 forward passes, how activation checkpointing shrinks memory from $O(T)$ to $2\sqrt{T}$, how forward-over-reverse computes $\mathbf{H}\mathbf{v}$ without a 4 TB Hessian, and **§34.7** for Monte Carlo gradient variance.
:::
:::

::: {.d2l-note}
**Up next (§34.1 Single-Variable Calculus):** How first and second derivatives build local linear and quadratic models, bound approximation error, and certify safe step sizes for gradient descent.
:::
:::

