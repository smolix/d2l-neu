# Optimization Algorithms
:label:`chap_optimization`

Every deep network in the preceding chapters was trained with a standard recipe—such as minibatch SGD with momentum $0.9$, or AdamW with linear warmup and cosine decay (:numref:`sec_training_recipes`). When a training run diverges to `nan`, we lower the learning rate; when progress stalls, we adjust the schedule or batch size. This chapter opens the black box of the optimizer to answer three practical questions: what determines the largest safe learning rate and the speed of convergence below it, how does each component of a modern training recipe reshape the update step, and how must those hyperparameters scale as models and batch sizes grow?

Gradient descent itself predates modern computing: Augustin-Louis Cauchy proposed it in 1847 to compute planetary orbits, and in 1951 Herbert Robbins and Sutton Monro analyzed root-finding under noisy observations with decaying step sizes. Across both classical and modern settings, two fundamental obstacles limit any gradient-based optimizer:

1. **Ill-conditioning (uneven curvature):** When the loss surface curves sharply in some directions and gently in others, a learning rate small enough to remain stable along the steepest direction makes painfully slow progress along the flattest directions.
2. **Gradient noise:** Because gradients are estimated from finite minibatches rather than the full dataset, every update carries sampling variance that prevents constant-step-size iterates from settling at a minimum.

Every design choice in a modern optimizer targets one of these two obstacles through three levers:

* **Step direction:** Momentum (:numref:`sec_momentum`), adaptive coordinate-wise methods such as Adam and AdamW (:numref:`sec_adam`), and matrix-level orthogonalized methods such as Muon (:numref:`sec_muon`) replace the raw gradient with a preconditioned or smoothed direction that travels faster along ill-conditioned valleys.
* **Step-size schedule:** Learning-rate warmup, cosine or linear decay, and warmup–stable–decay (WSD) schedules (:numref:`sec_scheduler`) navigate the trade-off between rapid early traversal and low final noise.
* **Minibatch size and scaling:** The critical batch size (:numref:`sec_batch_size`) determines when larger minibatches stop reducing the number of optimization steps, while Maximal Update Parametrization ($\mu\text{P}$, :numref:`sec_scaling`) allows learning rates tuned on a small proxy model to transfer directly to much wider networks.

We develop each method by pairing an analytical model—a two-dimensional quadratic valley with condition number $\kappa = 20$—with systematic empirical measurements on two neural network benchmarks introduced in :numref:`sec_test_models`: the convolutional classifier `FashionCNN` and the byte-level Transformer language model `TinyLM`. Finally, :numref:`sec_practice` combines gradient clipping, weight averaging, and hyperparameter scaling into an end-to-end protocol for configuring and diagnosing large-scale training runs.

This chapter assumes familiarity with gradients, eigenvalues of symmetric matrices, and basic minibatch SGD (:numref:`subsec_linear-regression-minibatch-sgd`). We give concise derivations where they directly guide hyperparameter choices and defer longer mathematical proofs to :numref:`chap_mdl-optimization`. Annotated references to textbooks, papers, and practitioner playbooks are collected in :numref:`sec_optimization-further-reading`.

```toc
:maxdepth: 2

gd
sgd
test-models
momentum
adam
muon
lr-scheduler
batch-size
scaling
practice
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9]{.kicker}

Optimization Algorithms: Geometry, Stochasticity, and Scaling<br>**ill-conditioning · gradient noise · momentum · AdamW & Muon · schedules · critical batch size · μP**
:::
:::

::: {.slide title="Two Obstacles Limit Every Deep Learning Optimizer"}
[The Central Problem]{.kicker}

Training a neural network minimizes an empirical risk $f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{x})$ over $d = 10^6\text{--}10^{12}$ parameters across a non-convex loss surface governed by two geometric and statistical bottlenecks:

::: {.cols}
::: {.col}
- **Obstacle 1 — Ill-Conditioning (Uneven Curvature):**
  - On the valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$, curvature is $\lambda_2 = 4$ across the steep wall and $\lambda_1 = 0.2$ along the flat floor ($\kappa = \lambda_{\max}/\lambda_{\min} = 20$).
  - Stability along $x_2$ requires $|1 - 4\eta| < 1$ ($\eta < 0.5$), forcing the flat axis $x_1$ to shrink by $<10\%$ per step ($47$ steps to reach $1\%$ error at $\eta^\star = 0.476$).
:::

::: {.col}
- **Obstacle 2 — Minibatch Gradient Noise:**
  - Estimating $\nabla f(\mathbf{x})$ from a random minibatch of $b$ examples injects sampling variance $\operatorname{tr}(\boldsymbol{\Sigma})/b$.
  - At a constant learning rate $\eta$, iterates cannot settle at a minimum; they fluctuate inside a stationary **noise ball** whose variance scales as $\eta\sigma^2 / (2\lambda b)$.
:::
:::

::: {.d2l-note}
Every component of a modern training recipe—momentum, AdamW, Muon, warmup, cosine/WSD decay, critical batch size, and $\mu\text{P}$—is an explicit engineering answer to one of these two obstacles.
:::
:::

::: {.slide title="Three Design Levers of a Modern Training Recipe"}
[Architectural Framework]{.kicker}

Modern optimization algorithms attack ill-conditioning and gradient noise through three orthogonal levers:

| Design Lever | Target Obstacle | Core Mechanism | Chapter 9 Sections |
| :--- | :--- | :--- | :--- |
| **1. Step Direction & Preconditioning** | Ill-conditioning ($\kappa \gg 1$) & heterogeneous layer scales | Replace $-\nabla f(\mathbf{x})$ with a temporal velocity average, per-coordinate RMS scaling, or spectral orthogonalization | **§9.4 Momentum**, **§9.5 Adam & AdamW**, **§9.6 Muon** |
| **2. Step-Size Schedule $\eta_t$** | Early sharpness instability & late-stage noise floor | Ramp $\eta_t$ up during warmup to steer below $2/\eta_0$, hold a high travel budget $B_T = \sum_t \eta_t$, and decay $\eta_T \to 0$ | **§9.7 Schedules (Warmup, Cosine, WSD)** |
| **3. Batch Size, Width Scaling & Guards** | Sampling variance $\operatorname{tr}(\boldsymbol{\Sigma})/b$, width drift, & loss spikes | Match $b$ to the gradient-noise scale $b_{\text{noise}}$, scale layer LRs via $\mu\text{P}$, and guard with clipping & weight EMA | **§9.8 Batch Size**, **§9.9 Scaling Up ($\mu\text{P}$)**, **§9.10 Practice** |
:::

::: {.slide title="From Exact Analytical Models to Neural Benchmarks"}
[Pedagogical Methodology]{.kicker}

We develop every optimizer by first deriving exact predictions on tractable analytical models and then testing those predictions under a controlled multi-seed protocol on real neural networks:

| Running Testbed | Scale & Geometry | What It Isolates and Measures |
| :--- | :--- | :--- |
| **Quadratic Valley** ($f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$) | $d = 2$, $\lambda_1 = 0.2, \lambda_2 = 4$, $\kappa = 20$ | Exact stability ceilings, contraction rates, critical damping ($\sqrt{\kappa}$), and noise-ball variance |
| **NASA Airfoil Linear Regression** | $n = 1{,}500$ examples, $d = 6$ weights | Exact Hessian $L = 2.10$ and gradient covariance $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) = 3.24$ to verify excess loss $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$ |
| **`FashionCNN` on Fashion-MNIST** | $421{,}642$ params ($2$ conv + $2$ FC layers) | Multi-epoch classification ($8.5$ epochs in $2{,}000$ steps at $b = 256$), test accuracy, and weight EMA |
| **`TinyLM` on WikiText-103 Bytes** | $470{,}784$ params ($2$-block Transformer; $3.3\text{M}$ in §9.10) | Single-pass language modeling ($3\%$ of $270\text{M}$ bytes in $2{,}000$ steps), $40\times$ gradient scale spread, AdamW & Muon |
:::

::: {.slide title="Chapter 9 Roadmap: Sections and Core Invariants"}
[Chapter Map]{.kicker}

| Section | Core Mathematical Relation or Invariant | Practical Takeaway |
| :--- | :--- | :--- |
| **§9.1 Gradient Descent** | Stability $\eta < 2/\lambda_{\max}$; steps $t \approx \frac{\kappa}{2}\log\frac{1}{\epsilon}$; edge of stability $\lambda_{\max} \approx 2/\eta$ | Scalar learning rates are bottlenecked by condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ |
| **§9.2 Minibatch SGD** | Noise ball $\mathbb{E}[x_\infty^2] \approx \frac{\eta\sigma^2}{2\lambda b}$; excess loss $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ | $\eta/b$ governs the steady-state noise floor; minibatches amortize hardware overhead |
| **§9.3 Two Test Models** | Initial loss $\ln V + s^2/2$; $3\times$ five-seed range threshold ($0.05$ nats on `TinyLM`) | Treat $\eta$ as a nuisance parameter swept per optimizer from shared initial weights |
| **§9.4 Momentum** | Effective step $\eta_{\text{eff}} = \frac{\eta}{1-\beta}$; critical damping $\rho^\star = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \approx 1 - \frac{2}{\sqrt{\kappa}}$ | Accelerates curvature-limited runs from $O(\kappa)$ to $O(\sqrt{\kappa})$; rescales $\eta \propto 1-\beta$ under noise |
| **§9.5 Adam & AdamW** | Step $\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$; decay timescale $\tau = \frac{1}{\eta\lambda}$; equilibrium RMS $\sqrt{\frac{\eta}{2\lambda}}$ | Equalizes $40\times$ layer gradient disparities; decouple weight decay on 2D matrices only |
| **§9.6–9.10 Muon & Run** | Spectral step $-\eta\mathbf{U}\mathbf{V}^\top$; budget $B_T$; $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$; $\mu\text{P}$ hidden LR $\eta/m$ | Orthogonalize 2D hidden matrices; pair warmup + WSD/cosine decay with $b_{\text{crit}}$ and $\mu\text{P}$ |
:::

::: {.slide title="How to Navigate Chapter 9 and the Math Appendix"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core practitioner sequence (§9.1–§9.5, §9.7–§9.8, §9.10):**
  - Master the two bottlenecks (**§9.1–§9.2**), experimental benchmarking (**§9.3**), momentum and AdamW (**§9.4–§9.5**), learning-rate schedules (**§9.7**), critical batch size (**§9.8**), and production diagnostics (**§9.10**).
- **Advanced scaling & matrix methods (§9.6, §9.9):**
  - Study **§9.6 (Muon)** for spectral-norm steepest descent via Newton–Schulz polynomials and **§9.9 (Scaling Up)** for zero-shot learning-rate transfer across widths via $\mu\text{P}$.
:::

::: {.col}
- **Connection to subsequent chapters & proofs:**
  - `TinyLM`, its WikiText-103 byte loader, and its AdamW/schedule harness return in the chapters on Attention, Transformers, State Space Models, and Performance.
  - Concise derivations that directly guide hyperparameter choices appear inline; full convergence proofs and convex analysis live in **Chapter 35 (*Mathematics for Deep Learning: Optimization*)**.
:::
:::

::: {.d2l-note}
**Up next (§9.1 Landscapes and Gradient Descent):** How directional curvature and the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ determine the maximum stable step size and the edge of stability.
:::
:::

