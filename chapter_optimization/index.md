# Optimization Algorithms
:label:`chap_optimization`

Every deep network in the preceding chapters was trained with a standard recipe—such as minibatch SGD with momentum $0.9$, or AdamW with linear warmup and cosine decay (:numref:`sec_training_recipes`). When a training run diverges to `nan`, we lower the learning rate; when progress stalls, we adjust the schedule or batch size. This chapter opens the black box of the optimizer to answer three practical questions: what determines the largest safe learning rate and the speed of convergence below it, how does each component of a modern training recipe reshape the update step, and how must those hyperparameters scale as models and batch sizes grow?

Gradient descent itself predates modern computing: Augustin-Louis Cauchy proposed it in 1847 to compute planetary orbits, and in 1951 Herbert Robbins and Sutton Monro analyzed root-finding under noisy observations with decaying step sizes. Across both classical and modern settings, two fundamental obstacles limit any gradient-based optimizer:

1. **Ill-conditioning (uneven curvature):** When the loss surface curves sharply in some directions and gently in others, a learning rate small enough to remain stable along the steepest direction makes painfully slow progress along the flattest directions.
2. **Gradient noise:** Because gradients are estimated from finite minibatches rather than the full dataset, every update carries sampling variance that prevents constant-step-size iterates from settling at a minimum.

![The two obstacles on a valley whose curvature is $20$ times larger across than along ($\kappa = 20$). Left: gradient descent zig-zags across the steep direction and creeps along the flat one. Right: with noisy gradients and a constant step size, the iterates reach the minimum but never settle; they wander inside a noise ball (dashed ellipse: two standard deviations).](../img/mdl-opt-two-obstacles.svg)
:label:`fig_opt_two_obstacles`

:numref:`fig_opt_two_obstacles` shows both obstacles on the same two-dimensional valley. Every design choice in a modern optimizer targets one of them through three levers:

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

::: {.cols .vc}
::: {.col}
- **Obstacle 1 — Ill-Conditioning ($\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$):**
  - On $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ ($\lambda_1 = 0.2, \lambda_2 = 4, \kappa = 20$), stability along $x_2$ caps $\eta < 2/\lambda_2 = 0.5$, forcing $x_1$ to shrink by $<10\%$ per step ($47$ steps to $1\%$ error).
- **Obstacle 2 — Minibatch Gradient Noise ($\operatorname{tr}(\boldsymbol{\Sigma})/b$):**
  - Sampling $b$ examples traps constant-rate iterates inside a stationary **noise ball** of variance $\eta\sigma^2 / (2\lambda b)$.
- **Chapter thesis:** Every modern optimizer lever attacks one of these two bottlenecks.
:::

::: {.col .fig .big}
@fig:ch09-opt-two-obstacles
:::
:::
:::

::: {.slide title="Three Design Levers of a Modern Training Recipe"}
[Architectural Framework]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Step Direction & Preconditioning (§§{sec_momentum}–§{num:sec_muon}):**
  - Replace $-\nabla f(\mathbf{x})$ with momentum velocity ($\mathbf{v}_t$), per-coordinate RMS normalization (AdamW), or spectral orthogonalization ($\mathbf{U}\mathbf{V}^\top$, Muon).
- **2. Step-Size Schedule $\eta_t$ (§{sec_scheduler}):**
  - Ramp $\eta_t$ during warmup to steer below $2/\eta_0$, hold a high travel budget $B_T = \sum_t \eta_t$, and decay $\eta_T \to 0$.
- **3. Batch Size, Width Scaling & Guards (§§{sec_batch_size}–§{num:sec_practice}):**
  - Match $b$ to $b_{\text{noise}}$, transfer LRs across widths via $\mu\text{P}$ ($\eta/m$), and guard with global-norm clipping and weight EMA.
:::

::: {.col .fig .big}
@fig:ch09-opt-three-levers
:::
:::
:::

::: {.slide title="From Exact Analytical Models to Neural Benchmarks"}
[Pedagogical Methodology]{.kicker}

::: {.cols .vc}
::: {.col}
- **Analytical quadratics & linear data (§§{sec_gd}–§{num:sec_sgd}):**
  - Solve exact stability ceilings, $\sqrt{\kappa}$ critical damping, and excess loss $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ on the $\kappa=20$ valley and NASA airfoil dataset ($n=1{,}500, d=6$).
- **Two complementary neural networks (§§{sec_test_models}–§{num:sec_practice}):**
  - **`FashionCNN` ($421\text{K}$ params):** Multi-epoch image classification ($8.5$ epochs in $2{,}000$ steps at $b = 256$) with held-out test accuracy.
  - **`TinyLM` ($471\text{K}$ / $3.3\text{M}$ params):** Single-pass byte-level Transformer on WikiText-103 ($40\times$ layer gradient spread).
:::

::: {.col .fig .big}
@fig:ch09-opt-four-testbeds
:::
:::
:::

::: {.slide title="Chapter §{num:chap_optimization} Roadmap: Sections and Core Invariants"}
[Chapter Map]{.kicker}

::: {.cols .vc}
::: {.col}
- **Foundations & Benchmarks (§§{sec_gd}–§{num:sec_test_models}):**
  - Stability $\eta < 2/\lambda_{\max}$, edge of stability $\lambda_{\max} \approx 2/\eta$, noise ball $\frac{\eta\sigma^2}{2\lambda b}$, count landmarks, and $3\times$ multi-seed thresholds.
- **Preconditioned Directions (§§{sec_momentum}–§{num:sec_muon}):**
  - Momentum $\eta_{\text{eff}} = \frac{\eta}{1-\beta}$ ($O(\sqrt{\kappa})$), AdamW timescale $\tau = \frac{1}{\eta\lambda}$, and Muon 5-step Newton–Schulz orthogonalization $-\eta\mathbf{U}\mathbf{V}^\top$.
- **Schedules, Scaling & Operations (§§{sec_scheduler}–§{num:sec_practice}):**
  - Warmup + WSD/cosine budget $B_T$, critical batch size $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$, $\mu\text{P}$ hidden LR $\eta/m$, clipping ($\theta=1$), and weight EMA.
:::

::: {.col .fig .big}
@fig:ch09-opt-roadmap
:::
:::
:::

::: {.slide title="How to Navigate Chapter §{num:chap_optimization} and the Math Appendix"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core practitioner sequence (§{sec_gd}–§{sec_adam}, §{sec_scheduler}–§{sec_batch_size}, §{sec_practice}):**
  - Master the two bottlenecks (**§{sec_gd}–§{sec_sgd}**), experimental benchmarking (**§{sec_test_models}**), momentum and AdamW (**§{sec_momentum}–§{sec_adam}**), learning-rate schedules (**§{sec_scheduler}**), critical batch size (**§{sec_batch_size}**), and production diagnostics (**§{sec_practice}**).
- **Advanced scaling & matrix methods (§{sec_muon}, §{sec_scaling}):**
  - Study **§{sec_muon} (Muon)** for spectral-norm steepest descent via Newton–Schulz polynomials and **§{sec_scaling} (Scaling Up)** for zero-shot learning-rate transfer across widths via $\mu\text{P}$.
:::

::: {.col}
- **Connection to subsequent chapters & proofs:**
  - `TinyLM`, its WikiText-103 byte loader, and its AdamW/schedule harness return in the chapters on Attention, Transformers, State Space Models, and Performance.
  - Concise derivations that directly guide hyperparameter choices appear inline; full convergence proofs and convex analysis live in **Chapter §{num:chap_mdl-optimization} (*Mathematics for Deep Learning: Optimization*)**.
:::
:::

::: {.d2l-note}
**Up next (§{sec_gd} Landscapes and Gradient Descent):** How directional curvature and the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ determine the maximum stable step size and the edge of stability.
:::
:::

