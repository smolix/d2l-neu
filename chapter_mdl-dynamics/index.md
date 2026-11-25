# Dynamics: Differential Equations and Generative Flows
:label:`chap_mdl-dynamics`

A residual network adds a small update to its input at every layer, gradient descent nudges weights downhill step by step, and a diffusion model turns noise into an image along a learned vector field. All three discretize differential equations—rules specifying how fast a state changes. Two practical questions follow immediately: how few steps can a sampler take, and how can we learn the velocity or score direction from example images alone?

Adding noise gradually to training images answers the second question from the reverse end. Although each noisy particle wanders unpredictably, the density of the whole cloud evolves deterministically, and reversing that evolution turns Gaussian noise back into data. Earlier chapters stop short of this machinery: backpropagation differentiates a fixed computation graph rather than an ODE solution, static probability does not track moving densities, and Brownian increments of order $\sqrt{\Delta t}$ break the ordinary chain rule.

The chapter unfolds in three stages. First, deterministic ODEs move single states exactly and numerically: the fastest decaying mode caps the step size of gradient descent and every explicit Runge–Kutta solver (:numref:`sec_mdl-euler-runge-kutta`), while straighter transport paths enable few-step generation (:numref:`sec_mdl-rectified-flow`). Along the way, driven linear ODEs yield the recurrence of state space models (:numref:`subsec_mdl-ode-driven-linear`, :numref:`sec_ssm`), and deep residual networks (:numref:`sec_resnet`) pass to neural ODEs and continuous normalizing flows (:numref:`sec_mdl-neural-odes-cnf`). Second, stochastic calculus repairs the chain rule (:numref:`sec_mdl-sdes`) and tracks cloud densities through the Fokker–Planck equation, showing that time reversal requires only the *score* $\nabla \log p_t$ (:numref:`sec_mdl-fokker-planck-probability-flow`). Third, least-squares regression on noisy or interpolated pairs learns the score or marginal velocity directly without ground-truth field labels (:numref:`sec_mdl-score-matching-diffusion-flow`, :numref:`sec_mdl-flow-matching-transport`).

Readers headed straight for :numref:`chap_diffusion` may skip the continuous adjoint method (:numref:`sec_mdl-adjoint-method`) and optimal transport (:numref:`sec_mdl-ot-connection`) on a first pass. Most experiments here test samplers on closed-form vector fields so discretization errors can be isolated cleanly. We assume eigendecompositions (:numref:`sec_mdl-eigendecompositions`), Jacobians (:numref:`subsec_mdl-jacobian`), and Gaussian conditioning, without measure theory.

```toc
:maxdepth: 2

mdl-odes-solvers
mdl-numerical-solvers
mdl-neural-odes
mdl-sdes
mdl-fokker-planck-probability-flow
mdl-score-matching-diffusion-flow
mdl-flow-matching-transport
further-reading
```

## Notation {.unnumbered}

:numref:`tab_mdl-dyn-notation` lists the symbols that recur from one section
to the next. Each section restates the ones it needs and defines the letters
it uses only once. Where a letter in the table carries two meanings, its row
says where each one holds.

:Notation for the dynamics chapter.
:label:`tab_mdl-dyn-notation`

| Symbol | Meaning | First used in |
|----------|----------------|------|
| $\mathbf x(t)\in\mathbb R^d$, $\dot{\mathbf x}$ | state, its time derivative | :numref:`sec_mdl-odes-solvers` |
| $\mathbf x_0$ | state at $t=0$: the start of a trajectory; for a noising process, a data point | :numref:`sec_mdl-odes-solvers` |
| $\mathbf f(\mathbf x,t)$ | vector field of an ODE; drift of an SDE | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-sdes` |
| $\Phi_t$; $\Phi_{s\to t}$ | flow map, $\Phi_t(\mathbf x_0)=\mathbf x(t)$; the map from the state at time $s$ to the state at time $t$ | :numref:`sec_mdl-odes-solvers` |
| $L$ | Lipschitz constant of $\mathbf f$ | :numref:`sec_mdl-odes-solvers` |
| $T$ | final time (horizon) | :numref:`sec_mdl-odes-solvers` |
| $\mathbf A$; $\lambda_j=a_j+\mathrm i\,b_j$; $\mathbf V$, $\boldsymbol\Lambda$ | matrix of a linear field $\dot{\mathbf x}=\mathbf A\mathbf x$; its eigenvalues; matrix of eigenvectors, diagonal matrix of eigenvalues | :numref:`sec_mdl-odes-solvers` |
| $\lambda$ | an eigenvalue of $\mathbf A$ or of a Jacobian; for the reverse-time samplers, the multiple $\lambda g$ of the noise amplitude that they inject | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-fokker-planck-probability-flow` |
| $\mathbf K$ | matrix $\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ of the rotation $\dot{\mathbf x}=\mathbf K\mathbf x$ | :numref:`sec_mdl-odes-solvers` |
| $\mathbf J$ | Jacobian of $\mathbf f$ | :numref:`sec_mdl-odes-solvers` |
| $\mathbf x^\star$ | fixed point of a field, $\mathbf f(\mathbf x^\star)=\mathbf 0$; for a gradient flow, a critical point of the loss | :numref:`sec_mdl-odes-solvers` |
| $\mathbf u(t)$, $\mathbf B$ | an input and its matrix, in $\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf B\mathbf u(t)$ | :numref:`sec_mdl-odes-solvers` |
| $h$, $t_n=nh$, $\mathbf x_n$ | step size (written $\Delta t$ in :numref:`sec_mdl-sdes` and :numref:`sec_mdl-fokker-planck-probability-flow`), time grid, state at $t_n$ (exact, or a solver's iterate) | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-euler-runge-kutta` |
| $\bar{\mathbf A}$, $\bar{\mathbf B}$ | exact step with the input held over $h$, $\mathbf x_{n+1}=\bar{\mathbf A}\mathbf x_n+\bar{\mathbf B}\mathbf u_n$ | :numref:`sec_mdl-odes-solvers` |
| $R(z)$, $z=h\lambda$ | stability function of a solver | :numref:`sec_mdl-euler-runge-kutta` |
| $\varepsilon_n$ | global error $\lVert\mathbf x_n-\mathbf x(t_n)\rVert$ | :numref:`sec_mdl-euler-runge-kutta` |
| $\mathcal L$, $\eta$, $\mathbf H$ | loss, learning rate, Hessian | :numref:`sec_mdl-odes-solvers`; :numref:`sec_mdl-euler-runge-kutta` |
| $N$ | number of steps to the final time: solver steps $n=1,\ldots,N$ with $Nh=T$ (the first two sections write $n$ for this number as well); in :numref:`sec_mdl-score-matching-diffusion-flow` also the noise levels $t=1,\ldots,N$ of DDPM | :numref:`sec_mdl-neural-odes-cnf` |
| $\boldsymbol\theta$ | network parameters | :numref:`sec_mdl-neural-odes-cnf` |
| $\mathbf a(t)$ | adjoint state | :numref:`sec_mdl-neural-odes-cnf` |
| $p_t$ | density of the state at time $t$ | :numref:`sec_mdl-neural-odes-cnf` |
| $\boldsymbol\epsilon$ | in :numref:`sec_mdl-neural-odes-cnf`, the random probe of the trace estimator; from :numref:`sec_mdl-sdes` on, standard Gaussian noise, as in $\mathbf x_t=\alpha_t\mathbf x_0+\sigma_t\boldsymbol\epsilon$, which :numref:`sec_mdl-flow-matching-transport` also writes $\mathbf x_1$, the noise endpoint | :numref:`sec_mdl-neural-odes-cnf`; :numref:`sec_mdl-sdes` |
| $\mathbf X_t$ ($X_t$ in one dimension) | state of a stochastic process at time $t$, a random variable; $\mathbf x$ is a value it takes | :numref:`sec_mdl-sdes` |
| $\mathbf W_t$ | Brownian motion | :numref:`sec_mdl-sdes` |
| $g$, $g(t)$; $\mathbf G$ | noise amplitude of an SDE (its diffusion coefficient), constant or varying in time; noise matrix | :numref:`sec_mdl-sdes` |
| $\theta$ | rate of the OU process $dX=-\theta X\,dt+g\,dW$; in the scalar example of :numref:`sec_mdl-neural-odes-cnf`, the parameter of $\dot x=\theta x$ | :numref:`sec_mdl-neural-odes-cnf`; :numref:`sec_mdl-sdes` |
| $\alpha_t$, $\sigma_t$ | mean factor and standard deviation of $\mathbf X_t$ given $\mathbf X_0$; $\sigma_t$ is the noise level | :numref:`sec_mdl-sdes` |
| $\phi$ | a smooth function of the state: the function in Itô's lemma and the test function behind the Fokker--Planck equation (elsewhere an angle) | :numref:`sec_mdl-sdes` |
| $\psi$ | a smooth scalar field, in the vector-calculus preliminaries | :numref:`sec_mdl-fokker-planck-probability-flow` |
| $\mathbf s_t=\nabla\log p_t$; $\mathbf s_{\boldsymbol\theta}$ | score; learned score | :numref:`sec_mdl-fokker-planck-probability-flow`; :numref:`sec_mdl-score-matching-diffusion-flow` |
| $\mathbf u_t$; $\mathbf u_{\boldsymbol\theta}$ | velocity that transports $p_t$ (the probability-flow velocity), unrelated to the input $\mathbf u(t)$; learned velocity | :numref:`sec_mdl-fokker-planck-probability-flow`; :numref:`sec_mdl-flow-matching-transport` |
| $\mathbf j$ | probability current | :numref:`sec_mdl-fokker-planck-probability-flow` |
| $\bar t=T-t$ | reversed clock | :numref:`sec_mdl-fokker-planck-probability-flow` |
| $w_k$, $r_k(\mathbf x)$, $v_k$ | mixture weights, responsibilities, component variances | :numref:`sec_mdl-fokker-planck-probability-flow` |
| $\beta(t)$; $\beta_t$, $\bar\alpha_t$ | rate of the variance-preserving SDE; noise schedule of DDPM, about $\beta(t)$ times the step (written $\beta_k$ in :numref:`sec_mdl-sdes`), and its running product $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$, which is our $\alpha_t^2$ | :numref:`sec_mdl-sdes`; :numref:`sec_mdl-score-matching-diffusion-flow` |
| $\bar{\mathbf x}_t$, $\bar\sigma_t$ | DDIM coordinates: the rescaled point $\mathbf x_t/\sqrt{\bar\alpha_t}$ and its ratio of noise to signal $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ | :numref:`sec_mdl-score-matching-diffusion-flow` |
| $w(t)$; $\gamma$ | weight of noise level $t$ in the loss; guidance scale | :numref:`sec_mdl-score-matching-diffusion-flow` |
| $\pi$ | coupling: a joint law of pairs of data and noise, also called a pairing | :numref:`sec_mdl-flow-matching-transport` |
| $\mathrm i$, $\mathbf I$ | imaginary unit (upright), identity matrix | throughout |

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §38]{.kicker}

Dynamics: Differential Equations, Stochastic Flows, and Transport<br>**ODEs & solvers · Neural ODEs · Itô SDEs · Fokker–Planck PDE · score matching · flow matching**
:::
:::

::: {.slide title="Why Deep Learning Needs Continuous-Time Dynamics"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Architectures & optimizers as Euler steps:** A ResNet block $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}(\mathbf{x}_l)$ ($h=1$) and gradient descent $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\nabla\mathcal{L}$ ($h=\eta$) are forward-Euler steps of an ODE, capped at $\eta < 2/\lambda_{\max}(\mathbf{H})$.
- **Generative sampling by reversing noise:** Adding Brownian kicks ($\sim \sqrt{dt}$) smooths $p_t(\mathbf{x})$ deterministically via the Fokker–Planck PDE, which can be reversed from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ back to $p_{\text{data}}$.
:::

::: {.col .fig .big}
@fig:ch38-dyn-unified-primitives
:::
:::
:::

::: {.slide title="The Three-Stage Mathematical Roadmap of Chapter §{num:chap_mdl-dynamics}"}
[Architecture of the Chapter]{.kicker}

::: {.cols .vc}
::: {.col}
- **Stage 1: Deterministic ODEs & Flows (§§{sec_mdl-odes-solvers}–§{num:sec_mdl-neural-odes-cnf}):** Picard–Lindelöf uniqueness, matrix exponential $e^{\mathbf{A}t}$, Runge–Kutta stability $R(h\lambda)$, continuous adjoint ODE, and Liouville's trace rule $\frac{d}{dt}\log p_t = -\operatorname{tr}\mathbf{J}$.
- **Stage 2: Stochastic Calculus & Density Flow (§§{sec_mdl-sdes}–§{num:sec_mdl-fokker-planck-probability-flow}):** Quadratic variation $(dW)^2 = dt$, Itô's lemma, OU Gaussian kernels, Fokker–Planck PDE, Anderson's reverse SDE, and the probability-flow ODE.
- **Stage 3: Learning Generative Fields (§§{sec_mdl-score-matching-diffusion-flow}–§{num:sec_mdl-flow-matching-transport}):** Vincent's denoising score matching, Tweedie's formula, DDPM/DDIM, Conditional Flow Matching, and Reflow.
:::

::: {.col .fig .big}
@fig:ch38-dyn-three-stage-roadmap
:::
:::
:::

::: {.slide title="Concrete Running Benchmarks Anchoring the Chapter"}
[Pedagogical Thread]{.kicker}

::: {.cols .vc}
::: {.col}
- **Linear & stiff ODEs:** Planar spiral sink ($\lambda = -0.5 \pm \mathrm{i}$) and stiff two-mode decay ($\mathrm{diag}(-50, -1)$) test $e^{\mathbf{A}t}$, RK4 order, adjoints, and explicit step ceilings.
- **Stochastic & multimodal benchmarks:** Ornstein–Uhlenbeck SDE ($dX = -\theta X\,dt + g\,dW$) and the bimodal Gaussian mixture test exact transition kernels, scores, reverse SDEs, and DDIM strides.
- **2D transport benchmark:** Two-moons paired with $\mathcal{N}(\mathbf{0}, \mathbf{I})$ tests crossing segments, the CFM floor $C$, and Reflow straightening.
:::

::: {.col .fig .big}
@fig:ch38-dyn-five-benchmarks
:::
:::
:::

::: {.slide title="Unifying Equations: Particles, Densities, Reversal"}
[Core Dictionary]{.kicker}

::: {.cols .vc}
::: {.col}
- **Forward & reverse SDEs:** $d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ reverses via $d\mathbf{X} = [\mathbf{f} - g^2\nabla\log p_t]\,dt + g\,d\bar{\mathbf{W}}$.
- **Deterministic probability-flow ODE:** $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \tfrac12 g(t)^2\nabla\log p_t(\mathbf{x})$ shares all marginals $p_t(\mathbf{x})$.
- **Key identity:** Rewriting $\tfrac12 g^2\Delta p_t = \nabla\cdot(p_t \cdot \tfrac12 g^2\nabla\log p_t)$ turns Laplacian diffusion into deterministic transport along the score field.
:::

::: {.col .fig .big}
@fig:ch38-dyn-four-equations-bridge
:::
:::
:::

::: {.slide title="Prerequisites and Reading Paths Through Chapter §{num:chap_mdl-dynamics}"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Mathematical Prerequisites:** Eigendecompositions and spectral norms (§{chap_mdl-linear-algebra}), multivariable Jacobians, divergence, and integration by parts (§{chap_mdl-calculus}), and Gaussian conditional expectations (§{chap_mdl-probability-statistics}). No measure theory is required.
- **Optimization & SSM Path (§{sec_mdl-odes-solvers}–§{num:sec_mdl-euler-runge-kutta}):** Covers exact zero-order hold discretizations for State Space Models (Mamba/S4), non-normal transient growth in momentum, and numerical stiffness in gradient descent.
:::

::: {.col}
- **Fast-Track to Diffusion & Flow Models:** Read **§{sec_mdl-odes-solvers}–§{num:sec_mdl-euler-runge-kutta}** for flows and Runge–Kutta solvers, **§{sec_mdl-sdes}** for Itô's lemma and the Ornstein–Uhlenbeck kernel, **§{sec_mdl-fokker-planck-probability-flow}** for reverse-time SDEs and the probability-flow ODE, and **§{sec_mdl-score-matching-diffusion-flow}–§{num:sec_mdl-flow-matching-transport}** for score and flow matching.
- **Optional Deep Dives:** You can skip the continuous adjoint method (**§{sec_mdl-neural-odes-cnf}**) and Wasserstein optimal transport theory (**§{sec_mdl-flow-matching-transport}**) on a first pass toward image generation.
:::
:::
:::

