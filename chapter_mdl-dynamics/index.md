# Dynamics: Differential Equations and Generative Flows
:label:`chap_mdl-dynamics`

A residual network adds a small update to its input at every layer, and
gradient descent moves the weights a short step downhill. Both solve, step
by step, a differential equation: a rule that says for every state how fast
it changes. A diffusion model turns noise into an image the same way. Each
call to a network returns a direction, and the sample moves a short step
along it. How few steps can such a model take, and where does the direction
come from when all we have are example images?

The second question is answered from the wrong end. Add noise to the
training images little by little, and they wander like particles in a fluid.
A cloud of noisy particles is unpredictable one particle at a time, yet its
density changes deterministically, and running that change backward turns
noise into data. The tools of the Basics part stop short of this.
Backpropagation differentiates a fixed computation, not the solution of an
equation. Probability describes a random variable at one time, not a density
that moves. And noise whose size grows like the square root of time breaks
the ordinary chain rule.

The chapter has three stages, each with a result to keep. First a given rule
moves one state, exactly and then in steps: the fastest rate in the rule caps
the step of gradient descent and of every solver that computes its next
state by a formula (:numref:`sec_mdl-euler-runge-kutta`). That is half the
answer to the first question; the other half is a straighter path from noise
to data (:numref:`sec_mdl-rectified-flow`). Then noise moves a
whole cloud, whose density can be run backward once we know, at every point,
the direction in which the cloud grows denser, its *score*
(:numref:`sec_mdl-fokker-planck-probability-flow`). Last, that direction is
learned: it turns out that least squares on noisy samples learns the score,
or the velocity of the particles in its place, although no sample comes
labeled with either (:numref:`sec_mdl-score-matching-diffusion-flow`,
:numref:`sec_mdl-flow-matching-transport`).

On the way, a linear equation with an input yields the recurrence of a
state space model (:numref:`subsec_mdl-ode-driven-linear`,
:numref:`sec_ssm`), and a residual network (:numref:`sec_resnet`) with more
layers and smaller updates becomes a differential equation
(:numref:`sec_mdl-neural-odes-cnf`). The hardest section is
:numref:`sec_mdl-sdes`, which repairs the chain rule. Readers headed for
diffusion models can skip the adjoint method
(:numref:`sec_mdl-adjoint-method`) and optimal transport
(:numref:`sec_mdl-ot-connection`).

:numref:`chap_diffusion` trains networks and judges their images. It
repeats our short results, leaves the long derivations here, and can be read
without them. Most experiments here run on fields known in closed form, so a
sampler that misses has only itself to blame.

We assume eigendecompositions (:numref:`sec_mdl-eigendecompositions`),
Jacobians (:numref:`subsec_mdl-jacobian`), Gaussians and conditional
expectation, and no measure theory. Where a proof would need it, we give a
plausible argument and cite the proof.

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

A residual network block, a gradient descent update, and a diffusion model denoising step all execute the same primitive operation: moving a state vector a short step along a prescribed velocity field.

::: {.cols}
::: {.col}
- **Architectures & Optimizers as Solvers:** A ResNet $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}(\mathbf{x}_l)$ and gradient descent $\mathbf{w}_{k+1} = \mathbf{w}_k - \eta \nabla \mathcal{L}(\mathbf{w}_k)$ are forward-Euler steps ($h = 1$ and $h = \eta$) of an underlying ODE.
- **Why Learning Rates Blow Up:** The fastest eigenvalue rate in the vector field caps the stable step size of every explicit solver, explaining why gradient descent diverges when $\eta > 2/\lambda_{\max}(\mathbf{H})$.
:::

::: {.col}
- **Generative Sampling by Reversing Noise:** Adding Gaussian noise to images causes particles to wander randomly, yet the cloud's probability density $p_t(\mathbf{x})$ evolves deterministically and can be integrated backward from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to $p_{\text{data}}$.
- **Why Standard Calculus Breaks:** Brownian kicks scale as $\sqrt{dt}$, so their squares $(dW_t)^2 = dt$ survive at first order and add a second-derivative correction to the chain rule.
:::
:::
:::

::: {.slide title="The Three-Stage Mathematical Roadmap of Chapter 38"}
[Architecture of the Chapter]{.kicker}

The chapter progresses from deterministic trajectories of a single state to stochastic evolution of probability densities and finally to simulation-free training of generative vector fields:

| Stage & Sections | Central Mathematical Question | Key Theorem or Result |
| :--- | :--- | :--- |
| **1. Deterministic ODEs & Flows** (§38.1–38.3) | When does a velocity rule define an invertible map, and how do discrete steps track it? | Picard–Lindelöf uniqueness, stability function $R(h\lambda)$, adjoint ODE, and $\frac{d}{dt}\log p_t = -\operatorname{tr}\mathbf{J}$ |
| **2. Stochastic Calculus & Density Flow** (§38.4–38.5) | How does fresh $\sqrt{dt}$ noise move a particle cloud, and how can that motion be reversed? | Itô's lemma ($(dW)^2 = dt$), OU Gaussian kernels, Fokker–Planck PDE, reverse SDE & PF-ODE |
| **3. Learning Generative Fields** (§38.6–38.7) | How can a network learn the score $\nabla \log p_t$ or velocity $\mathbf{u}_t$ from unlabeled data samples? | Vincent's denoising score matching, DDPM/DDIM equivalence, Conditional Flow Matching & Reflow |
:::

::: {.slide title="Concrete Running Benchmarks Anchoring the Chapter"}
[Pedagogical Thread]{.kicker}

Every theorem and numerical solver is tested first on closed-form dynamical systems where the exact trajectory, density, or score is known analytically:

| Running Benchmark | Mathematical Definition | What It Tests Across §38.1–38.7 |
| :--- | :--- | :--- |
| **Planar Spiral Sink** | $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$, $\mathbf{A} = \left(\begin{smallmatrix}-0.5 & -1 \\ 1 & -0.5\end{smallmatrix}\right)$, $\lambda = -0.5 \pm \mathrm{i}$ | Matrix exponential $e^{\mathbf{A}t}$, Euler vs. RK4 order, adjoint sensitivity, volume shrinkage $e^{-t}$ |
| **Stiff Two-Mode Decay** | $\dot{\mathbf{x}} = \mathrm{diag}(-50, -1)\mathbf{x}$ and quadratic valley $\mathbf{H} = \mathrm{diag}(10, 1)$ | Explicit step-size ceiling $h < 2/|\lambda|$, A-stability of backward Euler, ill-conditioned GD |
| **Ornstein–Uhlenbeck SDE** | $dX = -\theta X\,dt + g\,dW$ ($\theta = 1$, $g = \sqrt{2}$) | Gaussian transition kernel $\mathcal{N}(\alpha_t x_0, \sigma_t^2)$, variance preservation ($g^2 = 2\theta$), Euler–Maruyama |
| **Bimodal Gaussian Mixture** | $p_0 = w_1\mathcal{N}(-2, 0.25) + w_2\mathcal{N}(2, 0.25)$ | Exact score $s_t(x)$, probability-flow ODE vs. reverse SDE, Langevin trapping, DDIM strides |
| **Two-Moons Transport** | Planar crescent distribution paired with $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | Crossing straight segments, loss floor $C$, 1-step collapse to the mean, Reflow straightening |
:::

::: {.slide title="Unifying Equations: Particles, Densities, Reversal"}
[Core Dictionary]{.kicker}

Four coupled differential equations govern how individual particles and their marginal probability densities $p_t(\mathbf{x})$ move forward and backward in time:

| Perspective | Forward Evolution ($t: 0 \to T$) | Reverse / Generative Evolution ($t: T \to 0$) |
| :--- | :--- | :--- |
| **Stochastic Particle (SDE)** | $d\mathbf{X} = \mathbf{f}(\mathbf{X}, t)\,dt + g(t)\,d\mathbf{W}$ | $d\mathbf{X} = \big[\mathbf{f}(\mathbf{X}, t) - g(t)^2 \nabla \log p_t(\mathbf{X})\big]\,dt + g(t)\,d\bar{\mathbf{W}}$ |
| **Deterministic Particle (ODE)** | $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \tfrac{1}{2}g(t)^2 \nabla \log p_t(\mathbf{x})$ | Same probability-flow ODE integrated from $T$ down to $0$ |
| **Marginal Density (PDE)** | $\partial_t p_t = -\nabla \cdot (\mathbf{f} p_t) + \tfrac{1}{2}g^2 \Delta p_t = -\nabla \cdot (p_t \mathbf{u}_t)$ | $\frac{d}{dt}\log p_t(\mathbf{x}(t)) = -\nabla \cdot \mathbf{u}_t(\mathbf{x}(t))$ along ODE characteristics |

::: {.d2l-note .rule}
Converting the Laplacian diffusion term $\frac{1}{2}g^2 \Delta p_t = \nabla \cdot (p_t \cdot \frac{1}{2}g^2 \nabla \log p_t)$ turns random Brownian smoothing into transport along the score field $\mathbf{s}_t(\mathbf{x}) = \nabla \log p_t(\mathbf{x})$.
:::
:::

::: {.slide title="Prerequisites and Reading Paths Through Chapter 38"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Mathematical Prerequisites:** Eigendecompositions and spectral norms (§33), multivariable Jacobians, divergence, and integration by parts (§34), and Gaussian conditional expectations (§36). No measure theory is required.
- **Optimization & SSM Path (§38.1–38.2):** Covers exact zero-order hold discretizations for State Space Models (Mamba/S4), non-normal transient growth in momentum, and numerical stiffness in gradient descent.
:::

::: {.col}
- **Fast-Track to Diffusion & Flow Models:** Read **§38.1–38.2** for flows and Runge–Kutta solvers, **§38.4** for Itô's lemma and the Ornstein–Uhlenbeck kernel, **§38.5** for reverse-time SDEs and the probability-flow ODE, and **§38.6–38.7** for score and flow matching.
- **Optional Deep Dives:** You can skip the continuous adjoint method (**§38.3**) and Wasserstein optimal transport theory (**§38.7**) on a first pass toward image generation.
:::
:::
:::

