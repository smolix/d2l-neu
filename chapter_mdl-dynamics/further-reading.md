# Further Reading
:label:`sec_mdl-dynamics-further-reading`

The sections of this chapter integrate a given field, train a field through
its solver, add noise, follow the density of a noisy cloud, and learn the
score or the velocity from samples. This section collects, in the order of
the chapter, topics they had no room for: solvers that respect structure,
other continuous-time models, the physics and the statistics of noise, and
recent variants of the learned generators. Each topic states the problem it
addresses and what it changes relative to the chapter. Resources for the
material the chapter does teach are listed in its introduction
(:numref:`chap_mdl-dynamics`). Nothing else in the book depends on this
section.

## Solvers

### Geometric Integration and Leapfrog

On a frictionless oscillator forward Euler gains energy at every step and
backward Euler loses it, so neither can be run for long. A *symplectic*
method, such as symplectic Euler or leapfrog, does not try to follow the flow
more closely. It preserves phase-space volume instead, and backward error
analysis then shows that it conserves a nearby modified energy up to an
exponentially small error, so its energy error stays bounded over
exponentially long times rather than growing with the horizon
(:exref:`ex_mdl-odes-solvers-symplectic-euler` works the smallest case).
:citet:`Hairer.Lubich.Wanner.2006` develop this theory, including the
modified equations whose first correction for gradient descent is the
implicit gradient regularizer of :citet:`Barrett.Dherin.2021`. Leapfrog is
the integrator inside Hamiltonian Monte Carlo
:cite:`Duane.Kennedy.Pendleton.ea.1987,Neal.2011` because a reversible,
volume-preserving proposal keeps the Metropolis acceptance rule simple.

### Exponential Integrators

The field of a diffusion sampler is semilinear: a stiff linear decay plus a
learned nonlinear part. An explicit Runge--Kutta step must then stay inside
the stability region set by the linear part
(:numref:`sec_mdl-stiffness-implicit`), however smooth the learned part is.
An exponential integrator solves the linear part exactly, through the
variation-of-constants formula with $e^{\mathbf At}$, and approximates only
the nonlinear remainder. :citet:`Hochbruck.Ostermann.2010` survey the
method. DEIS :cite:`Zhang.Chen.2023` and DPM-Solver
:cite:`Lu.Zhou.Bao.ea.2022` apply it to the probability-flow ODE, which
brings the cost of a sample down from hundreds of network evaluations to
ten or twenty. For SDEs, the stochastic Taylor expansion behind Milstein's
correction continues to higher orders, but the terms beyond Milstein's
involve iterated stochastic integrals that are expensive to sample
:cite:`Kloeden.Platen.1992`.

## Continuous-Time Models

### Flows with Cheap Determinants

A discrete normalizing flow needs the log-determinant of every layer's
Jacobian, which costs $O(d^3)$ in general, and
:numref:`sec_mdl-continuous-normalizing-flows` replaced it by an integrated
trace. Discrete flows instead restrict the layer. Coupling layers
:cite:`Dinh.Pascanu.Bengio.ea.2017` and autoregressive flows make the
Jacobian triangular, so its determinant is the product of the diagonal.
Invertible residual networks :cite:`Behrmann.Grathwohl.Chen.ea.2019` keep
the Lipschitz constant of each block below one and estimate the
log-determinant by the power series of
:exref:`ex_mdl-odes-solvers-residual-logdet`. FFJORD
:cite:`Grathwohl.Chen.Bettencourt.ea.2018` is the free-form alternative the
chapter teaches. :citet:`Papamakarios.Nalisnick.Rezende.ea.2021` survey the
families and what each gives up.

### Reversible Solvers and Straighter Flows

The adjoint method of :numref:`sec_mdl-adjoint-method` rebuilds the forward
states by integrating backward, and a strongly contracting flow amplifies the
error of that reconstruction. A reversible solver recovers the forward
trajectory exactly from its endpoint, so the backward pass carries no drift;
:citet:`Kidger.2022` (ch. 5) compares it with checkpointing. A separate cost
is the number of steps a trained field needs. Penalizing the kinetic energy
$\int\|\mathbf f_{\boldsymbol\theta}\|^2\,dt$ during likelihood training
:cite:`Finlay.Jacobsen.Nurbekyan.ea.2020` pushes the learned flow toward
straight paths, the goal that flow matching reaches by choosing its
regression target (:numref:`sec_mdl-ot-connection`). On the line, a flow
also cannot reverse the order of points (:numref:`subsec_mdl-flow-order`);
augmented Neural ODEs :cite:`Dupont.Doucet.Teh.2019` remove this limit by
solving in a space with extra dimensions.

### Latent and Controlled Differential Equations

The chapter's models map an initial state to a final one, but time series
arrive at irregular times and must be read as they come. A latent ODE
:cite:`Rubanova.Chen.Duvenaud.2019` encodes the observations into an
initial state and decodes the trajectory at any set of times. A neural
controlled differential equation :cite:`Kidger.Morrill.Foster.ea.2020`
drives the state by the interpolated data path itself,
$d\mathbf z=\mathbf f_{\boldsymbol\theta}(\mathbf z)\,d\mathbf X_t$, which
is the continuous-time counterpart of a recurrent network.

## Noise

### Physical Noise and the Wong--Zakai Theorem

:numref:`sec_mdl-sdes` defines the Itô integral with the left endpoint and
shows that the midpoint rule, which defines the Stratonovich integral, keeps
the ordinary chain rule. Which one describes a physical system driven by
fast but smooth noise? If the Brownian path is replaced by smooth
approximations and the resulting ODEs are solved classically, the solutions
converge to the Stratonovich solution :cite:`Wong.Zakai.1965`. In one
dimension the two readings of $dX=f\,dt+g\,dW$ differ by the drift
$\tfrac12g\,\partial_xg$. They agree for the additive noise of diffusion
models; for multiplicative noise the modeler has to choose.

### Neural SDEs

What changes when the learned model is an SDE, with a learned drift and a
learned diffusion? The adjoint of :numref:`sec_mdl-adjoint-method` becomes a
backward SDE driven by the same Brownian path, which must be stored or
regenerated from its random seed :cite:`Li.Wong.Chen.ea.2020`. The training
signal is then a divergence between laws of whole paths, or a discriminator
that compares them, rather than a loss on one endpoint :cite:`Kidger.2022`.

### Stochastic Gradient Descent as an SDE

Minibatch noise makes stochastic gradient descent a random recurrence.
Reading a step with learning rate $\eta$ as an Euler--Maruyama step of
$d\boldsymbol\theta=-\nabla\mathcal L\,dt+\sqrt{\eta}\,\boldsymbol\Sigma^{1/2}d\mathbf W$,
where $\boldsymbol\Sigma$ is the covariance of a minibatch gradient, turns
the learning rate into a temperature :cite:`Li.Tai.E.2017`. On a quadratic
loss the iterates then settle into the stationary Gaussian of an OU process
(:numref:`sec_mdl-ornstein-uhlenbeck`), whose variance grows with the ratio
of the learning rate to the batch size. :citet:`Mandt.Hoffman.Blei.2017`
use this to run SGD as an approximate posterior sampler. The model needs
small learning rates and nearly Gaussian gradient noise, and both
assumptions can fail in practice.

### Wasserstein Gradient Flows

Is there a variational reading of the Fokker--Planck equation? For a
gradient drift $\mathbf f=-\nabla V$ and a constant $g$, the equation is the
steepest descent of the free energy
$\mathcal F(p)=\int Vp\,d\mathbf x+\tfrac{g^2}2\int p\log p\,d\mathbf x$ in the
geometry of the Wasserstein-2 distance. The JKO scheme
:cite:`Jordan.Kinderlehrer.Otto.1998` discretizes it as a sequence of
proximal steps,
$p_{k+1}=\arg\min_p\,\mathcal F(p)+\tfrac1{2h}W_2^2(p,p_k)$, the analogue
for densities of the backward Euler step of
:numref:`sec_mdl-stiffness-implicit`. This is why transport, from the
continuity equation to Benamou--Brenier :cite:`Benamou.Brenier.2000`, keeps
reappearing in the chapter.

## Learned Generators

### Sliced Score Matching

Implicit score matching needs the divergence of the score network, the trace
of a $d\times d$ Jacobian, which costs $d$ backward passes. Sliced score
matching :cite:`Song.Garg.Shi.ea.2019` replaces the trace by Hutchinson's
estimate from :numref:`sec_mdl-continuous-normalizing-flows`, one
vector--Jacobian product per random direction. It learns the score of the
clean data without adding noise, at the price of the estimator's variance.

### The Diffusion Design Space

The same posterior mean can be regressed as a score, a noise, a clean
sample, a velocity, or the "$v$" combination of noise and data
:cite:`Salimans.Ho.2022`, and the noise schedule $(\alpha_t,\sigma_t)$ can
be chosen freely. :citet:`Kingma.Salimans.Poole.ea.2021` show that the
continuous-time likelihood bound depends on the schedule only through the
signal-to-noise ratio $\alpha_t^2/\sigma_t^2$ at its two ends, so schedules
are best compared in the log-SNR coordinate. EDM
:cite:`Karras.Aittala.Aila.ea.2022` fixes $\alpha_t=1$ and $\sigma_t=t$ and
chooses the preconditioning of the network, the loss weighting and a
second-order Heun sampler separately, which turns sampling into an ODE
solve to be tuned like those of :numref:`sec_mdl-euler-runge-kutta`.

### Few-Step Generators

Even a nearly straight learned field needs several evaluations per sample.
Consistency models :cite:`Song.Dhariwal.Chen.ea.2023` train a network to
map every point of a probability-flow trajectory to the trajectory's
endpoint, so one evaluation replaces the solve. Guided models can be
distilled into one network that needs one evaluation per step instead of the
two of classifier-free guidance :cite:`Meng.Rombach.Gao.ea.2023`. A student
distilled from a teacher inherits its errors, as reflow inherits those of
the first model. Latent and discrete diffusion, the forms used for images
and text, are taught in :numref:`chap_diffusion`.

### Schrödinger Bridges and Entropic Transport

The minibatch pairing of :numref:`sec_mdl-ot-connection` is an exact
assignment on each batch, and the generator it trains is an ODE. The
Schrödinger bridge asks instead for the law of paths that is closest in
relative entropy to a reference diffusion and has the two given marginals.
Its pairing is the plan of entropy-regularized optimal transport, computed
by Sinkhorn iterations :cite:`Cuturi.2013,Peyre.Cuturi.2019` rather than an
assignment, and the generator is an SDE.
:citet:`DeBortoli.Thornton.Heng.ea.2021` compute it by fitting forward and
backward diffusions in alternation.

### Stochastic Interpolants

Diffusion, an SDE reversed, and flow matching, a prescribed ODE path, look
like two theories. :citet:`Albergo.Boffi.VandenEijnden.2023` write every
path as $a_t\mathbf x_0+b_t\mathbf x_1+c_t\mathbf w$, with endpoints drawn
from any two distributions, an independent Gaussian $\mathbf w$, and
$c_t$ zero at both ends. The marginals can then be generated by an ODE or by
a family of SDEs, and the score is available as
$-\mathbb{E}[\mathbf w\mid\mathbf x_t]/c_t$.
:exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`
derives the one-sided case.

### Straight Couplings and Optimal Couplings

Repeated reflow converges to a pairing whose straight paths do not cross. In
more than one dimension such a pairing need not be optimal: the quarter-turn
pairing $\mathbf x_1=\mathbf R\mathbf x_0$ of a standard Gaussian in
:numref:`sec_mdl-ot-connection` has no crossings and costs $4$, although
the two marginals are equal and $W_2^2=0$. :citet:`Liu.2022` restricts the
velocity field to a gradient $\nabla\phi_t$. Each round still does not raise
the transport cost, and the fixed points are then exactly the optimal
couplings.

### Transformers as Interacting Particles

What does a deep stack of self-attention layers do to a cloud of token
representations? :citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` read the
layers as Euler steps of a mean-field interacting-particle ODE on the sphere
and prove that the tokens cluster as depth grows. The analysis uses the
chapter's language of fields, flows and the continuity equation on a model
that does not generate data.
