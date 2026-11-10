# Further Reading
:label:`sec_mdl-dynamics-further-reading`

The list in the chapter introduction (:numref:`chap_mdl-dynamics`) names
books, courses and papers that teach again what the chapter taught. This
section points past the chapter, and past :numref:`chap_diffusion` as well:
where that chapter or its own further reading
(:numref:`sec_diffusion-further-reading`) covers a topic, we give the pointer
and no more. What remains is mathematics, and it is not of equal weight.
We would read three entries first: exponential integrators, one reason a
diffusion sampler gets by with few steps; Wasserstein gradient flows, the
reason transport kept reappearing in the chapter; and Schrödinger bridges,
the stochastic counterpart of an optimal pairing. The rest answer questions
a reader may or may not have. Nothing else in the book depends on this section.

## Solvers

### Exponential Integrators

The field of a diffusion sampler is semilinear: a linear term known in
closed form plus a learned nonlinear part. An explicit Runge--Kutta step
must stay inside the stability region set by the linear part
(:numref:`sec_mdl-stiffness-implicit`), however smooth the learned part is.
An exponential integrator solves the linear part exactly, through the
variation-of-constants formula :eqref:`eq_mdl-ode-variation-of-constants`,
and approximates only the integral of the nonlinear remainder.
:citet:`Hochbruck.Ostermann.2010` survey the method. Its use on the
probability-flow ODE, in DEIS :cite:`Zhang.Chen.2023` and DPM-Solver
:cite:`Lu.Zhou.Bao.ea.2022`, is taken up in :numref:`sec_diffusion-samplers`.
For SDEs, the stochastic Taylor expansion behind Milstein's correction
continues to higher orders, but the terms beyond Milstein's involve iterated
stochastic integrals that are expensive to sample
:cite:`Kloeden.Platen.1992`.

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
:exref:`ex_mdl-odes-solvers-residual-logdet`.
:citet:`Papamakarios.Nalisnick.Rezende.ea.2021` survey the families and what
each gives up.

### Reversible Solvers and Regularized Flows

The adjoint method of :numref:`sec_mdl-adjoint-method` rebuilds the forward
states by integrating backward, and a strongly contracting flow amplifies the
error of that reconstruction. A reversible solver recovers its own forward
steps from the endpoint up to rounding, so the backward pass carries no
drift; :citet:`Kidger.2022` (ch. 5) compares it with checkpointing.

A flow trained by likelihood also has no reason to be cheap to integrate.
Penalizing the kinetic energy
$\int\|\mathbf f_{\boldsymbol\theta}\|^2\,dt$ during training
:cite:`Finlay.Jacobsen.Nurbekyan.ea.2020` pushes it toward straight paths,
the goal that flow matching reaches by choosing its regression target
(:numref:`sec_mdl-ot-connection`).

### Latent and Controlled Differential Equations

The chapter's models map an initial state to a final one, but time series
arrive at irregular times and must be read as they come. A latent ODE
:cite:`Rubanova.Chen.Duvenaud.2019` encodes the observations into an
initial state and decodes the trajectory at any set of times. A neural
controlled differential equation :cite:`Kidger.Morrill.Foster.ea.2020`
drives the state by the interpolated data path itself,
$d\mathbf z=\mathbf f_{\boldsymbol\theta}(\mathbf z)\,d\mathbf X_t$, which
is the continuous-time counterpart of a recurrent network.

### Transformers as Interacting Particles

:citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` read a deep stack of
self-attention layers as Euler steps of an interacting-particle ODE for the
token representations on the sphere, and study how the tokens cluster as
depth grows. The analysis uses the chapter's language of fields, flows and
the continuity equation on a model that does not generate data.

## Noise

### Wasserstein Gradient Flows

The Fokker--Planck equation has a variational reading. For a gradient drift
$\mathbf f=-\nabla V$ and a constant $g$, it is the steepest descent of the
free energy
$\mathcal F(p)=\int Vp\,d\mathbf x+\tfrac{g^2}2\int p\log p\,d\mathbf x$ in the
geometry of the Wasserstein-2 distance. The JKO scheme
:cite:`Jordan.Kinderlehrer.Otto.1998` discretizes it as a sequence of
proximal steps,
$p_{k+1}=\arg\min_p\,\mathcal F(p)+\tfrac1{2h}W_2^2(p,p_k)$, the analogue
for densities of the backward Euler step of
:numref:`sec_mdl-stiffness-implicit`. This is why transport, from the
continuity equation to Benamou--Brenier :cite:`Benamou.Brenier.2000`, keeps
reappearing in the chapter.

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

### Neural SDEs

When the learned model is itself an SDE, with a learned drift and a learned
diffusion, the adjoint of :numref:`sec_mdl-adjoint-method` becomes a backward
SDE driven by the same Brownian path, which must be stored or regenerated
from its random seed :cite:`Li.Wong.Chen.ea.2020`. The training signal is
then a divergence between laws of whole paths, or a discriminator that
compares them, rather than a loss on one endpoint :cite:`Kidger.2022`. For
two diffusions with the same noise, Girsanov's theorem gives that divergence
as an integral of the squared difference of their drifts
:cite:`Oksendal.2003`.

## Learned Generators

Most of what follows the chapter in this direction is taught in
:numref:`chap_diffusion`: prediction targets, loss weightings and schedules
in :numref:`sec_diffusion-design-space`, where the log-SNR coordinate of
:citet:`Kingma.Salimans.Poole.ea.2021` orders the choices, and generation in
one or a few steps in :numref:`sec_diffusion-few-step`. Three topics lie
beyond both chapters.

### Schrödinger Bridges and Entropic Transport

The minibatch pairing of :numref:`sec_mdl-ot-connection` is an exact
assignment on each batch, and the generator it trains is an ODE. The
Schrödinger bridge asks instead for the law of paths that is closest in
relative entropy to a reference diffusion and has the two given marginals.
Its pairing is the plan of entropy-regularized optimal transport, computed
by the Sinkhorn iterations of :numref:`subsec_mdl-sinkhorn`
:cite:`Cuturi.2013,Peyre.Cuturi.2019` rather than an assignment, and the
generator is an SDE. :citet:`DeBortoli.Thornton.Heng.ea.2021` compute it by
fitting forward and backward diffusions in alternation.
:citet:`Albergo.Boffi.VandenEijnden.2023` reach both kinds of generator from
one construction: a path $a_t\mathbf x_0+b_t\mathbf x_1+c_t\mathbf w$
between any two distributions, with independent Gaussian $\mathbf w$ and
$c_t$ zero at both ends, whose marginals an ODE or a family of SDEs can
generate (:exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`
works through the construction).

### Straight Couplings and Optimal Couplings

A pairing that reflow leaves unchanged has straight paths that do not cross,
and in more than one dimension it need not be optimal: the quarter-turn
pairing of :numref:`sec_mdl-ot-connection` costs $4$ where $W_2^2=0$.
:citet:`Liu.2022` restricts the velocity field to a gradient
$\nabla\phi_t$ and argues that the pairings this restricted reflow leaves
unchanged are the optimal ones.

### Sliced Score Matching

Implicit score matching needs the divergence of the score network, the trace
of a $d\times d$ Jacobian, which costs $d$ backward passes. Sliced score
matching :cite:`Song.Garg.Shi.ea.2019` replaces the trace by Hutchinson's
estimate from :numref:`sec_mdl-continuous-normalizing-flows`, one
vector--Jacobian product per random direction. It learns the score of the
clean data without adding noise, at the price of the estimator's variance.

[Discussions](https://d2l.discourse.group/t/36870)
