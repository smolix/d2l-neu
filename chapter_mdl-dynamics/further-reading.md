# Further Reading
:label:`sec_mdl-dynamics-further-reading`

Seven sections took us from the trajectory of one point to a flow that turns
noise into data, and they passed a good many questions without stopping.
Which stochastic integral describes the noise in a physical system? What
becomes of the adjoint when the model is itself an SDE? Each entry below
takes up one such question and says where its answer is.

Books and courses for the chapter's own material are listed in its
introduction (:numref:`chap_mdl-dynamics`). The practice of diffusion models
belongs to :numref:`chap_diffusion` and its further reading
(:numref:`sec_diffusion-further-reading`), so a method treated there gets a
pointer and no more. What remains is mathematics. We would read three
entries first: exponential integrators, one reason a diffusion sampler gets
by with few steps; Wasserstein gradient flows, which say what a diffusing
density descends; and Schrödinger bridges, the stochastic counterpart of an
optimal pairing. The rest answer questions a reader may or may not have. A
reader who stops at :numref:`sec_mdl-flow-matching-transport` misses nothing
that the rest of the book uses.

## Solvers

### Exponential Integrators

The probability-flow ODE of the VP process with a learned score
(:numref:`sec_mdl-score-based-generative-modeling`),
$\dot{\mathbf x}=-\tfrac12\beta\,\mathbf x-\tfrac12\beta\,\mathbf s_{\boldsymbol\theta}(\mathbf x,t)$,
is semilinear: a linear term known in closed form plus a learned part. A
Runge--Kutta step discretizes both, so the linear term adds to its error
and, through the stability limit of :numref:`sec_mdl-stiffness-implicit`,
bounds its step, however smooth the learned part is. That is a waste, since
the linear term can be solved. An exponential integrator treats the learned
part as the input of a driven linear equation
(:numref:`subsec_mdl-ode-driven-linear`): the variation-of-constants formula
carries the state through the linear term exactly, and only the integral of
the learned part is approximated.

Hold the learned part fixed over the step and the result is the zero-order
hold of that subsection, with the network's output as the input. The step
would be exact if that output did not change, and it is of first order
because it does. The chapter has taken this step once without the name:
dividing by $\sqrt{\bar\alpha_t}$ in :numref:`sec_mdl-ddim-guidance` removed
the linear term, and DDIM holds the noise prediction fixed over a stride.
DDIM is the exponential integrator of first order.

The samplers of this family differ in which output they hold (the score, the
noise prediction or the prediction of the clean point) and in which clock.
Methods of higher order replace the constant by a polynomial through several
predictions. :citet:`Hochbruck.Ostermann.2010` survey the theory. The
diffusion samplers built on it include DEIS :cite:`Zhang.Chen.2023` and
DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022`. The second is the subject of
Exercise :exref:`ex_samplers-dpm-solver` of
:numref:`sec_diffusion-samplers`, and its successors are listed in
:numref:`sec_diffusion-further-reading`. For an SDE the same idea gives the
exponential Euler--Maruyama step of Exercise
:exref:`ex_mdl-sdes-exact-vp-discretization` of :numref:`sec_mdl-sdes`.

### Geometric Integration and Leapfrog

On a frictionless oscillator forward Euler gains energy at every step and
backward Euler loses it, so in a long run the first flies off and the second
comes to rest. A method of higher order only postpones this. *Geometric
integration* asks something else of a method: that it preserve a structure
of the exact flow, whether or not it follows the flow closely. A
*symplectic* method, such as symplectic Euler or leapfrog, preserves area in
the plane of position and momentum, as the exact flow does (with more
coordinates the area becomes a volume).

Backward error analysis then shows that such a method follows, up to an
error that is exponentially small in $1/h$, the exact flow of a slightly
different energy. Its error in the true energy therefore stays bounded over
very long times instead of growing with the horizon. Exercise
:exref:`ex_mdl-odes-solvers-symplectic-euler` of
:numref:`sec_mdl-euler-runge-kutta` finds the different energy for
symplectic Euler on the oscillator: its level sets are ellipses close to the
circles of the true one. Gradient descent has the same analysis with a loss
in place of the energy, and the term it adds to the loss is the implicit
gradient regularizer of :citet:`Barrett.Dherin.2021` (Exercise
:exref:`ex_mdl-odes-solvers-implicit-gradient-regularization` of the same
section). :citet:`Hairer.Lubich.Wanner.2006` develop the theory.

Leapfrog is also the integrator inside Hamiltonian Monte Carlo
:cite:`Duane.Kennedy.Pendleton.ea.1987,Neal.2011`. This sampler proposes a
distant move by simulating a frictionless particle whose potential energy is
the negative log-density of the target, and accepts or rejects it by the
Metropolis rule (:numref:`sec_mdl-bayes-mcmc`). The move can be accepted
because the energy error of leapfrog stays bounded. And since a leapfrog
step is reversible and preserves volume, the rule compares two energies and
needs no Jacobian.

## Continuous-Time Models

### Flows with Cheap Determinants

Discrete normalizing flows restrict each layer so that the determinant of
its Jacobian is cheap, and :numref:`sec_mdl-continuous-normalizing-flows`
named the usual restriction, a triangular Jacobian, without saying how a
layer comes by one. A coupling layer :cite:`dinh2014nice,dinh2017density`
leaves half of the coordinates unchanged and shifts and scales the other
half by functions of the first. Its Jacobian is triangular, and the layer is
inverted by a subtraction and a division. One layer changes only half of the
state, so many are stacked, and the halves trade places from layer to layer.

An invertible residual network :cite:`Behrmann.Grathwohl.Chen.ea.2019`
restricts something else. A residual block is an Euler step
(:numref:`sec_mdl-neural-odes`), and a branch whose Lipschitz constant is
below one makes that step invertible, as the flow of a Lipschitz field is
(:numref:`subsec_mdl-ode-flow-map`). The Jacobian keeps its general shape,
and its log-determinant is estimated by the power series of Exercise
:exref:`ex_mdl-odes-solvers-residual-logdet` of
:numref:`sec_mdl-neural-odes-cnf`.
:citet:`Papamakarios.Nalisnick.Rezende.ea.2021` survey the families and what
each gives up.

### Reversible Solvers and Regularized Flows

The adjoint method of :numref:`sec_mdl-adjoint-method` saves memory by
rebuilding the forward states on the way back, and a contracting flow
amplifies the rounding error of that reconstruction. Rebuilding has a second
error, which needs no contraction. A backward step of the solver is close to
the inverse of a forward step but is not the inverse. The states that come
back are therefore not the ones that went out, and the gradient is not that
of the computation that was run. A reversible solver is built so that each
step can be undone algebraically, which removes this second error. Rounding
is still amplified, so for a stiff field checkpointing remains the safer
choice; :citet:`Kidger.2022` (ch. 5) compares the two.

A flow trained by likelihood has no reason to be cheap to integrate: the
likelihood depends on where each point ends up, not on the route it took.
:citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` therefore penalize the kinetic
energy $\int\mathbb{E}\|\mathbf f_{\boldsymbol\theta}\|^2\,dt$ during
training. Between two fixed endpoints the straight path at constant speed
has the least kinetic energy, so the penalty straightens the paths, and a
solver crosses a straight path in few steps. Flow matching pursues the same
goal through its straight regression targets and the pairing of noise with
data (:numref:`sec_mdl-ot-connection`).

### Latent and Controlled Differential Equations

The chapter's models map an initial state to a final one. A time series,
such as a patient's record, arrives at irregular times, and a model of it
should take in each observation as it comes. A latent ODE
:cite:`Rubanova.Chen.Duvenaud.2019` does half of this. It encodes the
observations into one initial state and decodes the trajectory at any set of
times, so an observation that arrives afterwards cannot change the course.

A neural controlled differential equation
:cite:`Kidger.Morrill.Foster.ea.2020` lets every observation act. It drives
the state by the interpolated data path $\mathbf X_t$ itself,
$d\mathbf z=\mathbf f_{\boldsymbol\theta}(\mathbf z)\,d\mathbf X_t$, with a
matrix-valued $\mathbf f_{\boldsymbol\theta}$. The equation has the form of
an SDE with the data in place of the Brownian path, but the interpolated
path is smooth, so the integral is the ordinary one. Its Euler step,
$\mathbf z_{n+1}=\mathbf z_n+\mathbf f_{\boldsymbol\theta}(\mathbf z_n)(\mathbf X_{n+1}-\mathbf X_n)$,
is a recurrent network that reads the increments of the data.

### Transformers as Interacting Particles

A residual network moves one state by Euler steps
(:numref:`sec_mdl-neural-odes`). A transformer moves all the tokens of a
sequence at once, and attention makes the velocity of each token depend on
where the others are. The tokens are particles that interact.
:citet:`Geshkovski.Letrouit.Polyanskiy.ea.2023` study this system in an
idealized model, in which layer normalization keeps every token on the unit
sphere, and show that the tokens gather into clusters as depth grows. The
one difference from the chapter is that the field depends on the density it
moves, so the continuity equation is no longer linear in the density.

## Noise

### Wasserstein Gradient Flows

A point on a gradient flow descends its loss
(:numref:`sec_mdl-gd-as-solver`). The density of a diffusion also moves
deterministically, by the Fokker--Planck equation. Does it descend anything?
It does when the drift is a gradient, $\mathbf f=-\nabla V$, and $g$ is
constant. The density then descends the free energy
$\mathcal F(p)=\int Vp\,d\mathbf x+\tfrac{g^2}2\int p\log p\,d\mathbf x$.
The first term is the mean potential energy, which is lowest when all the
mass sits at the bottom of $V$. The second is the negative entropy, which is
lowest when the mass is spread out. The drift serves the first and the noise
the second, and the two balance at the stationary density, which is
proportional to $e^{-2V/g^2}$.

Steepest descent depends on how distance is measured, and here the distance
between two densities has to be the Wasserstein-2 distance, which measures
the cost of transporting one into the other. The backward Euler step of
:numref:`sec_mdl-gd-as-solver` shows where the distance enters: for a convex
loss it minimizes
$\mathcal L(\mathbf x)+\tfrac1{2\eta}\|\mathbf x-\mathbf x_k\|^2$. The JKO
scheme :cite:`Jordan.Kinderlehrer.Otto.1998` is this step with densities in
place of points and the cost of transport in place of the squared distance,
$p_{k+1}=\arg\min_p\,\mathcal F(p)+\tfrac1{2\eta}W_2^2(p,p_k)$. As
$\eta\to0$ its steps converge to the solution of the Fokker--Planck
equation.

Langevin dynamics is the case to remember. Its drift is half the score of
its target and its noise has $g=1$ (:numref:`sec_mdl-langevin-sampling`).
Write $p_\infty$ for the target, to keep $p$ for the density of the cloud.
Then $V=-\tfrac12\log p_\infty$, and the free energy is
$\tfrac12D_{\textrm{KL}}(p\,\|\,p_\infty)$. A cloud of Langevin particles
runs gradient descent on its KL divergence to the target, with distance
measured by transport. This is why transport, from the continuity equation
to Benamou--Brenier :cite:`Benamou.Brenier.2000`, keeps reappearing in the
chapter.

### Physical Noise and the Wong--Zakai Theorem

:numref:`sec_mdl-ito-integral` defines the Itô integral with the left
endpoint and notes that the midpoint rule, which defines the Stratonovich
integral, keeps the ordinary chain rule. Which of the two describes a
physical system? The noise in a physical system is not a Brownian path. It
has a short memory and a smooth path, and for a smooth path there is only
one integral, the ordinary one. :citet:`Wong.Zakai.1965` proved that as such
paths approach a Brownian path, the solutions of the ordinary equations
converge to the Stratonovich solution. In one dimension the Stratonovich
reading of $dX=f\,dt+g\,dW$ is the Itô equation with the drift
$f+\tfrac12g\,\partial_xg$.

The two readings agree for the additive noise of diffusion models. For
multiplicative noise they differ, and the theorem says how to choose. An
equation that idealizes smooth physical noise is read as Stratonovich. A
model built in discrete time, whose coefficient cannot see the increment it
multiplies, is read as Itô, and a step of stochastic gradient descent is of
this kind.

### Stochastic Gradient Descent as an SDE

Gradient descent is forward Euler on the gradient flow
(:numref:`sec_mdl-gd-as-solver`). Of what is stochastic gradient descent the
discretization? A minibatch gradient is the full gradient plus noise, with a
covariance $\boldsymbol\Sigma$ that shrinks as $1/B$ with the batch size
$B$. :citet:`Li.Tai.E.2017` propose the SDE
$d\boldsymbol\theta=-\nabla\mathcal L\,dt+\sqrt{\eta}\,\boldsymbol\Sigma^{1/2}d\mathbf W$.
Take an Euler--Maruyama step of length $\eta$. It multiplies the noise
coefficient by another $\sqrt\eta$, and the result is one step of stochastic
gradient descent with learning rate $\eta$.

The learning rate therefore appears twice, as the length of the step and
inside the noise. When $\boldsymbol\Sigma$ is the same multiple
$c\,\mathbf I$ of the identity everywhere, the free energy above applies
with $V=\mathcal L$ and $g^2=\eta c$, which weighs the entropy against the
loss. In this sense the learning rate is a temperature. On a quadratic loss
the iterates settle into the stationary Gaussian of an OU process
(:numref:`sec_mdl-ornstein-uhlenbeck`), whose variance grows with the ratio
$\eta/B$. :citet:`Mandt.Hoffman.Blei.2017` use this to run stochastic
gradient descent as an approximate posterior sampler. The model needs small
learning rates and nearly Gaussian gradient noise, and both assumptions can
fail in practice.

### Neural SDEs

What becomes of the adjoint when the learned model is itself an SDE, with a
learned drift and a learned noise amplitude $g$? It becomes a backward SDE
driven by the same Brownian path, which must be stored or regenerated from
its random seed :cite:`Li.Wong.Chen.ea.2020`. The loss changes as well. The
output of such a model is random, so one endpoint cannot be compared with
one target. The training signal is a divergence between the laws of whole
paths, or a discriminator that compares them :cite:`Kidger.2022`.

Girsanov's theorem :cite:`Oksendal.2003` gives this divergence in closed
form for two processes with the same $g$ and the drifts $\mathbf f$ and
$\mathbf f'$. The KL divergence between their path laws is the KL divergence
between their starting laws plus
$\tfrac12\int\mathbb{E}\|\mathbf f-\mathbf f'\|^2/g^2\,dt$, with the
expectation over paths of the first process. The condition on $g$ cannot be
dropped: two processes with different $g$ can be told apart from a single
path, by its quadratic variation (:numref:`sec_mdl-quadratic-variation`),
and the KL divergence between their path laws is infinite.

The laws of the endpoints cannot differ by more than the laws of the paths,
by the data-processing inequality of
:numref:`subsec_mdl-f-div-data-processing`. Applied to the exact and the
learned reverse drift of a diffusion model, Girsanov's formula therefore
gives the likelihood bound that Exercise
:exref:`ex_mdl-score-matching-diffusion-flow-likelihood-bound` of
:numref:`sec_mdl-score-matching-diffusion-flow` derives without path laws.

## Learned Generators

Most of what follows the chapter in this direction is taught in
:numref:`chap_diffusion`. Prediction targets, loss weightings and schedules
are in :numref:`sec_diffusion-design-space`, where the logarithm of the
signal-to-noise ratio, $-2\log\bar\sigma_t$ in the notation of
:numref:`sec_mdl-ddim-guidance`, orders the choices
:cite:`Kingma.Salimans.Poole.ea.2021`. Generation in one or a few steps is
in :numref:`sec_diffusion-few-step`. Four topics lie beyond both chapters.

### Sliced Score Matching

:numref:`sec_mdl-score-matching` set implicit score matching aside for the
cost of its trace, and mentioned a way out that it did not take:
Hutchinson's estimate, one vector--Jacobian product per random direction.
Trained with that estimate, the method is called sliced score matching
:cite:`Song.Garg.Shi.ea.2019`. It learns the score of the data themselves,
with no noise level to choose. A diffusion model has no use for this
advantage. It needs the scores of the noised data at every level, and
denoising score matching delivers those without a trace, which is why
denoising became the standard.

### Score Error and Sampling Error

Nearly every sampler of the chapter ran on an exact score, and a trained
score is exact nowhere. How good must it be? Good on average, and no more.
:citet:`Chen.Chewi.Li.ea.2023` bound the total variation between the data
and the output of the discretized reverse SDE by a sum of three terms, and
the chapter has met each of them alone. The distance of $p_T$ from the
Gaussian at which the sampler starts decays like $e^{-T}$, and the
discretization error grows with the dimension and falls with the step;
:numref:`sec_mdl-time-reversal` measured both. The third term is the root
mean squared error of the score under the noised data, which is what a
denoising loss measures above its floor. The theorem says that nothing else
enters, and it asks little: a finite second moment of the data and Lipschitz
scores of the noised densities. A single mode is not required, so the two
modes that defeated Langevin dynamics in :numref:`sec_mdl-langevin-sampling`
are covered.

### Schrödinger Bridges and Entropic Transport

In 1931 Schrödinger asked a question about particles. A cloud that diffuses
freely is seen in one shape now and in another, improbable one later: what
is the most likely way it got there? For a large cloud of independent
particles the probability of seeing the cloud follow some other law of paths
falls exponentially, at a rate given by the KL divergence of that law from
the free diffusion. The answer is therefore the law of paths that is closest
in KL divergence to the free diffusion among those with the two given
marginals. It is called the *Schrödinger bridge*.

Set the bridge beside the minibatch pairing of
:numref:`sec_mdl-ot-connection`. That pairing is an exact assignment on each
batch, and the generator it trains is an ODE. The bridge pairs its two ends
by the plan of entropy-regularized optimal transport, computed by the
Sinkhorn iterations of :numref:`subsec_mdl-sinkhorn`
:cite:`Cuturi.2013,Peyre.Cuturi.2019`, and its generator is an SDE. As the
noise of the free diffusion shrinks, the plan tends to an optimal one.

Set beside a diffusion model, the bridge is the harder problem. A diffusion
model fixes its forward process, learns only the reverse, and reaches its
prior only approximately. The bridge joins two given laws in finite time, so
both directions must be learned. :citet:`DeBortoli.Thornton.Heng.ea.2021`
fit forward and backward diffusions in alternation, and the first round of
the alternation is the training of a diffusion model.
:citet:`Albergo.Boffi.VandenEijnden.2023` reach both kinds of generator, ODE
and SDE, without solving a bridge problem. Take samples $\mathbf x_0$ and
$\mathbf x_1$ of any two distributions and an independent Gaussian
$\mathbf w$. The path $a_t\mathbf x_0+b_t\mathbf x_1+c_t\mathbf w$, with
$c_t$ zero at both ends, has marginals that an ODE or a family of SDEs can
generate (Exercise
:exref:`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants` of
:numref:`sec_mdl-flow-matching-transport`).

### Straight Couplings and Optimal Couplings

A pairing that reflow leaves unchanged has straight paths that do not cross,
and in more than one dimension it need not be optimal. The example of
:numref:`sec_mdl-ot-connection` pairs each point of a Gaussian cloud with
its quarter turn. The segments never cross, and they cost $4$ where
$W_2^2=0$. What that pairing has and an optimal one lacks is rotation. The
velocity field of an optimal flow is a gradient
:cite:`Benamou.Brenier.2000`, and a field that turns a cloud about the
origin is not.

:citet:`Liu.2022` therefore restricts the velocity field to a gradient
$\nabla\phi_t$ and argues that the pairings this restricted reflow leaves
unchanged are the optimal ones. The statement needs more than it assumes.
:citet:`Hertrich.Chambolle.Delon.2025` prove it when the cloud in transit
covers all of space at every time, and give counterexamples in which the
cloud consists of separate pieces. So the restriction does not settle the
question that
:numref:`sec_mdl-ot-connection` left: how to find an optimal pairing by
training alone.

[Discussions](https://d2l.discourse.group/t/36870)
