# Ordinary Differential Equations and Numerical Solvers
:label:`sec_mdl-odes-solvers`

Every continuous-time generative model — Neural ODEs, continuous normalizing
flows, and the deterministic samplers of diffusion and flow matching — is an
**ordinary differential equation that you integrate**. This section builds the
math of that integration from the ground up: what a vector field is, when a
trajectory exists and is unique, how eigenvalues decide stability, and how
numerical solvers (Euler, Runge–Kutta) trade step size for accuracy. The payoff
is conceptual unification — *a residual block is one Euler step*, *backprop
through an ODE is reverse-mode AD*, and *the discrete log-det-Jacobian of a
normalizing flow becomes a trace integral* — which is exactly the toolkit §6.3
and §6.4 need to turn diffusion into a solvable ODE.

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §6.1, the first section of the
Dynamics chapter (:numref:`chap_mdl-dynamics`). No prose, code, or figures have
been authored yet; every subsection below is a planning stub in the standard
format. The body framing for the section as a whole:

> *Every continuous-time generative model is an ODE you integrate; this section
> tells you why a ResNet block is an Euler step and why sampler choice trades
> speed for accuracy.*

**Prerequisites:** 1.2 (`sec_mdl-eigendecompositions`: eigenvalues, matrix
exponential, spectral radius); 2.1–2.2 (`sec_mdl-multivariable_calculus`:
Jacobian, trace, Taylor); 2.3 (`sec_mdl-matrix-calculus-autodiff`: reverse-mode
AD as vector–Jacobian products); 2.4 (`sec_mdl-integral_calculus`:
integration, change of variables); 4.1 (`sec_mdl-random_variables`: the
change-of-variables formula for densities).
**Capstone payoff:** continuous normalizing flows (§6.1.8) and the Neural-ODE /
adjoint view (§6.1.6–6.1.7), reused directly by the probability-flow ODE (§6.3)
and learned-dynamics sampling (§6.4).
**Forward bridge:** the linear-stability analysis (§6.1.3) explains the
mean-reversion of the Ornstein–Uhlenbeck forward process (§6.2.7); the solver
error orders (§6.1.4) set the step/quality tradeoff for diffusion samplers
(§6.4.8).
:::

## Vector Fields and Trajectories
:label:`sec_mdl-vector-fields-trajectories`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** An ODE is a *velocity rule*: at every point of space (and
time) it tells you which way and how fast to move. A solution is just the path
you trace by always following the local arrow.
**Outline:** 1. the initial-value problem $\dot{\mathbf{x}}(t)=\mathbf{f}(\mathbf{x},t)$,
$\mathbf{x}(0)=\mathbf{x}_0$ — read $\mathbf{f}$ as a (possibly
time-dependent) vector field · 2. a *trajectory* / integral curve is a function
$\mathbf{x}(t)$ whose tangent matches the field everywhere · 3. autonomous
($\mathbf{f}(\mathbf{x})$) vs. non-autonomous ($\mathbf{f}(\mathbf{x},t)$);
the *flow map* $\Phi_t:\mathbf{x}_0\mapsto\mathbf{x}(t)$ and why it is a
bijection when the field is nice (sets up invertible generative flows) · 4. the
integral form $\mathbf{x}(t)=\mathbf{x}_0+\int_0^t\mathbf{f}(\mathbf{x}(s),s)\,ds$
(ties to `sec_mdl-integral_calculus`).
**Key results to state:** $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$;
$\mathbf{x}(t)=\mathbf{x}_0+\int_0^t \mathbf{f}(\mathbf{x}(s),s)\,ds$;
flow map $\Phi_t$ with $\Phi_0=\mathrm{id}$, $\Phi_{t+s}=\Phi_t\circ\Phi_s$.
**Diagrams:** `fig_mdl-vector-field-flow` — a 2-D vector field (arrow grid) with
several integral curves threaded through it from different start points, showing
the field "steering" each trajectory. :numref:`fig_mdl-dyn-ode-field` shows this
for a linear spiral-sink field.

![An ODE as a velocity field. The faint grid arrows are the field $\mathbf{f}(\mathbf{x})=A\mathbf{x}$ with $A=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$; the two coloured integral curves are trajectories that always follow the local arrow, spiralling into the stable fixed point at the origin (eigenvalues $-0.5\pm i$).](../img/mdl-dyn-ode-field.svg)
:label:`fig_mdl-dyn-ode-field`
**Worked example(s):** sketch and integrate $\dot x=-x$ (exponential decay) and a
rotational field $\dot{\mathbf{x}}=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)\mathbf{x}$
(circular orbits) — framework-agnostic plots of the field plus a few curves.
**Exercises (draft):** (1) verify $\mathbf{x}(t)=e^{-t}\mathbf{x}_0$ solves
$\dot{\mathbf{x}}=-\mathbf{x}$; (2) show the rotational field's trajectories are
circles of constant radius; (3) write the integral form for $\dot x = t$.
**Prereqs / cross-refs:** `sec_mdl-integral_calculus` (integral form);
forward to §6.1.6 (the field becomes a neural network $\mathbf{f}_\theta$).
:::

## Existence and Uniqueness
:label:`sec_mdl-ode-existence-uniqueness`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Before we trust "follow the arrows," we need to know a unique
path actually exists. The Picard–Lindelöf theorem says a Lipschitz velocity
field guarantees exactly one trajectory through each point — which is precisely
what makes a generative flow *invertible* (no two source points collide on the
same target).
**Outline:** 1. state Picard–Lindelöf: $\mathbf{f}$ continuous in $t$ and
Lipschitz in $\mathbf{x}$ $\Rightarrow$ a unique solution on some interval ·
2. intuition via the contraction-mapping / Picard-iteration picture (the
integral operator is a contraction) · 3. *non-uniqueness* when Lipschitz fails:
$\dot x=\sqrt{|x|}$ leaks off $x=0$; *finite-time blow-up* when growth is
super-linear: $\dot x=x^2$ escapes to $\infty$ in finite time · 4. why this
licenses invertible flows: the flow map $\Phi_t$ is a bijection with inverse
$\Phi_{-t}$ (run the ODE backward).
**Key results to state:** Lipschitz $\|\mathbf{f}(\mathbf{x},t)-\mathbf{f}(\mathbf{y},t)\|\le L\|\mathbf{x}-\mathbf{y}\|$
$\Rightarrow$ unique solution (Picard–Lindelöf); $\dot x=x^2,\,x_0>0$ blows up at
$t=1/x_0$; $\Phi_t$ invertible with $\Phi_t^{-1}=\Phi_{-t}$.
**Diagrams:** `fig_mdl-existence-uniqueness` — left: a unique curve through each
point (Lipschitz field); right: the $\dot x=\sqrt{|x|}$ fan of solutions all
leaving the origin (uniqueness failure).
**Worked example(s):** exhibit two distinct solutions of $\dot x=\sqrt{|x|}$ from
$x_0=0$; compute the blow-up time of $\dot x=x^2$ in closed form.
**Exercises (draft):** (1) verify the Lipschitz constant of a linear field is
$\|A\|$; (2) construct a second solution of $\dot x=\sqrt{|x|}$; (3) show
$\dot x=x^2$ escapes at $t=1/x_0$ and explain why no Lipschitz constant works.
**Prereqs / cross-refs:** `sec_mdl-multivariable_calculus` (Lipschitz / norms);
3.2 forward (contraction-mapping flavor); forward to §6.1.6 (Lipschitz
$\mathbf{f}_\theta$ keeps the Neural ODE well-posed and invertible).
:::

## Linear ODEs and Stability
:label:`sec_mdl-linear-odes-stability`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The one ODE we can solve in closed form, $\dot{\mathbf{x}}=A\mathbf{x}$,
is also the one that tells us everything about *local* behavior of nonlinear
fields. Its solution is the matrix exponential $e^{At}$, and the **eigenvalues of
$A$** (callback to `sec_mdl-eigendecompositions`) decide whether trajectories
decay, grow, or oscillate — the same spectral story that governs the OU forward
process in §6.2.
**Outline:** 1. solve $\dot{\mathbf{x}}=A\mathbf{x}$ as $\mathbf{x}(t)=e^{At}\mathbf{x}_0$;
define $e^{At}=\sum_k (At)^k/k!$ · 2. diagonalize $A=W\Lambda W^{-1}$ so
$e^{At}=W e^{\Lambda t}W^{-1}$ — each eigenmode evolves as $e^{\lambda_i t}$
(tie to `sec_mdl-eigendecompositions`) · 3. the stability dictionary:
$\operatorname{Re}(\lambda_i)<0$ decays to the fixed point, $>0$ grows,
$=0$ marginal; imaginary part $\Rightarrow$ rotation/oscillation · 4. linearize a
nonlinear field at a fixed point via its Jacobian; eigenvalues of the Jacobian
give local stability · 5. preview: a scalar contracting mode $\dot x=-\theta x$
is the deterministic skeleton of the OU process (§6.2.7).
**Key results to state:** $\mathbf{x}(t)=e^{At}\mathbf{x}_0$;
$e^{At}=W e^{\Lambda t}W^{-1}$ with modes $e^{\lambda_i t}$;
asymptotic stability $\iff \operatorname{Re}(\lambda_i)<0\ \forall i$;
fixed-point stability from $\operatorname{Re}$ of $\operatorname{eig}(D\mathbf{f})$.
**Diagrams:** `fig_mdl-linear-ode-phase-portraits` — the standard 2-D phase-portrait
gallery (stable node, unstable node, saddle, stable spiral, center) labeled with
the eigenvalue signature that produces each.
**Worked example(s):** solve $\dot{\mathbf{x}}=A\mathbf{x}$ for a $2\times2$ $A$
by eigendecomposition; classify three fixed points by Jacobian eigenvalues;
plot $e^{\lambda t}$ for real-negative, real-positive, and complex $\lambda$.
**Exercises (draft):** (1) show $e^{At}$ commutes with $A$ and
$\frac{d}{dt}e^{At}=Ae^{At}$; (2) classify the phase portrait of a given $2\times2$
matrix; (3) for a damped oscillator, read decay rate and frequency off the
eigenvalues.
**Prereqs / cross-refs:** `sec_mdl-eigendecompositions` (eigendecomposition,
matrix exponential, spectral radius); `sec_mdl-multivariable_calculus`
(Jacobian); forward to §6.2.7 (OU mean reversion) and §6.1.5 (stiffness =
widely-separated $\operatorname{Re}\lambda_i$).
:::

## Explicit Solvers: Euler and Runge–Kutta
:label:`sec_mdl-euler-runge-kutta`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Almost no ODE has a closed-form solution, so we *march*: take
the local velocity, step, repeat. Euler follows the single arrow at the current
point; Runge–Kutta averages several probe slopes for a far more accurate step.
The error-vs-step-size scaling (order of the method) is the knob that sets the
speed/quality tradeoff in every diffusion sampler.
**Outline:** 1. forward Euler $\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}(\mathbf{x}_n,t_n)$
as a truncated Taylor step · 2. *local truncation error* $O(h^2)$ per step vs.
*global error* $O(h)$ over a fixed interval (one order lost to accumulation) ·
3. the midpoint/RK2 idea: probe the slope at the half-step · 4. classical RK4 —
the four-slope weighted average $\frac{h}{6}(k_1+2k_2+2k_3+k_4)$ — with global
error $O(h^4)$ · 5. the general "order $p$" statement: global error $\sim C h^p$,
so a log–log error-vs-$h$ plot has slope $p$.
**Key results to state:** Euler $\mathbf{x}_{n+1}=\mathbf{x}_n+h\mathbf{f}(\mathbf{x}_n,t_n)$,
global error $O(h)$; RK4 update with $k_1{=}\mathbf{f}(\mathbf{x}_n,t_n)$,
$k_2{=}\mathbf{f}(\mathbf{x}_n+\tfrac{h}{2}k_1,t_n+\tfrac{h}{2})$, …,
$\mathbf{x}_{n+1}=\mathbf{x}_n+\tfrac{h}{6}(k_1+2k_2+2k_3+k_4)$, global error
$O(h^4)$; general global error $\sim C h^p$.
**Diagrams:** `fig_mdl-euler-error-vs-h` — log–log plot of global error vs. step
size $h$ for Euler and RK4 on a test ODE, reference lines of slope $1$ and $4$
confirming the orders (with an inset of an Euler step "overshooting" a curved
trajectory).
**Worked example(s):** integrate $\dot x=-\lambda x$ (known solution $e^{-\lambda t}$)
with Euler vs. RK4 across a grid of $h$; measure error and fit the log–log slopes
(≈1 and ≈4). Framework-agnostic.
**Exercises (draft):** (1) derive the Euler $O(h^2)$ local / $O(h)$ global error
from Taylor; (2) read the convergence order off a log–log error plot; (3) show RK2
midpoint is second order; (4) count function evaluations per step for Euler vs.
RK4 and discuss the cost/accuracy trade.
**Prereqs / cross-refs:** `sec_mdl-single_variable_calculus` (Taylor),
`sec_mdl-integral_calculus` (the integral being approximated); forward to §6.2.6
(Euler–Maruyama = Euler + noise) and §6.4.8 (few-step ODE sampling).
:::

## Adaptive Steps, Stiffness, and Implicit Methods
:label:`sec_mdl-stiffness-implicit`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Real solvers do not use a fixed $h$ — they estimate their own
error and shrink the step where the dynamics are fast. And some systems are
*stiff*: explicit methods need absurdly tiny steps to stay stable, which is when
*implicit* methods earn their keep.
**Outline:** 1. embedded error estimates and adaptive step control (e.g.
Runge–Kutta–Fehlberg / Dormand–Prince `RK45`): compare two orders, accept/reject
the step, retune $h$ · 2. *stiffness*: when $A$ has eigenvalues of wildly
different magnitudes (huge $\operatorname{Re}\lambda$ spread, tie to §6.1.3), the
fastest mode caps the *stable* step even after it has decayed · 3. the explicit
stability limit: Euler on $\dot x=-\lambda x$ is stable only for
$h<2/\lambda$ (it amplifies otherwise) · 4. implicit / backward Euler
$\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}(\mathbf{x}_{n+1},t_{n+1})$ — solve a
(possibly nonlinear) equation per step, but gain *A-stability* (stable for any
$h>0$ on the linear test problem) · 5. practical guidance: reach for adaptive
explicit solvers by default, implicit when stiff.
**Key results to state:** explicit-Euler stability on $\dot x=-\lambda x$ requires
$|1-h\lambda|<1 \iff h<2/\lambda$; backward Euler amplification
$1/(1+h\lambda)$ is $<1$ for all $h>0$ (A-stable); adaptive control adjusts $h$ to
hold local error under a tolerance.
**Diagrams:** `fig_mdl-stiff-explicit-vs-implicit` — a stiff 2-D system where
explicit Euler at moderate $h$ oscillates and diverges while backward Euler stays
on the slow manifold; inset showing the $|1-h\lambda|$ vs. $1/(1+h\lambda)$
amplification factors.
**Worked example(s):** a stiff 2-D linear system ($\lambda_1\!\gg\!\lambda_2$)
where explicit Euler blows up at a step size backward Euler handles cleanly;
sweep $h$ to find the explicit stability threshold $2/\lambda$.
**Exercises (draft):** (1) derive the $h<2/\lambda$ stability bound for explicit
Euler; (2) show backward Euler is unconditionally stable on the linear test
equation; (3) explain why a 100:1 eigenvalue ratio makes an explicit solver slow;
(4) one backward-Euler step requires solving what equation when $\mathbf{f}$ is
nonlinear?
**Prereqs / cross-refs:** §6.1.3 (eigenvalues set the time scales);
`sec_mdl-eigendecompositions`; forward to §6.4.8 (why some learned ODEs are easy
to integrate in few steps and others are not).
:::

## Continuous-Depth Models: Neural ODEs
:label:`sec_mdl-neural-odes`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The bridge from ODEs to deep learning: a residual block
$\mathbf{x}_{n+1}=\mathbf{x}_n+\mathbf{f}_\theta(\mathbf{x}_n)$ is *exactly one
forward-Euler step* of $\dot{\mathbf{x}}=\mathbf{f}_\theta(\mathbf{x},t)$ with
$h=1$. Taking the depth-to-infinity limit replaces a stack of layers with a
single learned vector field integrated by an ODE solver (Chen et al., 2018).
**Outline:** 1. ResNet update = Euler step (compare to §6.1.4) · 2. the
continuous-depth limit $\dot{\mathbf{x}}=\mathbf{f}_\theta(\mathbf{x},t)$; "layers"
become integration time, "depth" becomes the time horizon · 3. the forward pass
is *calling an ODE solver* (adaptive solver $\Rightarrow$ adaptive effective
depth) · 4. parameter efficiency and the smooth, invertible map (Lipschitz
$\mathbf{f}_\theta$, §6.1.2) · 5. pointer: this is the *engineering* covered by
the main book's flow/Neural-ODE material — here we teach only the math.
**Key results to state:** ResNet
$\mathbf{x}_{n+1}=\mathbf{x}_n+\mathbf{f}_\theta(\mathbf{x}_n)$ $\equiv$ Euler
with $h=1$; continuous limit $\dot{\mathbf{x}}=\mathbf{f}_\theta(\mathbf{x},t)$,
output $\mathbf{x}(T)=\mathbf{x}_0+\int_0^T\mathbf{f}_\theta(\mathbf{x}(t),t)\,dt$
(Chen, Rubanova, Bettencourt, Duvenaud, *Neural ODEs*, NeurIPS 2018).
**Diagrams:** `fig_mdl-resnet-as-euler` — a stack of residual blocks on the left
morphing into a continuous trajectory through a vector field on the right, with
the skip connection annotated as the "+ $h\mathbf{f}$" Euler increment.
**Worked example(s):** show numerically that stacking $N$ residual blocks
$\mathbf{x}+\tfrac1N\mathbf{f}_\theta(\mathbf{x})$ approaches the ODE solution as
$N\to\infty$; a tiny 2-D Neural ODE flowing an inner ring to an outer ring.
**Exercises (draft):** (1) write the residual update as an Euler step and identify
$h$; (2) what does halving the solver step size correspond to in "layers"?;
(3) why does a Lipschitz $\mathbf{f}_\theta$ guarantee the map is invertible
(callback §6.1.2)?; (4) contrast fixed-depth ResNet with adaptive-solver Neural
ODE in compute.
**Prereqs / cross-refs:** §6.1.2 (well-posedness/invertibility), §6.1.4 (Euler);
main-book residual-network and generative-flow chapters (engineering); forward to
§6.1.7 (how to train it) and §6.1.8 (turn it into a density model).
:::

## The Adjoint Method = Reverse-Mode AD Through the ODE
:label:`sec_mdl-adjoint-method`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** To train a Neural ODE we need gradients of a loss on
$\mathbf{x}(T)$ with respect to $\theta$ — without storing every internal solver
state. The **adjoint method** computes them by integrating a second ODE
*backward in time*. This is nothing other than reverse-mode AD lifted to
continuous time (tie to `sec_mdl-matrix-calculus-autodiff`): the adjoint
$\mathbf{a}(t)=\partial L/\partial\mathbf{x}(t)$ is the continuous analogue of the
backprop "delta."
**Outline:** 1. define the adjoint $\mathbf{a}(t)=\partial L/\partial \mathbf{x}(t)$
· 2. it obeys its own ODE $\dot{\mathbf{a}}=-\mathbf{a}^\top\,\partial\mathbf{f}/\partial\mathbf{x}$
run backward from $\mathbf{a}(T)=\partial L/\partial\mathbf{x}(T)$ · 3. the
parameter gradient is an integral
$\partial L/\partial\theta=-\int_T^0 \mathbf{a}^\top\,\partial\mathbf{f}/\partial\theta\,dt$
· 4. the punchline: each adjoint step is a *vector–Jacobian product*, the same
primitive as reverse-mode AD (callback to `sec_mdl-matrix-calculus-autodiff`) —
backprop *is* the discrete adjoint · 5. the memory/accuracy tradeoff:
$O(1)$ memory by re-integrating $\mathbf{x}$ backward, vs. numerical error if the
reverse trajectory drifts (when to just store activations instead).
**Key results to state:** adjoint dynamics
$\dot{\mathbf{a}}(t)=-\mathbf{a}(t)^\top\dfrac{\partial\mathbf{f}}{\partial\mathbf{x}}$,
$\mathbf{a}(T)=\dfrac{\partial L}{\partial\mathbf{x}(T)}$;
$\dfrac{\partial L}{\partial\theta}=-\displaystyle\int_T^0 \mathbf{a}(t)^\top\dfrac{\partial\mathbf{f}}{\partial\theta}\,dt$;
each evaluation is a VJP $\Rightarrow$ adjoint method = continuous reverse-mode AD
(Chen et al., 2018).
**Diagrams:** `fig_mdl-adjoint-reverse-ad` — forward trajectory left-to-right,
adjoint trajectory right-to-left underneath, with the per-step VJP arrows linking
the two, captioned "backprop = discrete adjoint."
**Worked example(s):** for the linear ODE $\dot{\mathbf{x}}=A\mathbf{x}$, derive
the adjoint $\dot{\mathbf{a}}=-A^\top\mathbf{a}$ in closed form and check it
against reverse-mode AD on the unrolled Euler solver.
**Exercises (draft):** (1) derive the adjoint ODE from the Lagrangian / variation
of the loss; (2) show one adjoint step equals one VJP; (3) compare adjoint
($O(1)$ memory, re-integrate) vs. store-all-activations on a deep solve;
(4) where does reverse-trajectory numerical error come from?
**Prereqs / cross-refs:** `sec_mdl-matrix-calculus-autodiff` (reverse-mode AD =
sequence of VJPs); §6.1.6 (the model being trained); `sec_mdl-integral_calculus`;
main-book autograd material.
:::

## Continuous Normalizing Flows
:label:`sec_mdl-continuous-normalizing-flows`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** If we push samples through a Neural ODE, how does their
*density* change? The discrete change-of-variables formula (from
`sec_mdl-random_variables`) sums a log-det-Jacobian over layers; in continuous
time it collapses into the **instantaneous change of variables** — the log-density
changes at a rate equal to *minus the trace (divergence) of the velocity's
Jacobian*. This turns an $O(d^3)$ determinant into an $O(d)$-estimable trace and
is the engine of likelihood-based continuous flows (Chen et al., 2018; FFJORD,
Grathwohl et al., 2019).
**Outline:** 1. recall discrete change of variables
$\log p(\mathbf{x}_{n+1})=\log p(\mathbf{x}_n)-\log\lvert\det J\rvert$ (callback
to `sec_mdl-random_variables`) · 2. take the continuous limit to get
$\frac{d}{dt}\log p_t(\mathbf{x}(t)) = -\nabla\!\cdot\mathbf{f} = -\operatorname{tr}\!\big(\partial\mathbf{f}/\partial\mathbf{x}\big)$
· 3. integrate alongside the state to get exact log-likelihood
$\log p_T(\mathbf{x}(T))$ from a base density · 4. why the *trace* (not the full
determinant) makes this scalable; Hutchinson's stochastic trace estimator
(FFJORD) as the $O(d)$ trick · 5. preview: this same continuity/trace identity
reappears as the *continuity equation* and probability-flow ODE in §6.3.
**Key results to state:** instantaneous change of variables
$\dfrac{d\log p_t(\mathbf{x}(t))}{dt} = -\operatorname{tr}\!\Big(\dfrac{\partial\mathbf{f}}{\partial\mathbf{x}}\Big)=-\nabla\!\cdot\mathbf{f}$;
$\log p_T(\mathbf{x}(T))=\log p_0(\mathbf{x}_0)-\int_0^T\operatorname{tr}\!\big(\partial\mathbf{f}/\partial\mathbf{x}\big)\,dt$;
Hutchinson estimator $\operatorname{tr}(M)=\mathbb{E}_{\boldsymbol\epsilon}[\boldsymbol\epsilon^\top M\boldsymbol\epsilon]$.
**Diagrams:** reuse `fig_mdl-vector-field-flow` with a density overlaid —
a probability blob carried and reshaped by the field, annotated with the
$-\nabla\!\cdot\mathbf{f}$ expansion/compression rate.
**Worked example(s):** for a linear field $\mathbf{f}=A\mathbf{x}$, show the
instantaneous rule reduces to $\frac{d}{dt}\log p=-\operatorname{tr}(A)$ and
recovers the discrete $-\log\lvert\det e^{At}\rvert$ over a finite step; a tiny 2-D
CNF tracking $\log p_t$ as a ring is flowed to a Gaussian.
**Exercises (draft):** (1) derive the instantaneous rule from the discrete
log-det as $h\to0$; (2) show it reduces to $-\log\lvert\det\rvert$ for a single
linear layer (matches `sec_mdl-random_variables`); (3) why is the trace cheaper
than the determinant?; (4) verify the Hutchinson estimator is unbiased.
**Prereqs / cross-refs:** `sec_mdl-random_variables` (change of variables for
densities), `sec_mdl-multivariable_calculus` (trace, divergence), §6.1.6 (Neural
ODE); **forward bridge** to §6.3.3 (continuity equation) and §6.3.4
(probability-flow ODE), and §6.4.8 (exact-likelihood sampling).
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Bullet recap to be written once the body lands.
**Outline (planned bullets):** an ODE is a velocity field, a solution follows the
arrows · Lipschitz fields give unique, invertible flows (Picard–Lindelöf) · for
linear systems $e^{At}$ and the eigenvalues of $A$ decide decay/growth/oscillation
(callback `sec_mdl-eigendecompositions`) · explicit solvers march with global
error $O(h^p)$ (Euler $p{=}1$, RK4 $p{=}4$); stiff systems force tiny explicit
steps or implicit methods · a ResNet block is an Euler step, and its depth limit
is a Neural ODE · the adjoint method is reverse-mode AD through the ODE · the
instantaneous change of variables turns the log-det into a trace integral — the
CNF density rule that §6.3 reuses.
**Cross-refs:** `sec_mdl-eigendecompositions`, `sec_mdl-matrix-calculus-autodiff`,
`sec_mdl-random_variables`; §6.2, §6.3, §6.4.
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Show explicit Euler on $\dot x=-\lambda x$ is stable iff $h<2/\lambda$, and
   identify the amplification factor.
2. Read the convergence orders (slopes $1$ and $4$) off a log–log error-vs-$h$
   plot for Euler and RK4 on a test ODE.
3. Show the instantaneous change of variables reduces to the discrete
   log-det-Jacobian for a single linear layer (callback
   `sec_mdl-random_variables`).
4. The non-Lipschitz field $\dot x=x^2$ blows up in finite time — compute the
   blow-up time and explain why uniqueness/existence is only local.
5. Derive the adjoint ODE and argue each step is a vector–Jacobian product
   (callback `sec_mdl-matrix-calculus-autodiff`).
**Cross-refs:** §6.1.2–§6.1.8.
:::

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:
