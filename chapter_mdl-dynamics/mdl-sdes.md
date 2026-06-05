# Stochastic Differential Equations
:label:`sec_mdl-sdes`

Diffusion models are built on a **forward noising SDE** that corrupts clean data
into pure Gaussian noise. To read DDPM (:cite:`ho2020denoising`) and score-based
models (:cite:`song2021score`) you need three things this section develops from
scratch: Brownian motion (the canonical source of continuous randomness), the
Itô calculus (why a "$\tfrac12 g^2\partial_{xx}$" correction term appears the
moment noise enters), and the Ornstein–Uhlenbeck process — the simple
mean-reverting SDE that the variance-preserving diffusion forward process
discretizes. The payoff is a precise, simulatable description of the
*information-destroying* half of every diffusion model; §6.3 then shows how to
reverse it.

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §6.2, the second section of the
Dynamics chapter (:numref:`chap_mdl-dynamics`). No prose, code, or figures have
been authored yet; every subsection below is a planning stub in the standard
format. The body framing for the section as a whole:

> *Diffusion models are built on a forward noising SDE that corrupts data into
> Gaussian noise; to read DDPM and score-based models you need Brownian motion,
> the Itô correction, and the OU process — the SDE that variance-preserving
> diffusion discretizes.*

**Prerequisites:** §6.1 (`sec_mdl-odes-solvers`: ODEs, vector fields, Euler,
linear stability); 4.1–4.2 (`sec_mdl-random_variables`, distributions:
Gaussians, expectation, variance, independence); 4.3 (CLT, independence);
2.1–2.2 (`sec_mdl-single_variable_calculus`, `sec_mdl-multivariable_calculus`:
Taylor, chain rule).
**Capstone payoff:** the Ornstein–Uhlenbeck process (§6.2.7) as the
variance-preserving forward noising process reused throughout §6.3 and §6.4.
**Forward bridge:** the drift+diffusion form $d\mathbf{X}=\mathbf{f}\,dt+g\,d\mathbf{W}$
and Euler–Maruyama (§6.2.6) feed the Fokker–Planck equation and reverse SDE of
§6.3, and the DDPM-as-discretized-VP-SDE story of §6.4.4.
:::

## Why Add Randomness
:label:`sec_mdl-why-randomness`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** §6.1 gave us a *deterministic* velocity field. Adding a noise
term turns the field into a stochastic one — and that buys us something we
genuinely need for generative modeling: a process that *destroys information
controllably*, smearing any data distribution into a known, simple noise
distribution, in a way we can later run in reverse on average.
**Outline:** 1. from $\dot{\mathbf{x}}=\mathbf{f}$ to a noisy law of motion ·
2. the goal: a forward process with a *known, distribution-independent* endpoint
(Gaussian) so we always know where we are starting from when we generate ·
3. why noise (not just a deterministic squish) — it forgets the data
*smoothly* and makes the reverse process tractable via the score (preview §6.3) ·
4. roadmap: define the noise (Brownian motion), learn its calculus (Itô), write
the SDE, simulate it (Euler–Maruyama), and study the workhorse example (OU).
**Key results to state:** target forward behavior
$p_0=p_{\text{data}}\;\longrightarrow\;p_T\approx\mathcal N(\mathbf{0},\sigma^2 I)$;
SDE template $d\mathbf{X}=\underbrace{\mathbf{f}(\mathbf{X},t)\,dt}_{\text{drift}}+\underbrace{g(t)\,d\mathbf{W}}_{\text{diffusion}}$ (stated here, built below).
**Diagrams:** none new (motivational); optionally a teaser strip "data → noise"
foreshadowing `fig_mdl-forward-reverse` in §6.3.
**Worked example(s):** none — motivation only.
**Exercises (draft):** (1) name two reasons a *known* noise endpoint is useful for
sampling; (2) contrast destroying information with a deterministic contraction vs.
with added noise (which is reversible-on-average?).
**Prereqs / cross-refs:** §6.1 (deterministic fields); forward to §6.2.5 (the SDE),
§6.3 (reversal needs the score).
:::

## The Wiener Process / Brownian Motion
:label:`sec_mdl-wiener-process`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Brownian motion $W_t$ is *the* elementary continuous random
process — the continuous-time limit of a random walk. Its defining property,
Gaussian independent increments whose variance grows like $t$, forces the
hallmark "$\sqrt{\Delta t}$" scaling of noise that makes stochastic calculus
different from ordinary calculus.
**Outline:** 1. definition: $W_0=0$; independent increments; $W_t-W_s\sim\mathcal N(0,t-s)$;
continuous sample paths · 2. derive it as the scaling limit of a $\pm\sqrt{\Delta t}$
random walk (CLT, callback to 4.3) · 3. the variance law $\operatorname{Var}(W_t)=t$
and the spreading $\sqrt t$ envelope · 4. the paradox: sample paths are continuous
but *nowhere differentiable* — increments scale like $\sqrt{\Delta t}$, so
$\Delta W/\Delta t\to\infty$ (this is *why* we cannot write $dW/dt$ and must build
a new integral) · 5. multivariate $\mathbf{W}_t$ with independent coordinates.
**Key results to state:** $W_0=0$; $W_t-W_s\sim\mathcal N(0,t-s)$ independent of
the past; $\mathbb E[W_t]=0$, $\operatorname{Var}(W_t)=t$;
$\mathbb E[(W_t-W_s)^2]=t-s$; increment $\Delta W\sim\sqrt{\Delta t}\,\xi$,
$\xi\sim\mathcal N(0,1)$; paths continuous but nowhere differentiable.
**Diagrams:** `fig_mdl-brownian-paths` — many simulated 1-D Brownian paths from
$0$, overlaid with the $\pm\sqrt t$ standard-deviation envelope, showing variance
growing linearly in $t$.
**Worked example(s):** simulate Brownian motion as a cumulative sum of
$\sqrt{\Delta t}\,\xi$ increments; empirically check $\operatorname{Var}(W_t)=t$
across many paths; show the path "roughens" (does not smooth out) as $\Delta t\to0$.
**Exercises (draft):** (1) show the random-walk limit gives variance $t$;
(2) compute $\operatorname{Cov}(W_s,W_t)=\min(s,t)$; (3) explain why
$\Delta W/\Delta t$ has no finite limit; (4) simulate and verify the $\sqrt t$
envelope captures ≈68% of paths.
**Prereqs / cross-refs:** `sec_mdl-random_variables` / distributions (Gaussian),
4.3 (CLT); forward to §6.2.3 (quadratic variation), §6.2.4 (Itô integral).
:::

## Why Ordinary Calculus Fails: Quadratic Variation
:label:`sec_mdl-quadratic-variation`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Here is the single fact from which *every* extra term in
stochastic calculus springs: for Brownian motion, $(dW)^2$ does **not** vanish to
higher order the way $(dt)^2$ does — it equals $dt$. Squared Brownian increments
accumulate deterministically, so a second-order Taylor term survives where in
ordinary calculus it would be negligible.
**Outline:** 1. recall in ordinary calculus $(dt)^2\to0$ faster than $dt$, so
Taylor stops at first order · 2. for Brownian motion the *quadratic variation*
$\sum (\Delta W_i)^2\to t$ (a limit in mean square), not $0$ · 3. the working
rules of the Itô "multiplication table": $(dW)^2=dt$, $dW\,dt=0$, $(dt)^2=0$ ·
4. consequence: the second-order Taylor term $\tfrac12 f''(dX)^2$ contributes a
*finite* $\tfrac12 f''g^2\,dt$ — this is the Itô correction, derived next.
**Key results to state:** $\displaystyle\sum_{i}(\Delta W_i)^2 \xrightarrow{\text{m.s.}} t$
(quadratic variation of Brownian motion is $t$); Itô rules
$(dW)^2=dt,\ dW\,dt=0,\ (dt)^2=0$.
**Diagrams:** `fig_mdl-quadratic-variation` — partial sums $\sum(\Delta W_i)^2$ vs.
the line $y=t$ as the partition is refined, showing convergence to $t$ (contrast
with $\sum(\Delta t_i)^2\to0$ for a smooth path).
**Worked example(s):** numerically accumulate $\sum(\Delta W_i)^2$ over a fine grid
and watch it converge to $t$ with vanishing variance; contrast with the smooth
function $t$ whose $\sum(\Delta t_i)^2\to0$.
**Exercises (draft):** (1) show $\mathbb E[\sum(\Delta W_i)^2]=t$ and
$\operatorname{Var}\to0$ as the mesh shrinks; (2) verify $(dW)^2=dt$ heuristically
from $\Delta W\sim\sqrt{\Delta t}\,\xi$; (3) explain why the same sum is $0$ for any
differentiable path.
**Prereqs / cross-refs:** §6.2.2 (Brownian increments), 2.1 (Taylor); forward to
§6.2.4 (Itô's lemma uses these rules directly).
:::

## The Itô Integral and Itô's Lemma
:label:`sec_mdl-ito-lemma`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** To integrate against a nowhere-differentiable $W_t$ we need a
new integral — the **Itô integral**, which always evaluates the integrand at the
*left* endpoint (non-anticipating, "no peeking at the future"). The chain rule for
this calculus is **Itô's lemma**: it is the ordinary chain rule plus the
$\tfrac12 g^2\partial_{xx}$ correction forced by $(dW)^2=dt$.
**Outline:** 1. the Itô integral $\int_0^t G_s\,dW_s$ as a left-endpoint Riemann
sum limit; the martingale / zero-mean property
$\mathbb E[\int G\,dW]=0$ and the Itô isometry · 2. why left-endpoint matters
(contrast Stratonovich briefly) · 3. **Itô's lemma** for $Y=\phi(X_t,t)$ where
$dX=f\,dt+g\,dW$: the extra $\tfrac12 g^2\phi_{xx}\,dt$ term · 4. the canonical
checks: $d(W^2)=2W\,dW+dt$ (the $+dt$ is the correction), and the geometric
Brownian-motion solution $d(\log X)$ · 5. emphasize this single correction term is
the seed of the Fokker–Planck Laplacian (§6.3.2).
**Key results to state:** Itô integral $\int_0^t G_s\,dW_s$, $\mathbb E[\cdot]=0$,
isometry $\mathbb E[(\int G\,dW)^2]=\mathbb E[\int G^2\,ds]$;
**Itô's lemma** for $dX=f\,dt+g\,dW$:
$$d\phi(X,t)=\Big(\phi_t+f\,\phi_x+\tfrac12 g^2\,\phi_{xx}\Big)dt+g\,\phi_x\,dW;$$
$d(W^2)=2W\,dW+dt$ (vs. the naïve $2W\,dW$).
**Diagrams:** `fig_mdl-ito-correction` — simulated $W_t^2$ vs. the naïve integral
$\int 2W\,dW$ and the corrected $\int 2W\,dW + t$, showing the $+t$ gap is exactly
the accumulated Itô correction.
**Worked example(s):** apply Itô's lemma to $\phi(X)=X^2$ on $dX=g\,dW$ to get the
$g^2\,dt$ drift; simulate $W_t^2$ and confirm it tracks $\int 2W\,dW + t$, not
$\int 2W\,dW$. Framework-agnostic.
**Exercises (draft):** (1) apply Itô's lemma to $\phi(X)=X^2$ for $dX=f\,dt+g\,dW$;
(2) derive $d(\log X)$ for geometric Brownian motion and read off the $-\tfrac12\sigma^2$
correction; (3) show $\mathbb E[\int_0^t W_s\,dW_s]=0$ and evaluate the integral as
$\tfrac12(W_t^2-t)$; (4) state where the correction term comes from in the Taylor
expansion (callback §6.2.3).
**Prereqs / cross-refs:** §6.2.3 ($(dW)^2=dt$), 2.1–2.2 (Taylor, chain rule);
forward to §6.3.2 (the $\tfrac12 g^2\partial_{xx}$ becomes the Fokker–Planck
diffusion/Laplacian term).
:::

## Stochastic Differential Equations
:label:`sec_mdl-sde-definition`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Now we can write the object the whole chapter is about:
$d\mathbf{X}=\mathbf{f}(\mathbf{X},t)\,dt+g(t)\,d\mathbf{W}$. The **drift**
$\mathbf{f}$ steers the average motion (the deterministic ODE of §6.1 is the
$g=0$ case); the **diffusion** $g$ injects Brownian jitter. Every diffusion model
is a particular choice of these two functions.
**Outline:** 1. the general (Itô) SDE $d\mathbf{X}=\mathbf{f}(\mathbf{X},t)\,dt+G(\mathbf{X},t)\,d\mathbf{W}$;
the scalar-diffusion special case $g(t)\,d\mathbf{W}$ used in diffusion models ·
2. read it as "deterministic flow + noise kick"; recover §6.1's ODE when $g=0$ ·
3. existence/uniqueness of strong solutions (Lipschitz drift+diffusion, the
stochastic analogue of §6.1.2) · 4. a *solution is a distribution over paths*, not
a single curve — this is the conceptual leap that §6.3 turns into densities ·
5. name the two diffusion-model families to come: variance-exploding (VE,
$\mathbf{f}=0$, growing $g$) and variance-preserving (VP, the OU-type drift),
foreshadowing §6.4.
**Key results to state:**
$d\mathbf{X}=\mathbf{f}(\mathbf{X},t)\,dt+g(t)\,d\mathbf{W}$;
drift = $\mathbb E[d\mathbf{X}]/dt$, diffusion sets
$\operatorname{Cov}(d\mathbf{X})=g^2 I\,dt$; $g\equiv0$ recovers the §6.1 ODE.
**Diagrams:** `fig_mdl-sde-trajectory-cloud` — one SDE simulated from a fixed start
many times: a fan/cloud of jittery paths whose mean follows the drift and whose
spread is set by the diffusion, with the mean curve highlighted.
:numref:`fig_mdl-dyn-sde-paths` shows this for the Ornstein–Uhlenbeck process.

![Many sample paths of the SDE $dX=-\theta X\,dt+\sigma\,dW$ simulated by Euler–Maruyama from a single start $X_0$. The thin blue curves are the jittery individual paths; their mean follows the deterministic drift $X_0 e^{-\theta t}$ (green), and the spreading ensemble is captured by the analytic $\pm 2$-standard-deviation time-marginal envelope (orange), whose width $\sqrt{(\sigma^2/2\theta)(1-e^{-2\theta t})}$ saturates as the process reaches its stationary distribution.](../img/mdl-dyn-sde-paths.svg)
:label:`fig_mdl-dyn-sde-paths`
**Worked example(s):** simulate the same SDE from one initial condition many times
and visualize the path cloud; vary $g$ to show drift-dominated vs.
diffusion-dominated regimes. (Simulation method is §6.2.6.)
**Exercises (draft):** (1) identify drift and diffusion in three example SDEs;
(2) show setting $g=0$ gives back an ODE; (3) for $d X=g\,dW$ (pure diffusion)
compute $\mathbb E[X_t]$ and $\operatorname{Var}(X_t)$.
**Prereqs / cross-refs:** §6.1 (the $g=0$ deterministic limit), §6.2.2–§6.2.4
(Brownian motion, Itô); forward to §6.2.7 (OU as a concrete SDE), §6.3 (paths →
densities), §6.4 (VE/VP families).
:::

## Euler–Maruyama Discretization
:label:`sec_mdl-euler-maruyama`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** To *simulate* an SDE we discretize it exactly as we discretized
ODEs in §6.1.4 — Euler's step plus one extra ingredient: a Gaussian noise kick of
size $\sqrt{\Delta t}$ (the $\sqrt{\Delta t}$ scaling from §6.2.2). This is the
single forward step of a diffusion process, and (in disguise) one step of DDPM's
forward chain.
**Outline:** 1. the Euler–Maruyama update
$\mathbf{X}_{n+1}=\mathbf{X}_n+\mathbf{f}(\mathbf{X}_n,t_n)\Delta t+g(t_n)\sqrt{\Delta t}\,\boldsymbol\xi_n$,
$\boldsymbol\xi_n\sim\mathcal N(\mathbf 0,I)$ · 2. why noise scales as
$\sqrt{\Delta t}$, not $\Delta t$ (callback §6.2.2) · 3. it reduces to forward
Euler (§6.1.4) when $g=0$ · 4. convergence: *strong* order $\tfrac12$ (pathwise),
*weak* order $1$ (distributional) — the gap is a feature of stochastic
integration, and weak order is what matters for matching *marginals* (the §6.3/§6.4
quantity of interest) · 5. this update *is* one step of the diffusion forward
process; foreshadow it being run in reverse in §6.3/§6.4.
**Key results to state:**
$\mathbf{X}_{n+1}=\mathbf{X}_n+\mathbf{f}(\mathbf{X}_n,t_n)\,\Delta t+g(t_n)\sqrt{\Delta t}\,\boldsymbol\xi_n$,
$\boldsymbol\xi_n\sim\mathcal N(\mathbf 0,I)$; strong order $\tfrac12$, weak order
$1$; $g=0\Rightarrow$ forward Euler.
**Diagrams:** none new (reuse `fig_mdl-sde-trajectory-cloud`, annotating one
$\Delta t$ step as "drift increment + $\sqrt{\Delta t}$ noise kick").
**Worked example(s):** implement Euler–Maruyama for the OU process; check that the
empirical marginal at time $t$ matches the analytic Gaussian from §6.2.7; shrink
$\Delta t$ and watch the marginal converge (weak order 1).
**Exercises (draft):** (1) show Euler–Maruyama $\to$ Euler as $g\to0$; (2) why does
the noise term scale as $\sqrt{\Delta t}$ rather than $\Delta t$?; (3) empirically
estimate the weak convergence order on OU by tracking the marginal mean/variance;
(4) what goes wrong if you (wrongly) use a $\Delta t$ noise scaling?
**Prereqs / cross-refs:** §6.1.4 (Euler), §6.2.2 ($\sqrt{\Delta t}$ scaling),
§6.2.5 (the SDE); forward to §6.4.4 (DDPM as discretized VP-SDE), §6.4.8 (SDE vs.
ODE sampling).
:::

## Worked Process: Ornstein–Uhlenbeck
:label:`sec_mdl-ornstein-uhlenbeck`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The one SDE to know by heart: the Ornstein–Uhlenbeck process
$dX=-\theta X\,dt+\sigma\,dW$. Its drift *pulls toward the origin* (mean reversion
— exactly the stable linear ODE of §6.1.3 with noise added), it has a *closed-form
Gaussian* transition density at every time, and it relaxes to a *stationary
Gaussian* $\mathcal N(0,\sigma^2/2\theta)$. This is the variance-preserving forward
noising process at the heart of DDPM and score-based diffusion.
**Outline:** 1. the SDE $dX=-\theta X\,dt+\sigma\,dW$; mean reversion as §6.1.3's
$\dot x=-\theta x$ skeleton plus jitter · 2. solve it (Itô on $e^{\theta t}X$):
closed-form mean and variance, Gaussian at all times · 3. the stationary
distribution $\mathcal N(0,\sigma^2/2\theta)$ — variance stops growing because
drift exactly balances diffusion (this is why it is "variance-preserving") ·
4. the conditional/transition density $p(x_t\mid x_0)$ in closed form — the
*Gaussian noising kernel* that makes denoising score matching cheap (§6.4.2) ·
5. map to diffusion: a (possibly time-scaled) OU process noises data to
$\mathcal N(0,\cdot)$, the known endpoint we sample from.
**Key results to state:**
$dX=-\theta X\,dt+\sigma\,dW$;
$\mathbb E[X_t\mid X_0]=X_0 e^{-\theta t}$;
$\operatorname{Var}(X_t\mid X_0)=\dfrac{\sigma^2}{2\theta}\big(1-e^{-2\theta t}\big)$;
transition $X_t\mid X_0\sim\mathcal N\!\big(X_0 e^{-\theta t},\,\tfrac{\sigma^2}{2\theta}(1-e^{-2\theta t})\big)$;
stationary $X_\infty\sim\mathcal N\!\big(0,\tfrac{\sigma^2}{2\theta}\big)$.
**Diagrams:** `fig_mdl-ou-mean-reversion` — OU paths from several start points all
relaxing toward $0$, with the shrinking-then-saturating variance band and the
stationary Gaussian drawn as a sideways density on the right edge.
**Worked example(s):** simulate OU by Euler–Maruyama from several $X_0$ and overlay
the analytic mean $X_0e^{-\theta t}$ and variance band; verify the long-time
histogram matches the stationary $\mathcal N(0,\sigma^2/2\theta)$.
**Exercises (draft):** (1) solve OU via Itô on $e^{\theta t}X$ and derive the mean
and variance; (2) take $t\to\infty$ to get the stationary variance and explain
"variance preserving"; (3) show OU $\to$ §6.1.3's deterministic decay as
$\sigma\to0$; (4) compute $\operatorname{Cov}(X_s,X_t)$ in the stationary regime.
**Prereqs / cross-refs:** §6.1.3 (linear stability / mean reversion), §6.2.4–§6.2.6
(Itô, Euler–Maruyama), `sec_mdl-random_variables` (Gaussian transition); forward
to §6.3.2 (OU's Gaussian satisfies Fokker–Planck), §6.4.2 (Gaussian kernel =
cheap denoising target), §6.4.4 (VP-SDE / DDPM).
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Bullet recap to be written once the body lands.
**Outline (planned bullets):** randomness gives a forward process with a known
Gaussian endpoint · Brownian motion has Gaussian independent increments,
$\operatorname{Var}(W_t)=t$, continuous but nowhere differentiable, increments
$\sim\sqrt{\Delta t}$ · quadratic variation $(dW)^2=dt$ is the source of every
extra term · the Itô integral is left-endpoint; Itô's lemma = chain rule +
$\tfrac12 g^2\partial_{xx}$ · an SDE is drift + diffusion, $g=0$ recovers the ODE ·
Euler–Maruyama = Euler + $\sqrt{\Delta t}$ noise (strong order $\tfrac12$, weak
order $1$) · the OU process mean-reverts, is Gaussian at all times, and is the
variance-preserving forward noising process behind diffusion.
**Cross-refs:** §6.1 (ODEs), §6.3 (Fokker–Planck / reversal), §6.4 (diffusion).
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Apply Itô's lemma to $f(X)=X^2$ for $dX=g\,dW$ and recover the $g^2\,dt$ drift
   that ordinary calculus misses.
2. For the OU process, derive $\mathbb E[X_t]$, $\operatorname{Var}(X_t)$, and the
   stationary distribution; explain why it is "variance preserving."
3. Show Euler–Maruyama reduces to forward Euler as the diffusion $g\to0$.
4. Compare the strong (pathwise, order $\tfrac12$) and weak (distributional,
   order $1$) convergence of Euler–Maruyama on OU empirically, and explain why weak
   order is the relevant one for matching marginals.
**Cross-refs:** §6.2.2–§6.2.7.
:::

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/)
:end_tab:
