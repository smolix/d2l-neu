# The Fokker–Planck Equation and Probability Flow
:label:`sec_mdl-fokker-planck-probability-flow`

This is the mathematical heart that makes diffusion work. A single SDE path
(§6.2) is a wiggly, unpredictable curve; but the *ensemble* of all paths is a
**density** $p_t(\mathbf{x})$ that flows and spreads according to a deterministic
PDE — the Fokker–Planck equation. From it follow two facts that the entire
generative-modeling edifice rests on: every noising SDE has a *deterministic twin*
ODE (the **probability-flow ODE**) sharing the same time-marginals $p_t$, and
running the process *backward* — turning noise into data — requires exactly one
unknown quantity, the **score** $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$. The
slogan of the chapter, "we just need the score," is proved here; §6.4 then learns
it (:cite:`song2021score`).

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §6.3, the third section of the Dynamics
chapter (:numref:`chap_mdl-dynamics`). No prose, code, or figures have been
authored yet; every subsection below is a planning stub in the standard format.
The body framing for the section as a whole:

> *An SDE has a deterministic twin ODE sharing the same time-marginals
> $p_t(\mathbf{x})$, and running the noising process backward requires exactly the
> score $\nabla\log p_t(\mathbf{x})$ — which §6.4 learns to approximate.*

**Prerequisites:** §6.2 (`sec_mdl-sdes`: SDEs, Itô, OU); §6.1 (`sec_mdl-odes-solvers`:
ODEs, the CNF continuity/trace identity §6.1.8); 2.1–2.2
(`sec_mdl-multivariable_calculus`: gradient, divergence, Laplacian); 4.1
(`sec_mdl-random_variables`: densities); 5.2 (`sec_mdl-divergences-distances`:
score / Fisher divergence — light, via `sec_mdl-fisher-divergence`).
**Capstone payoff:** the probability-flow ODE velocity
$\mathbf{v}_t=\mathbf{f}-\tfrac12 g^2\nabla\log p_t$ (§6.3.4) and Anderson's reverse
SDE (§6.3.6) — the deterministic and stochastic samplers learned in §6.4.
**Forward bridge:** the score function (§6.3.5) is the single unknown that §6.4
estimates by denoising score matching; the reverse SDE/PF-ODE pair is the §6.4.8
sampler dichotomy.
:::

## From Paths to Densities
:label:`sec_mdl-paths-to-densities`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Switch viewpoints from one trajectory to the whole crowd.
A single SDE path is unpredictable, but if we release a *cloud* of points
distributed as $p_0$ and let them all flow under the same SDE, the cloud's
density $p_t(\mathbf{x})$ evolves in a perfectly deterministic way — drift pushes
the cloud, diffusion smears it.
**Outline:** 1. the time-marginal $p_t(\mathbf{x})$ = the law of $\mathbf{X}_t$
when $\mathbf{X}_0\sim p_0$ · 2. Lagrangian (follow a particle) vs. Eulerian (watch
the density at a fixed point) descriptions · 3. the qualitative behavior we will
make precise: drift *transports* probability mass, diffusion *spreads/smooths* it
(a heat-equation intuition) · 4. promise: a single PDE (Fokker–Planck) governs
$p_t$ exactly.
**Key results to state:** $p_t(\mathbf{x})$ is the marginal density of the SDE
solution; for OU it is the closed-form Gaussian from §6.2.7 (a concrete $p_t$ to
track throughout).
**Diagrams:** `fig_mdl-density-spreading` — snapshots of a bimodal $p_0$ evolving
under a noising SDE, the two modes drifting together and smearing into a single
Gaussian $p_T$ (the "destroy information" picture, density view).
**Worked example(s):** simulate many OU paths and build histograms of $p_t$ at a
few times $t$; overlay the analytic Gaussian $p_t$ from §6.2.7 to confirm the
marginal view matches the path view.
**Exercises (draft):** (1) for OU, write $p_t$ from the §6.2.7 transition density
when $p_0=\delta_{x_0}$; (2) describe in words how drift vs. diffusion each act on
the density; (3) why is the *marginal* deterministic even though each path is
random?
**Prereqs / cross-refs:** §6.2.7 (OU marginals), `sec_mdl-random_variables`
(densities); forward to §6.3.2 (the PDE for $p_t$).
:::

## The Fokker–Planck (Kolmogorov Forward) Equation
:label:`sec_mdl-fokker-planck`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The exact law of motion for the density. The Fokker–Planck (a.k.a.
Kolmogorov forward) equation says $p_t$ changes by two effects: a *transport* term
from the drift (probability flows along $\mathbf{f}$) and a *diffusion* term from
the noise (a Laplacian that smooths, i.e. heat flow). The diffusion term is the
$(dW)^2=dt$ Itô correction of §6.2 made macroscopic.
**Outline:** 1. derive it from Itô's lemma applied to a test function and an
integration by parts (sketch; full derivation deferred but the mechanism shown) ·
2. read the two terms: $-\nabla\!\cdot(\mathbf{f}p)$ = advection/transport,
$\tfrac12\nabla\!\cdot(g^2\nabla p)$ = diffusion/heat · 3. the steady state
$\partial_t p=0$ and how it recovers the OU stationary Gaussian (§6.2.7) ·
4. pure-diffusion special case ($\mathbf{f}=0$) is literally the heat equation ·
5. note: the diffusion term traces straight back to the Itô $\tfrac12 g^2\partial_{xx}$
of §6.2.4.
**Key results to state:** Fokker–Planck
$$\partial_t p_t(\mathbf{x}) = -\nabla\!\cdot\!\big(\mathbf{f}(\mathbf{x},t)\,p_t(\mathbf{x})\big) + \tfrac12\,\nabla\!\cdot\!\big(g(t)^2\nabla p_t(\mathbf{x})\big);$$
scalar form $\partial_t p=-\partial_x(f p)+\tfrac12 g^2\partial_{xx}p$;
$\mathbf{f}=0$ gives the heat equation $\partial_t p=\tfrac12 g^2\Delta p$;
OU steady state $=\mathcal N(0,\sigma^2/2\theta)$.
**Diagrams:** reuse `fig_mdl-density-spreading` annotated with the two operators —
an arrow for the drift/transport term and a blur for the diffusion/Laplacian term.
**Worked example(s):** verify by direct differentiation that the OU closed-form
Gaussian $p_t$ (from §6.2.7) satisfies the Fokker–Planck PDE; show the heat
equation spreads a Gaussian with linearly growing variance.
**Exercises (draft):** (1) plug the OU Gaussian into Fokker–Planck and confirm
both sides match; (2) show the OU stationary Gaussian makes $\partial_t p=0$;
(3) identify which term vanishes for pure diffusion and recover the heat equation;
(4) trace the $\tfrac12 g^2$ factor back to Itô (§6.2.4).
**Prereqs / cross-refs:** §6.2.4 (Itô correction), §6.2.7 (OU),
`sec_mdl-multivariable_calculus` (divergence, Laplacian); forward to §6.3.3
(continuity-equation form), §6.3.4 (probability-flow ODE).
:::

## The Continuity Equation
:label:`sec_mdl-continuity-equation`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Probability mass is conserved — it is never created or destroyed,
only moved. That conservation law is the *continuity equation*
$\partial_t p+\nabla\!\cdot(p\,\mathbf{v}_t)=0$, the same equation that governs an
incompressible fluid. Crucially, *any* density evolution — including the
diffusion above — can be written this way for the right velocity field
$\mathbf{v}_t$, which is precisely the trick that converts an SDE into an ODE.
**Outline:** 1. derive the continuity equation from "rate of change of mass in a
region = net flux across its boundary" (divergence theorem) · 2. it is the same
identity as the CNF instantaneous change of variables from §6.1.8 (the
$-\nabla\!\cdot\mathbf{f}$ trace term) · 3. the key rewrite: rearrange
Fokker–Planck into continuity form by absorbing the diffusion term into an
*effective velocity* — set up §6.3.4 · 4. why this matters: continuity form means
"there is a deterministic flow with these marginals."
**Key results to state:** continuity equation
$\partial_t p_t + \nabla\!\cdot(p_t\,\mathbf{v}_t)=0$;
equivalence to the §6.1.8 CNF rule $\frac{d}{dt}\log p_t=-\nabla\!\cdot\mathbf{v}_t$
along trajectories; the diffusion term
$\tfrac12\nabla\!\cdot(g^2\nabla p)=\nabla\!\cdot\!\big(p\cdot(-\tfrac12 g^2\nabla\log p)\big)$
(the algebraic identity that powers §6.3.4).
**Diagrams:** `fig_mdl-continuity-flux` — a fixed control region with probability
flux arrows crossing its boundary, illustrating "change inside = net flux"; a
side note linking to the CNF trace picture of §6.1.8.
:numref:`fig_mdl-dyn-fokker-planck-flux` draws this balance in one dimension.

![The continuity equation as a flux balance. A density $p(x)$ (blue) is carried by a drift $f$, giving the probability current $j(x)=f\,p(x)$. Over the fixed region from $a$ to $b$ the enclosed mass (orange) changes only through the current crossing its two boundaries: $\partial_t\int_a^b p\,dx = j(a)-j(b)$, the net inward flux. Probability is conserved, never created or destroyed inside the region.](../img/mdl-dyn-fokker-planck-flux.svg)
:label:`fig_mdl-dyn-fokker-planck-flux`
**Worked example(s):** verify the diffusion-as-transport identity
$\tfrac12\nabla\!\cdot(g^2\nabla p)=-\nabla\!\cdot(p\cdot\tfrac12 g^2\nabla\log p)$
using $\nabla p=p\,\nabla\log p$; check conservation $\int p_t\,d\mathbf{x}=1$ for
all $t$ on the OU example.
**Exercises (draft):** (1) derive the continuity equation from the divergence
theorem; (2) prove the diffusion-as-transport identity using
$\nabla p=p\nabla\log p$; (3) connect it to the §6.1.8 CNF log-density rule;
(4) show total mass is conserved under the continuity equation.
**Prereqs / cross-refs:** §6.1.8 (CNF continuity/trace), §6.3.2 (Fokker–Planck),
`sec_mdl-multivariable_calculus` (divergence theorem); forward to §6.3.4 (the
probability-flow velocity).
:::

## The Probability-Flow ODE
:label:`sec_mdl-probability-flow-ode`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The first of the two payoffs. By writing Fokker–Planck in
continuity form (§6.3.3), the *random* SDE acquires a *deterministic* twin — an
ODE whose single trajectory, when started from $p_0$, reproduces the SDE's
time-marginals $p_t$ **exactly**. Its velocity is the drift minus a half-$g^2$
times the score. This is what lets diffusion models sample with a plain ODE solver
(few steps, deterministic, exact likelihood).
**Outline:** 1. plug the §6.3.3 identity into Fokker–Planck to read off the
*probability-flow velocity* $\mathbf{v}_t=\mathbf{f}-\tfrac12 g^2\nabla\log p_t$ ·
2. state the theorem: the ODE $\dot{\mathbf{x}}=\mathbf{v}_t(\mathbf{x})$ has the
*same marginals* $p_t$ as the SDE (:cite:`song2021score`) — though individual
ODE trajectories differ from individual SDE paths · 3. consequences: deterministic
sampling, and *exact* likelihood via the §6.1.8 CNF trace integral · 4. the
single missing ingredient is $\nabla\log p_t$ — set up §6.3.5.
**Key results to state:** probability-flow velocity
$$\mathbf{v}_t(\mathbf{x})=\mathbf{f}(\mathbf{x},t)-\tfrac12\,g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x});$$
the PF-ODE $\dot{\mathbf{x}}=\mathbf{v}_t(\mathbf{x})$ shares all marginals $p_t$
with the SDE (Song et al., 2021); likelihood via §6.1.8:
$\log p_0(\mathbf{x}_0)=\log p_T(\mathbf{x}_T)+\int_0^T\nabla\!\cdot\mathbf{v}_t\,dt$.
**Diagrams:** `fig_mdl-sde-vs-pfode-marginals` — top: the SDE path cloud (jittery);
bottom: the PF-ODE's smooth deterministic trajectories; right: identical marginal
histograms at several $t$, showing the two processes share $p_t$ despite different
paths.
**Worked example(s):** for OU, plug the analytic score (§6.3.5) into the PF-ODE,
integrate deterministically, and confirm the resulting marginals match the
forward-SDE histograms; show ODE trajectories are smooth where SDE paths are
rough.
**Exercises (draft):** (1) derive $\mathbf{v}_t=\mathbf{f}-\tfrac12 g^2\nabla\log p_t$
from the continuity rewrite; (2) explain why marginals match but individual paths
do not; (3) write the PF-ODE for OU using its analytic score; (4) why does the
factor $\tfrac12$ appear (callback §6.3.3)?
**Prereqs / cross-refs:** §6.3.2–§6.3.3 (Fokker–Planck → continuity), §6.1.8 (CNF
likelihood), `sec_mdl-odes-solvers` (solving it); forward to §6.4.8 (PF-ODE
sampling).
:::

## The Score Function
:label:`sec_mdl-score-function`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The central reduction of the whole subject. Look at the
probability-flow velocity (and, next, the reverse SDE): the *only* term we do not
already know is $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$, the **score** — the
gradient of the log-density, a vector field pointing toward higher data density.
Everything else (the drift $\mathbf{f}$, the schedule $g$) we *chose* when we
designed the forward process. So "we just need the score." (Ties to the Fisher
divergence of `sec_mdl-fisher-divergence`.)
**Outline:** 1. define the (Stein) score
$\mathbf{s}_t(\mathbf{x})=\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ — note it is the
gradient w.r.t. *the input*, not parameters · 2. geometric reading: arrows up the
density landscape, zero at modes, large where density changes fast · 3. the
crucial property (callback `sec_mdl-fisher-divergence`): the score is *independent
of the normalizing constant* $Z$, since $\nabla\log(\tilde p/Z)=\nabla\log\tilde p$
— this is *why* it is learnable for unnormalized models · 4. closed form for a
Gaussian: $\nabla\log\mathcal N(\mathbf{x};\boldsymbol\mu,\sigma^2 I)=-(\mathbf{x}-\boldsymbol\mu)/\sigma^2$
(the analytic score we use to validate everything) · 5. the slogan: every term in
the PF-ODE and reverse SDE except the score is known by design.
**Key results to state:** $\mathbf{s}_t(\mathbf{x})=\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$;
$\nabla_{\mathbf{x}}\log p$ is invariant to multiplying $p$ by any constant (drops
$Z$); Gaussian score
$\nabla\log\mathcal N(\mathbf{x};\boldsymbol\mu,\sigma^2 I)=-(\mathbf{x}-\boldsymbol\mu)/\sigma^2$.
**Diagrams:** reuse `fig_mdl-score-field` (from `sec_mdl-fisher-divergence`) — the
score vector field of a 2-D density, arrows climbing toward the modes; annotate
"the one unknown in the PF-ODE / reverse SDE." :numref:`fig_mdl-dyn-score-field`
shows the score of a two-mode density.

![The score $\mathbf{s}(\mathbf{x})=\nabla\log p(\mathbf{x})$ of a two-component Gaussian mixture, drawn as a vector field over the density contours. The arrows climb the density landscape, pointing toward the nearest mode, and vanish at the two modes (orange) where the density is locally flat. This vector field is the only quantity in the probability-flow ODE and the reverse SDE that is not known by design.](../img/mdl-dyn-score-field.svg)
:label:`fig_mdl-dyn-score-field`
**Worked example(s):** compute the score of a 1-D and a 2-D Gaussian by hand and
plot the field; show adding a constant to $\log\tilde p$ leaves the score
unchanged; compute the score of a Gaussian mixture and observe it points toward
the nearest mode.
**Exercises (draft):** (1) derive the Gaussian score; (2) show the score ignores
the normalizer $Z$; (3) compute and sketch the score field of a 2-component 1-D
mixture; (4) where is the score zero, and what does that mean?
**Prereqs / cross-refs:** `sec_mdl-fisher-divergence` (score / Fisher divergence,
normalizer-free), §6.3.4 (score is the PF-ODE's unknown),
`sec_mdl-multivariable_calculus` (gradient); forward to §6.4.1–§6.4.2 (learning
the score).
:::

## Time-Reversal of Diffusions (Anderson)
:label:`sec_mdl-time-reversal`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The second payoff, and the one that makes *generation* possible.
Anderson's theorem says the time-reversal of a diffusion is *itself* a diffusion —
a **reverse SDE** that turns noise back into data — and its drift is the forward
drift corrected by the score. So the same score that we needed for the
deterministic PF-ODE also drives the stochastic reverse process. Forward = noising
(known, easy); reverse = denoising (needs the learned score).
**Outline:** 1. state Anderson's reverse-time SDE
$d\mathbf{X}=[\mathbf{f}-g^2\nabla\log p_t]\,dt+g\,d\bar{\mathbf{W}}$ run from
$t=T$ down to $0$, $\bar{\mathbf{W}}$ a reverse-time Brownian motion · 2. contrast
the *reverse SDE* (stochastic, $-g^2\nabla\log p_t$ drift correction) with the
*PF-ODE* (deterministic, $-\tfrac12 g^2\nabla\log p_t$) — they share the same
marginals $p_t$ but the SDE injects fresh noise while the ODE does not (explain the
factor-of-2 difference, callback §6.3.4) · 3. the generative recipe in one line:
start from the known $p_T\approx\mathcal N(\mathbf{0},\cdot)$, integrate the reverse
SDE (or PF-ODE) using the score, arrive at $p_0=p_{\text{data}}$ · 4. emphasize the
score is the *only* learned object; §6.4 estimates it.
**Key results to state:** Anderson reverse SDE
$$d\mathbf{X}=\big[\mathbf{f}(\mathbf{X},t)-g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{X})\big]\,dt+g(t)\,d\bar{\mathbf{W}};$$
reverse drift correction $-g^2\nabla\log p_t$ (SDE) vs. $-\tfrac12 g^2\nabla\log p_t$
(PF-ODE); both share marginals $p_t$ (Anderson, 1982; Song et al., 2021).
**Diagrams:** `fig_mdl-forward-reverse` — a single panel with the forward
(data→noise) arrow on top and the reverse (noise→data) arrow on the bottom, the
score $\nabla\log p_t$ drawn as the field that "flips the arrow"; the only learned
piece highlighted. :numref:`fig_mdl-dyn-forward-reverse` shows the two processes
as a row of density slices.

![The forward and reverse diffusion processes on a density. Top row (left to right): the forward noising SDE turns a structured bimodal data density $p_0$ into a single Gaussian $p_T$ across a few time slices, each panel an exact variance-preserving (OU) marginal of the mixture. Bottom row (right to left): the reverse process, driven by the score $\nabla\log p_t$, runs the same marginals backward to recover the data density from noise. Forward is fixed by design; reverse needs only the learned score.](../img/mdl-dyn-forward-reverse.svg)
:label:`fig_mdl-dyn-forward-reverse`
**Worked example(s):** for 1-D OU, plug the analytic score into Anderson's reverse
SDE, integrate from the stationary Gaussian back to $t=0$, and recover the original
$p_0$; do the same with the PF-ODE and show the marginals agree.
**Exercises (draft):** (1) write the OU reverse SDE with its analytic score;
(2) explain the $g^2$ vs. $\tfrac12 g^2$ drift-correction difference between reverse
SDE and PF-ODE (callback §6.3.4); (3) derive the continuity equation for the
reverse process; (4) why must the reverse process know the score while the forward
does not?
**Prereqs / cross-refs:** §6.3.4–§6.3.5 (PF-ODE, score), §6.2 (SDEs);
`sec_mdl-fisher-divergence` (score); forward to §6.4.3 (score-based generation),
§6.4.8 (reverse SDE vs. PF-ODE sampling).
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Bullet recap to be written once the body lands.
**Outline (planned bullets):** the ensemble of SDE paths is a density $p_t$ with
deterministic evolution · Fokker–Planck:
$\partial_t p=-\nabla\!\cdot(\mathbf{f}p)+\tfrac12\nabla\!\cdot(g^2\nabla p)$ —
drift transports, diffusion (Laplacian) smooths · everything can be written as a
continuity equation $\partial_t p+\nabla\!\cdot(p\mathbf{v})=0$ (probability is
conserved; same law as the §6.1.8 CNF trace) · the probability-flow ODE
$\mathbf{v}_t=\mathbf{f}-\tfrac12 g^2\nabla\log p_t$ is a deterministic twin with
identical marginals · the score $\nabla\log p_t$ is the *only* unknown · Anderson's
reverse SDE $[\mathbf{f}-g^2\nabla\log p_t]\,dt+g\,d\bar{\mathbf{W}}$ turns noise
back into data, again needing only the score.
**Cross-refs:** §6.1 (CNF), §6.2 (SDEs/OU), §6.4 (learning the score),
`sec_mdl-fisher-divergence`.
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Show the OU stationary Gaussian is the steady-state solution of the
   Fokker–Planck equation.
2. The probability-flow ODE and the reverse SDE drifts differ by
   $\tfrac12 g^2\nabla\log p_t$ — explain where the factor comes from and why both
   still share the marginals $p_t$.
3. Derive the continuity equation from conservation of probability mass and relate
   it to the §6.1.8 CNF log-density rule.
4. Plug the analytic OU score into the 1-D reverse SDE, integrate from the
   stationary Gaussian, and verify you recover $p_0$.
**Cross-refs:** §6.3.2–§6.3.6.
:::

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:
