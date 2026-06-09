# Score Matching, Diffusion, and Flow Matching
:label:`sec_mdl-score-matching-diffusion-flow`

The capstone. Sections 6.1–6.3 established the machinery: solvers, the forward
noising SDE, and the central reduction that *the only unknown standing between
noise and data is the score* $\nabla\log p_t$. This section turns that machinery
into the training objectives behind today's image, audio, and video generators.
The recipe is strikingly simple: **learn a score (or a velocity field) by a
plain regression, then sample by solving the learned ODE/SDE.** We develop score
matching and its tractable denoising form, recognize DDPM
(:cite:`ho2020denoising`) as a discretized variance-preserving SDE, then derive
flow matching and rectified flow as a complementary route that *prescribes* a
clean noise→data path — closing with the optimal-transport connection and a single
table that unifies score-based diffusion, DDPM, flow matching, and continuous
normalizing flows as three choices: a probability path, a training objective, and
a sampler (:cite:`song2021score`).

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §6.4, the capstone section of the
Dynamics chapter (:numref:`chap_mdl-dynamics`) and of the Mathematics-for-Deep-
Learning Part. No prose, code, or figures have been authored yet; every subsection
below is a planning stub in the standard format. The body framing for the section
as a whole:

> *Learn a score or velocity field by a simple regression, then sample by solving
> the learned ODE/SDE — unifying score-based diffusion, DDPM, and flow
> matching/rectified flow under one transport picture.*

**Prerequisites:** §6.3 (`sec_mdl-fokker-planck-probability-flow`: score,
probability-flow ODE, reverse SDE); §6.2 (`sec_mdl-sdes`: forward SDE, OU / VP);
§6.1 (`sec_mdl-odes-solvers`: solvers, CNF); 5.2 (`sec_mdl-divergences-distances`:
Fisher divergence / score matching via `sec_mdl-fisher-divergence`, Wasserstein /
optimal transport via `sec_mdl-optimal-transport`); 4.1–4.2
(`sec_mdl-random_variables`, distributions: Gaussians, conditional expectation);
3.x (`sec_mdl-gradient-based-optimization`: SGD — light).
**Capstone payoff:** the unifying table (§6.4.9) tying diffusion, flow matching,
and CNFs together; the full *train-a-regression-then-solve-an-ODE* pipeline.
**Pointer:** the *engineering* (architectures, noise schedules, classifier-free
guidance, latent diffusion) lives in the main book's generative-models chapters —
here we teach only the math that makes those work.
:::

## Score Matching
:label:`sec_mdl-score-matching`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** §6.3 told us we need the score $\nabla\log p$. The naïve way to
learn it is to minimize the **Fisher divergence** (callback
`sec_mdl-fisher-divergence`) — the mean-squared gap between our model's score and
the data's score. The problem: we do not know the data's score. Hyvärinen's
integration-by-parts identity rewrites the objective so the unknown
$\nabla\log p$ disappears, leaving a loss we *can* compute from samples.
**Outline:** 1. the explicit objective: minimize
$\tfrac12\mathbb E_{p}\|\mathbf{s}_\theta-\nabla\log p\|^2$ (the Fisher divergence,
tie to `sec_mdl-fisher-divergence`) · 2. the obstruction: $\nabla\log p$ is exactly
what we don't have · 3. Hyvärinen's identity (integration by parts) turns it into
$\mathbb E_p[\tfrac12\|\mathbf{s}_\theta\|^2+\nabla\!\cdot\mathbf{s}_\theta]+\text{const}$
— score-free · 4. why it is impractical at scale: the divergence
$\nabla\!\cdot\mathbf{s}_\theta=\operatorname{tr}(\partial\mathbf{s}_\theta/\partial\mathbf{x})$
costs $O(d)$ backward passes (same trace pain as the CNF, §6.1.8) — motivates the
denoising trick next.
**Key results to state:** objective
$J(\theta)=\tfrac12\mathbb E_{\mathbf{x}\sim p}\big\|\mathbf{s}_\theta(\mathbf{x})-\nabla\log p(\mathbf{x})\big\|^2$;
Hyvärinen (integration by parts):
$J(\theta)=\mathbb E_{p}\big[\tfrac12\|\mathbf{s}_\theta(\mathbf{x})\|^2+\nabla\!\cdot\mathbf{s}_\theta(\mathbf{x})\big]+C$;
the $\nabla\!\cdot\mathbf{s}_\theta$ trace is the scalability bottleneck.
**Diagrams:** reuse `fig_mdl-score-field` (from `sec_mdl-fisher-divergence`) — model
score field $\mathbf{s}_\theta$ overlaid on the true score field, with the
mismatch arrows the Fisher loss penalizes.
**Worked example(s):** for a 1-D Gaussian where $\nabla\log p=-(x-\mu)/\sigma^2$ is
known, minimize the explicit Fisher objective and recover $\mathbf{s}_\theta$;
verify Hyvärinen's identity numerically (the two forms give the same minimizer).
**Exercises (draft):** (1) write the Fisher objective for fitting a linear score
$\mathbf{s}_\theta=-A(\mathbf{x}-\mathbf{b})$ to a Gaussian; (2) derive Hyvärinen's
identity by integration by parts in 1-D; (3) explain why the divergence term costs
$O(d)$ and connect to the §6.1.8 trace.
**Prereqs / cross-refs:** `sec_mdl-fisher-divergence` (Fisher divergence,
normalizer-free score), §6.3.5 (the score), §6.1.8 (trace cost),
`sec_mdl-matrix-calculus-autodiff` (the divergence as a trace of a Jacobian);
forward to §6.4.2 (the cheap denoising version).
:::

## Denoising Score Matching
:label:`sec_mdl-denoising-score-matching`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The trick that made score-based models practical. Instead of
matching the score of the clean data, match the score of *Gaussian-noised* data.
Because the noising kernel is Gaussian, its score has a **closed form** — it points
from the noisy sample back toward the clean one — so the target is just
"predict the noise you added." No intractable $\nabla\log p$, no divergence term:
score matching collapses to a plain $L_2$ regression (Vincent, 2011).
**Outline:** 1. noise the data with a Gaussian kernel $p_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal N(\tilde{\mathbf{x}};\mathbf{x},\sigma^2 I)$
(this is the OU transition density of §6.2.7) · 2. the closed-form conditional
score $\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=(\mathbf{x}-\tilde{\mathbf{x}})/\sigma^2=-\boldsymbol\epsilon/\sigma$
· 3. the denoising score-matching theorem: matching the *conditional* score has the
same minimizer as matching the (intractable) *marginal* score (Vincent, 2011) ·
4. the objective is therefore "predict the denoising direction / the added noise"
— a simple regression · 5. weight across noise levels $\sigma$ (a schedule) to
cover the whole forward process; this is the bridge to the time-indexed score
$\mathbf{s}_\theta(\mathbf{x},t)$.
**Key results to state:** Gaussian kernel
$p_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal N(\tilde{\mathbf{x}};\mathbf{x},\sigma^2 I)$,
$\tilde{\mathbf{x}}=\mathbf{x}+\sigma\boldsymbol\epsilon$;
closed-form target
$\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=\dfrac{\mathbf{x}-\tilde{\mathbf{x}}}{\sigma^2}=-\dfrac{\boldsymbol\epsilon}{\sigma}$;
DSM objective
$\mathbb E_{\mathbf{x},\boldsymbol\epsilon}\big\|\mathbf{s}_\theta(\tilde{\mathbf{x}})-\tfrac{\mathbf{x}-\tilde{\mathbf{x}}}{\sigma^2}\big\|^2$,
same minimizer as marginal score matching (Vincent, 2011).
**Diagrams:** `fig_mdl-dsm-target` — noisy samples $\tilde{\mathbf{x}}$ scattered
around a data manifold, each with an arrow back to its clean origin $\mathbf{x}$
(the conditional score target), the learned field $\mathbf{s}_\theta$ tracking the
average of those arrows.
**Worked example(s):** derive the Gaussian conditional score in one line; fit
$\mathbf{s}_\theta$ on a 1-D Gaussian-mixture by denoising score matching and
overlay it on the analytic marginal score; show predicting $\boldsymbol\epsilon$
vs. predicting the score differ only by the $-1/\sigma$ scaling.
**Exercises (draft):** (1) derive the closed-form $\nabla\log p_\sigma(\tilde x\mid x)$
for the Gaussian kernel; (2) show the DSM and marginal-score objectives share a
minimizer (Vincent's identity); (3) relate "predict $\boldsymbol\epsilon$" to
"predict the score" via the $-1/\sigma$ factor.
**Prereqs / cross-refs:** §6.2.7 (Gaussian/OU transition kernel), §6.3.5 (the
score), 4.1–4.2 (Gaussian, conditional density); Vincent (2011, *A connection
between score matching and denoising autoencoders*); forward to §6.4.3 (use it
across $t$), §6.4.4 (DDPM = $\boldsymbol\epsilon$-prediction).
:::

## Score-Based Generative Modeling
:label:`sec_mdl-score-based-generative-modeling`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Assemble the pipeline. Choose a forward SDE that noises data to a
known Gaussian (§6.2); learn a *time-conditioned* score $\mathbf{s}_\theta(\mathbf{x},t)\approx\nabla\log p_t$
by denoising score matching across all noise levels (§6.4.2); then generate by
plugging that learned score into either Anderson's reverse SDE or the
probability-flow ODE (§6.3) and **solving it** with a numerical integrator (§6.1).
Sampling literally *is* an ODE/SDE solve (:cite:`song2021score`).
**Outline:** 1. the time-indexed score network $\mathbf{s}_\theta(\mathbf{x},t)$
and the noise-conditioned DSM loss integrated over $t$ · 2. forward SDE noises data
$p_0\to p_T$ (§6.2); reverse SDE / PF-ODE denoise $p_T\to p_0$ (§6.3) · 3. the
three building blocks named explicitly: forward process (§6.2) + learned score
(§6.4.2) + sampler (§6.1) · 4. variance-exploding vs. variance-preserving SDE
families (callback §6.2.5) and which classical model each recovers (SMLD vs. DDPM)
· 5. practical knobs deferred to the main book; here the math closes the loop.
**Key results to state:** training loss
$\mathcal L=\mathbb E_{t}\,\lambda(t)\,\mathbb E_{\mathbf{x}_0,\mathbf{x}_t}\big\|\mathbf{s}_\theta(\mathbf{x}_t,t)-\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{x}_0)\big\|^2$;
generation = integrate reverse SDE $[\mathbf{f}-g^2\mathbf{s}_\theta]\,dt+g\,d\bar{\mathbf{W}}$
or PF-ODE $\dot{\mathbf{x}}=\mathbf{f}-\tfrac12 g^2\mathbf{s}_\theta$ from
$\mathbf{x}_T\sim\mathcal N(\mathbf{0},\cdot)$ (Song et al., 2021).
**Diagrams:** `fig_mdl-forward-reverse-strip` — an image strip showing data → noise
(forward, top) and noise → data (reverse, bottom), with the learned score field
$\mathbf{s}_\theta$ labeled as the engine driving the reverse strip.
**Worked example(s):** end-to-end on a 1-D Gaussian-mixture — train
$\mathbf{s}_\theta(x,t)$ by noise-conditioned DSM, then sample with both the
reverse SDE and the PF-ODE, comparing the generated histogram to the true density.
**Exercises (draft):** (1) write the noise-conditioned DSM loss with a general
weighting $\lambda(t)$; (2) identify forward process, learned object, and sampler
in this pipeline; (3) contrast VE and VP forward SDEs and the model each recovers;
(4) explain why we need the score across *all* $t$, not just one noise level.
**Prereqs / cross-refs:** §6.2 (forward SDE / VE-VP), §6.3 (reverse SDE / PF-ODE),
§6.1 (solvers), §6.4.2 (DSM); main-book diffusion chapter (engineering); forward to
§6.4.4 (DDPM), §6.4.8 (sampling tradeoffs).
:::

## DDPM as a Discretized SDE
:label:`sec_mdl-ddpm-discretized-sde`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The famous Denoising Diffusion Probabilistic Model
(:cite:`ho2020denoising`) is not a separate theory — it is exactly the
**Euler–Maruyama discretization of a variance-preserving (OU-type) SDE**, and its
celebrated "$\boldsymbol\epsilon$-prediction" loss is denoising score matching in
disguise. Recognizing this collapses two literatures into one
(:cite:`song2021score`).
**Outline:** 1. the discrete DDPM forward chain
$\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol\epsilon$
as Euler–Maruyama on the VP-SDE $d\mathbf{x}=-\tfrac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{W}$
(an OU process, §6.2.7) · 2. the closed-form marginal $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$
— a Gaussian kernel, so DSM applies directly · 3. the DDPM simple loss
$\mathbb E\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\|^2$ is
denoising score matching with a particular weighting (§6.4.2);
$\mathbf{s}_\theta=-\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ · 4. sampling
= the reverse VP-SDE (ancestral sampling) or its PF-ODE (DDIM-style deterministic
sampling) · 5. takeaway: discrete DDPM and continuous score-based models are one
object viewed at different resolutions.
**Key results to state:** VP-SDE
$d\mathbf{x}=-\tfrac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{W}$ (OU,
§6.2.7); marginal $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$;
DDPM loss $\mathbb E\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\|^2$
$=$ reweighted DSM; score relation
$\mathbf{s}_\theta(\mathbf{x}_t,t)=-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)/\sqrt{1-\bar\alpha_t}$.
**Diagrams:** reuse `fig_mdl-forward-reverse-strip` with the discrete timesteps
$t=1,\dots,T$ marked, annotating the forward chain as Euler–Maruyama steps on the
VP-SDE.
**Worked example(s):** show numerically that the DDPM forward chain and the VP-SDE
Euler–Maruyama produce matching marginals; verify the
$\mathbf{s}_\theta=-\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ identity on a
Gaussian; sample a 1-D toy with ancestral (SDE) vs. deterministic (PF-ODE/DDIM)
steps.
**Exercises (draft):** (1) show the DDPM forward step is Euler–Maruyama on the
VP-SDE; (2) derive the closed-form marginal $\mathbf{x}_t$ from $\mathbf{x}_0$;
(3) show the $\boldsymbol\epsilon$-loss equals reweighted DSM and identify the
weighting; (4) relate $\boldsymbol\epsilon_\theta$ to the score.
**Prereqs / cross-refs:** §6.2.6–§6.2.7 (Euler–Maruyama, OU/VP), §6.4.2 (DSM),
§6.3 (reverse SDE / PF-ODE); :cite:`ho2020denoising`, :cite:`song2021score`;
main-book diffusion chapter; forward to §6.4.8 (sampling), §6.4.9 (the table).
:::

## Flow Matching
:label:`sec_mdl-flow-matching`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A complementary route to the same goal. Instead of *deriving* a
velocity field from a noising SDE, flow matching lets us **prescribe** a clean
probability path from noise to data and directly regress the velocity field that
realizes it — training a continuous normalizing flow (§6.1.8) *without* simulating
the ODE. The key obstacle (the marginal velocity is intractable) is solved by
**conditional flow matching**: regress the per-sample conditional velocity, which
has the *same gradient* as the intractable marginal objective (Lipman et al.,
2023).
**Outline:** 1. prescribe a probability path $p_t$ interpolating $p_0=$ noise and
$p_1=$ data, with an associated marginal velocity $\mathbf{u}_t$ obeying the
continuity equation (§6.3.3) · 2. the flow-matching loss
$\mathbb E_{t,\mathbf{x}}\|\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x})\|^2$
— but $\mathbf{u}_t$ is intractable · 3. **conditional flow matching (CFM)**: pick a
tractable *conditional* path $p_t(\mathbf{x}\mid\mathbf{x}_1)$ with a known
conditional velocity $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)$, and regress against
*that* · 4. the CFM theorem: the conditional and marginal objectives have *the same
gradient*, hence the same minimizer — so we get the marginal field for free,
simulation-free (Lipman et al., 2023) · 5. Gaussian-conditional-path special case
recovers diffusion-style targets; flow matching generalizes beyond Gaussians.
**Key results to state:** FM loss
$\mathbb E_{t,\mathbf{x}\sim p_t}\|\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x})\|^2$;
CFM loss
$\mathbb E_{t,\mathbf{x}_1,\mathbf{x}\sim p_t(\cdot\mid\mathbf{x}_1)}\|\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\|^2$;
$\nabla_\theta\mathcal L_{\text{FM}}=\nabla_\theta\mathcal L_{\text{CFM}}$ (same
minimizer; Lipman, Chen, Ben-Hamu, Nickel, Le, 2023).
**Diagrams:** `fig_mdl-learned-velocity-field` — a 2-D arrow field
$\mathbf{v}_\theta(\mathbf{x},t)$ transporting a Gaussian blob into a two-moons
target, with several sample trajectories overlaid following the arrows.
**Worked example(s):** Gaussian → two-moons via CFM with a Gaussian conditional
path; train $\mathbf{v}_\theta$ and sample by integrating the ODE (§6.1); verify
the conditional and marginal losses give the same learned field on a 1-D toy.
**Exercises (draft):** (1) write the conditional velocity for a Gaussian
conditional path; (2) show the CFM and FM objectives share a gradient (the core
theorem); (3) check the velocity field satisfies the continuity equation for the
prescribed path; (4) contrast "derive velocity from an SDE" (diffusion) with
"prescribe the path" (flow matching).
**Prereqs / cross-refs:** §6.1.8 (CNF velocity), §6.3.3 (continuity equation),
§6.4.2 (Gaussian conditional kernels); Lipman et al. (2023, *Flow Matching for
Generative Modeling*); forward to §6.4.6 (rectified / straight paths), §6.4.7 (OT
coupling), §6.4.9 (the table).
:::

## Rectified Flow and Straight Paths
:label:`sec_mdl-rectified-flow`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The simplest possible flow-matching path is a *straight line*:
linearly interpolate between a noise sample and a data sample,
$\mathbf{x}_t=(1-t)\mathbf{x}_0+t\,\mathbf{x}_1$. Then the target velocity is the
constant $\mathbf{x}_1-\mathbf{x}_0$ — a trivially simple regression — and because
the learned trajectories are nearly straight, the ODE sampler needs *very few*
steps (even one). This is rectified flow (Liu et al., 2023).
**Outline:** 1. the linear interpolation path and its constant conditional velocity
$\mathbf{x}_1-\mathbf{x}_0$ · 2. why straightness matters: a straight trajectory is
integrated *exactly* by a single Euler step, so fewer solver steps for the same
quality (callback §6.1.4 error orders) · 3. the *reflow* procedure: re-pair
(noise, data) by the learned map and retrain, iteratively straightening the paths
toward an optimal-transport map (Liu et al., 2023) · 4. relationship to
flow matching: rectified flow is CFM with the straight-line conditional path ·
5. the step-count payoff that powers fast/distilled samplers.
**Key results to state:** path $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\,\mathbf{x}_1$;
constant target velocity $\dot{\mathbf{x}}_t=\mathbf{x}_1-\mathbf{x}_0$;
rectified-flow loss
$\mathbb E_{t,\mathbf{x}_0,\mathbf{x}_1}\|\mathbf{v}_\theta(\mathbf{x}_t,t)-(\mathbf{x}_1-\mathbf{x}_0)\|^2$;
reflow straightens trajectories and reduces transport cost toward the OT map
(Liu, Gong, Liu, 2023).
**Diagrams:** `fig_mdl-ot-vs-curved-paths` — curved diffusion trajectories (many
solver steps) beside straight rectified-flow / OT trajectories (few steps), each
annotated with the number of Euler steps needed for comparable accuracy.
**Worked example(s):** on a 2-D Gaussian → two-moons toy, train rectified flow
with the linear path and show single/few-step Euler sampling already matches the
target where curved diffusion paths need many steps; demonstrate one reflow
iteration straightening the paths.
**Exercises (draft):** (1) derive the constant target velocity from the linear
interpolation; (2) argue why a straight path is integrated exactly by one Euler
step (callback §6.1.4); (3) explain what reflow does and why it lowers transport
cost; (4) express rectified flow as a special case of CFM (§6.4.5).
**Prereqs / cross-refs:** §6.4.5 (flow matching / CFM), §6.1.4 (Euler error
orders); Liu et al. (2023, *Flow Straight and Fast: rectified flow*); forward to
§6.4.7 (OT connection), §6.4.8 (few-step sampling), §6.4.9 (the table).
:::

## The Optimal-Transport Connection
:label:`sec_mdl-ot-connection`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Why are straight paths special? Because the
*minimum-kinetic-energy* way to move one distribution onto another follows
straight geodesics — that is the dynamic (Benamou–Brenier) view of optimal
transport (callback `sec_mdl-optimal-transport`). Coupling noise and data by a
min-cost ($W_2$) transport plan before flow matching (OT-CFM) makes the learned
paths approach those straight geodesics, which is exactly what rectified flow
chases by reflow.
**Outline:** 1. recall the $W_2$ optimal-transport distance and its coupling /
transport plan (callback `sec_mdl-optimal-transport`) · 2. the dynamic
(Benamou–Brenier) formulation: $W_2^2$ = minimum kinetic energy
$\int\!\int\|\mathbf{v}_t\|^2 p_t\,d\mathbf{x}\,dt$ over paths bridging $p_0,p_1$ —
minimized by *straight-line* transport at constant speed · 3. **OT-CFM**: sample the
(noise, data) pair from an optimal coupling (a mini-batch OT plan) rather than
independently, so conditional paths align into near-geodesics (Tong et al., 2023;
Pooladian et al., 2023) · 4. the kinetic-energy / curvature view ties straightness,
few-step sampling, and OT together · 5. caveat: exact OT is expensive; mini-batch
OT and reflow are the practical approximations.
**Key results to state:** Benamou–Brenier
$W_2^2(p_0,p_1)=\min_{p_t,\mathbf{v}_t}\int_0^1\!\!\int\|\mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x})\,d\mathbf{x}\,dt$
subject to continuity (§6.3.3), minimized by straight constant-speed paths;
OT-CFM couples $(\mathbf{x}_0,\mathbf{x}_1)$ by a $W_2$ plan to straighten paths.
**Diagrams:** reuse `fig_mdl-ot-vs-curved-paths` — emphasize the straight OT
geodesics as the minimum-energy bridge, with independently-coupled (curved) paths
for contrast.
**Worked example(s):** 1-D OT plan between two Gaussians (the monotone map) and
show the straight-line interpolation it induces; on a 2-D toy compare independent
coupling vs. mini-batch-OT coupling and measure the resulting path straightness /
step count.
**Exercises (draft):** (1) show the dynamic-OT minimizer is the constant-speed
straight path; (2) explain why OT coupling reduces path crossings and curvature;
(3) connect OT-CFM straightening to rectified-flow reflow (§6.4.6);
(4) relate kinetic energy to the number of solver steps needed.
**Prereqs / cross-refs:** `sec_mdl-optimal-transport` (W₂ / coupling), §6.4.5–§6.4.6
(CFM, rectified flow), §6.3.3 (continuity); Tong et al. (2023), Pooladian et al.
(2023); forward to §6.4.8 (step counts), §6.4.9 (the table).
:::

## Sampling = Solving the Learned Dynamics
:label:`sec_mdl-sampling-learned-dynamics`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Whatever we learned — a score $\mathbf{s}_\theta$ or a velocity
$\mathbf{v}_\theta$ — generation is the *same* act: plug it into a §6.1 solver and
integrate from noise to data. The remaining design choice is **stochastic vs.
deterministic**: the reverse SDE (more steps, sample diversity, robust to score
error) versus the probability-flow ODE (few steps, deterministic, exact
likelihood). Step count vs. quality is the central practical tradeoff.
**Outline:** 1. the unified statement: sampling = numerically solve
$\dot{\mathbf{x}}=\mathbf{v}_\theta$ (ODE) or $d\mathbf{x}=\dots+g\,d\bar{\mathbf{W}}$
(SDE), reusing §6.1 / §6.2.6 solvers · 2. reverse SDE (§6.3.6): stochastic, tends to
need more steps but can correct accumulated error; PF-ODE (§6.3.4): deterministic,
few steps, exact likelihood via the CNF trace (§6.1.8) · 3. why straighter paths
(§6.4.6–§6.4.7) need fewer steps — connect to the §6.1.4 error orders · 4. solver
choice (Euler vs. higher-order / adaptive, §6.1.4–§6.1.5) as the speed/quality dial
· 5. distillation as "learn to take bigger steps" (pointer to main book).
**Key results to state:** ODE sampler $\dot{\mathbf{x}}=\mathbf{f}-\tfrac12 g^2\mathbf{s}_\theta$
(or $\mathbf{v}_\theta$); SDE sampler
$[\mathbf{f}-g^2\mathbf{s}_\theta]\,dt+g\,d\bar{\mathbf{W}}$; global solver error
$O(h^p)$ (§6.1.4) ⇒ straighter / lower-curvature paths tolerate larger $h$ (fewer
steps); PF-ODE gives exact likelihood via $\int\nabla\!\cdot\mathbf{v}_\theta\,dt$
(§6.1.8).
**Diagrams:** reuse `fig_mdl-ot-vs-curved-paths` annotated with step counts and
"SDE (many steps) vs. ODE (few steps)"; optional inset of solver error vs. step
size (reuse `fig_mdl-euler-error-vs-h`).
**Worked example(s):** on the 1-D mixture, sample with reverse SDE vs. PF-ODE and
vary the number of solver steps, plotting sample quality (e.g. KL/W₁ to truth) vs.
step count; show RK4 reaches target quality in fewer steps than Euler on a
flow-matching toy.
**Exercises (draft):** (1) write the Euler step of the PF-ODE sampler; (2) explain
why straighter paths permit larger $h$ (callback §6.1.4); (3) contrast reverse-SDE
and PF-ODE samplers on diversity, step count, and likelihood; (4) estimate the step
count for a target accuracy from the solver order.
**Prereqs / cross-refs:** §6.1.4–§6.1.5 (solvers, error orders), §6.2.6
(Euler–Maruyama), §6.3.4 & §6.3.6 (PF-ODE / reverse SDE), §6.1.8 (likelihood);
forward to §6.4.9 (the table); main-book sampling/distillation material.
:::

## A Unifying Table
:label:`sec_mdl-unifying-table`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The capstone of the capstone. Score-based diffusion, DDPM, flow
matching, rectified flow, and continuous normalizing flows look like different
models but are *the same template* with three free choices: a **probability path**
(how noise turns into data), a **training objective** (what regression you run),
and a **sampler** (which dynamics you solve). One table makes the whole zoo legible.
**Outline:** 1. assemble the master table with columns = {probability path,
training objective, sampler/dynamics} and rows = {score-based SDE, DDPM, flow
matching / CFM, rectified flow, OT-CFM, CNF / FFJORD} · 2. walk three rows in
detail (DDPM, flow matching, rectified flow) tying back to their home subsections ·
3. the thesis: choose a path and an objective, and the sampler (and its
speed/quality behavior) follows · 4. close the Part: this synthesizes calculus
(Ch2), probability (Ch4), optimization (Ch3), and divergences (Ch5) into the math
of modern generators.
**Key results to state (the table, schematically):**
score-based SDE → noising SDE path / Fisher-divergence (DSM) objective / reverse
SDE or PF-ODE sampler;
DDPM → VP-SDE (OU) path / $\boldsymbol\epsilon$-prediction (= reweighted DSM) /
ancestral SDE or DDIM ODE;
flow matching → prescribed path / conditional-velocity regression (CFM) / ODE;
rectified flow → straight-line path / constant-velocity regression / few-step ODE;
CNF → free-form ODE path / maximum likelihood via the trace (§6.1.8) / ODE.
**Diagrams:** `fig_mdl-unifying-table` — the master grid rendered as a labeled
table (rows = model families, columns = path / objective / sampler), color-coded by
which §6.x subsection supplies each cell.
**Worked example(s):** none new — this section *synthesizes*; it fills the table on
the toy examples already run (the 1-D mixture from §6.4.3/§6.4.4 and the 2-D
two-moons from §6.4.5/§6.4.6), pointing each cell back to its home worked example.
**Exercises (draft):** (1) place a new model (e.g. variance-exploding SMLD, or
stochastic interpolants) into the table; (2) given a probability path and
objective, predict the natural sampler and its step-count behavior; (3) argue which
rows are "the same model" up to reparameterization (DDPM vs. score-based VP);
(4) which choices favor few-step sampling and why?
**Prereqs / cross-refs:** all of §6.4 and §6.1–§6.3; `sec_mdl-divergences-distances`
(Fisher / OT); main-book generative-models chapters.
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Bullet recap to be written once the body lands.
**Outline (planned bullets):** score matching minimizes the Fisher divergence but
needs the unknown $\nabla\log p$; Hyvärinen removes it, denoising score matching
makes it a cheap Gaussian-kernel regression ("predict the noise") · score-based
generation = forward SDE (§6.2) + learned time-conditioned score (§6.4.2) + reverse
SDE / PF-ODE sampler (§6.3) · DDPM is Euler–Maruyama on the VP (OU) SDE and its
$\boldsymbol\epsilon$-loss is reweighted DSM · flow matching prescribes a path and
regresses the velocity; CFM makes the target tractable with the same gradient ·
rectified flow uses straight-line paths → constant target velocity → few-step
sampling; OT coupling explains why straight = optimal · sampling is always an
ODE/SDE solve, deterministic (few steps, exact likelihood) or stochastic (more
steps, diversity) · one table unifies the zoo as {path, objective, sampler}.
**Cross-refs:** §6.1 (solvers/CNF), §6.2 (forward SDE/VP), §6.3 (score / reverse
dynamics), `sec_mdl-divergences-distances` (Fisher / OT).
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Derive the closed-form conditional score $\nabla\log p_t(\mathbf{x}\mid\mathbf{x}_0)$
   for a Gaussian noising kernel and identify it as the denoising-score-matching
   target.
2. Show conditional flow matching has the same minimizer (same gradient) as
   marginal flow matching.
3. Derive the rectified-flow target velocity from the linear interpolation path and
   argue why straightness cuts the number of solver steps (callback §6.1.4).
4. Show the DDPM $\boldsymbol\epsilon$-loss equals reweighted denoising score
   matching and identify the weighting; relate $\boldsymbol\epsilon_\theta$ to the
   score.
5. Fill in the unifying table (§6.4.9) for a new model family and predict its
   sampler and step-count behavior.
**Cross-refs:** §6.4.1–§6.4.9.
:::

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:
