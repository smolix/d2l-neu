# Divergences and Distances Between Distributions
:label:`sec_mdl-divergences-distances`

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §5.2, the depth section of the
Information-Theory chapter. No prose, code, or figures have been authored yet;
every subsection below is a planning stub in the standard format. The body
framing for the section as a whole:

> *Every modern generative model is "make my distribution close to the data" —
> and **which notion of close** (KL, JS, Wasserstein, Fisher) determines the
> objective, the failure modes, and the gradients. This section is the map from
> divergence to training loss.*

**Prerequisites:** §5.1 (`sec_mdl-information_theory`: KL/CE, Gibbs);
4.1–4.2 (`sec_mdl-random_variables`, distributions / densities); 3.2 (convexity,
convex conjugate, Jensen); Ch1 (norms / inner products, for IPMs and RKHS);
2.4 (integrals).
**Capstone payoff:** the divergence ↔ generative-objective map (§5.2.10).
**Forward bridge:** §5.2.8 (Fisher / score) feeds the Ch6 score-matching /
diffusion / flow-matching capstone (`sec_mdl-score-matching-diffusion-flow`).
:::

## What Is a Divergence?
:label:`sec_mdl-what-is-a-divergence`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A divergence is the weakest useful notion of "how far apart two
distributions are" — non-negative and zero only when they coincide, but *not*
required to be symmetric or to satisfy a triangle inequality.
**Outline:** 1. axioms: $D(P\|Q)\ge 0$ and $D(P\|Q)=0 \iff P=Q$ · 2. why we drop
symmetry and the triangle inequality (KL fails both) · 3. contrast with genuine
*metrics* on distributions (Total Variation, Wasserstein, Hellinger *are* metrics)
· 4. taxonomy preview: $f$-divergences vs. integral probability metrics vs.
optimal-transport distances — three families, partially overlapping.
**Key results to state:** $D(P\|Q)\ge 0,\ =0\iff P=Q$; KL is asymmetric (callback
to `eq_mdl-gaussian_kl`); a *metric* additionally needs $D(P,Q)=D(Q,P)$ and
$D(P,R)\le D(P,Q)+D(Q,R)$.
**Diagrams:** `fig_mdl-divergence-taxonomy` — Venn/tree of the three families
showing where KL, TV, JS, MMD, $W_1$ land and which are metrics.
**Worked example(s):** show KL violates symmetry numerically (reuse the two
Gaussians from §5.1); show TV is symmetric on the same pair.
**Exercises (draft):** which of {KL, reverse-KL, TV, $W_1$, Hellinger} are metrics?
prove TV satisfies the triangle inequality.
**Prereqs / cross-refs:** §5.1 (KL); `sec_linear-algebra` (norms/metrics).
:::

## The f-Divergence Family
:label:`sec_mdl-f-divergences`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A huge swath of classical divergences are a single template — a
convex *generator* $f$ applied to the likelihood ratio $p/q$ and averaged under
$Q$ — so KL, reverse-KL, TV, $\chi^2$, Hellinger, and Jensen–Shannon are all "the
same object with a different $f$."
**Outline:** 1. define $D_f(P\|Q) = E_{x\sim Q}\!\left[f\!\big(p(x)/q(x)\big)\right]$
for convex $f$ with $f(1)=0$ · 2. non-negativity from Jensen (callback to Gibbs,
§5.1) · 3. the **generator table** (the centerpiece) · 4. how $f''(1)$ sets the
local (Fisher-information) curvature shared by the whole family · 5. note Rényi /
$\alpha$-divergences as a one-parameter bridge.
**Key results to state:**
$D_f(P\|Q)=E_Q[f(p/q)]$;
KL $\leftrightarrow f(u)=u\log u$;
reverse-KL $\leftrightarrow f(u)=-\log u$;
TV $\leftrightarrow f(u)=\tfrac12|u-1|$;
$\chi^2 \leftrightarrow f(u)=(u-1)^2$;
squared Hellinger $\leftrightarrow f(u)=(\sqrt u-1)^2$;
Jensen–Shannon $\leftrightarrow$ the symmetrized mixture form;
$D_f\ge 0$ with $=0\iff P=Q$ when $f$ strictly convex at $1$.
**Diagrams:** `fig_mdl-f-div-generator-table` — table/plot pairing each generator
$f(u)$ (curve on $u>0$, marked $f(1)=0$) with the divergence it produces and the
one-line DL use.
**Worked example(s):** evaluate $\chi^2$ and squared-Hellinger between two
3-class categoricals by hand and in code; recover KL from $u\log u$ to confirm the
template.
**Exercises (draft):** (1) show $f(u)=u\log u$ gives KL and $f(u)=-\log u$ gives
reverse-KL; (2) verify TV $=\tfrac12\|p-q\|_1$ from $f(u)=\tfrac12|u-1|$;
(3) show any $f$ and $\tilde f(u)=f(u)+c(u-1)$ give the same divergence.
**Prereqs / cross-refs:** §5.1 (KL as the $u\log u$ case); 3.2 (convexity);
forward to §5.2.3 (dual), §5.2.10 (objective map).
:::

## Variational / Dual Representation (f-GAN)
:label:`sec_mdl-f-gan-dual`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Using the convex conjugate $f^*$, *any* $f$-divergence rewrites
as a supremum over a "critic" function — turning an integral you cannot compute
(it needs the densities) into an adversarial game you can optimize with samples.
This is the mathematical core of the f-GAN.
**Outline:** 1. recall the Fenchel conjugate $f^*(t)=\sup_u\{ut-f(u)\}$ (callback
to 3.2 / 3.3 duality) · 2. the variational lower bound
$D_f(P\|Q)=\sup_T\{E_P[T(x)] - E_Q[f^*(T(x))]\}$ via bounding $f$ by its tangents
· 3. parameterize $T$ by a neural critic $\Rightarrow$ minimax training: critic
pushes the bound up toward $D_f$, generator pushes it down · 4. the original GAN
(Goodfellow) as the special case where $f$ yields Jensen–Shannon · 5. practical
note: the bound is tight only at the optimal critic, so estimates are biased when
the critic is under-trained.
**Key results to state:**
$f^*(t)=\sup_{u}\{ut-f(u)\}$;
$D_f(P\|Q)=\sup_{T}\big(E_{x\sim P}[T(x)]-E_{x\sim Q}[f^*(T(x))]\big)$
(Nowozin et al., *f-GAN*, 2016);
GAN $\approx$ Jensen–Shannon special case.
**Diagrams:** `fig_mdl-f-div-tangent-bound` — convex $f$ with a family of tangent
lines, the sup-over-tangents picture that *is* the variational bound.
**Worked example(s):** compute $f^*$ for $\chi^2$ ($f(u)=(u-1)^2$) and write the
explicit critic objective; show the bound is exact at $T=f'(p/q)$.
**Exercises (draft):** (1) derive $f^*$ for KL and recover the
Donsker–Varadhan-style bound (forward to §5.3.4); (2) f-GAN bound for $\chi^2$;
(3) show the optimal critic is $T^\star=f'(p/q)$.
**Prereqs / cross-refs:** §5.2.2 ($f$-divergences); 3.2/3.3 (convex conjugate,
duality); forward to §5.3.4 (variational MI bounds reuse this exact machinery);
GAN chapter in the main book.
:::

## Forward vs. Reverse KL: Mode-Covering vs. Mode-Seeking
:label:`sec_mdl-fwd-vs-rev-kl`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** *The* single most useful practical intuition in the chapter:
the *direction* of the KL you minimize decides whether your model spreads to
cover all the data's modes (forward / M-projection) or collapses onto one
(reverse / I-projection) — and this is exactly the difference between maximum
likelihood and variational inference.
**Outline:** 1. forward $D_{\textrm{KL}}(P\|Q_\theta)$ = "mass-covering /
zero-avoiding": $Q$ must put mass wherever $P$ does, or pay an infinite penalty ·
2. reverse $D_{\textrm{KL}}(Q_\theta\|P)$ = "mode-seeking / zero-forcing": $Q$ is
penalized for putting mass where $P$ has none, so it hugs one mode · 3. MLE
minimizes the forward KL to the data; the ELBO / variational inference minimizes
the reverse KL · 4. consequence for generative models: forward $\to$ blurry but
inclusive, reverse $\to$ sharp but mode-dropping.
**Key results to state:**
forward $D_{\textrm{KL}}(P\|Q)=E_P[\log p/q]$ blows up when $q\to 0$ on $\textrm{supp}(P)$;
reverse $D_{\textrm{KL}}(Q\|P)=E_Q[\log q/p]$ blows up when $p\to 0$ on $\textrm{supp}(Q)$;
MLE $\equiv$ forward, ELBO $\equiv$ reverse (callback to §5.1 MLE = min-KL-to-empirical).
**Diagrams:** `fig_mdl-fwd-vs-rev-kl` — a bimodal target $P$ with the *single*
Gaussian $Q$ that minimizes forward KL (broad, covering both modes) overlaid on
the one that minimizes reverse KL (narrow, sitting on one mode).
**Worked example(s):** numerically fit a single Gaussian $Q$ to a 2-component
mixture $P$ under each KL direction; report the two different optima (and that
reverse-KL has two symmetric local optima, one per mode).
**Exercises (draft):** (1) explain which direction a VAE uses and why it can drop
modes; (2) show forward KL forces $\textrm{supp}(Q)\supseteq\textrm{supp}(P)$;
(3) reproduce the two reverse-KL local optima.
**Prereqs / cross-refs:** §5.1 (KL, MLE); §5.2.10 (objective map); main-book
VAE / variational-inference chapters.
:::

## Total Variation and Pinsker's Inequality
:label:`sec_mdl-tv-pinsker`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Total variation is the most interpretable distance — the
largest possible gap in probability the two distributions assign to any event —
and Pinsker's inequality ties it to KL, so *bounding KL bounds everything you
could ever test*.
**Outline:** 1. define $\textrm{TV}(P,Q)=\sup_A |P(A)-Q(A)| = \tfrac12\|p-q\|_1$
and note it is a metric · 2. operational meaning: TV is the best achievable
advantage of any binary test distinguishing $P$ from $Q$ · 3. Pinsker:
$\textrm{TV}(P,Q)\le\sqrt{\tfrac12 D_{\textrm{KL}}(P\|Q)}$ · 4. why this matters:
a small KL training loss certifies indistinguishability under *any* downstream
test.
**Key results to state:**
$\textrm{TV}(P,Q)=\tfrac12\sum_x|p(x)-q(x)|=\tfrac12\|p-q\|_1$;
$\textrm{TV} \le \sqrt{\tfrac12\,D_{\textrm{KL}}(P\|Q)}$ (Pinsker).
**Diagrams:** `fig_mdl-tv-area` — two pmf bar charts with the shaded
$\tfrac12\sum|p-q|$ area, plus the optimal distinguishing event $A=\{p>q\}$.
**Worked example(s):** TV between two coins; verify Pinsker numerically against
the closed-form KL for two Gaussians (reuse `eq_mdl-gaussian_kl`).
**Exercises (draft):** (1) prove $\textrm{TV}=\tfrac12\|p-q\|_1$; (2) show the sup
is attained at $A=\{x:p(x)>q(x)\}$; (3) check Pinsker is not tight for far-apart
Gaussians and explain.
**Prereqs / cross-refs:** §5.1 (KL); `sec_linear-algebra` ($\ell_1$ norm);
§5.2.2 (TV as the $f=\tfrac12|u-1|$ divergence).
:::

## Integral Probability Metrics and MMD
:label:`sec_mdl-ipm-mmd`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A complementary family to $f$-divergences: instead of averaging
a function of the density ratio, an IPM measures the *largest difference in
expectation* of any test function drawn from a chosen class — which is exactly
what you can estimate from samples with no density at all.
**Outline:** 1. define $\textrm{IPM}_{\mathcal F}(P,Q)=\sup_{f\in\mathcal F}
\big(E_P f - E_Q f\big)$ · 2. choosing $\mathcal F$ = the RKHS unit ball gives the
**Maximum Mean Discrepancy (MMD)** (Gretton et al., kernel two-sample test) ·
3. kernel mean embedding $\mu_P=E_{x\sim P}[k(x,\cdot)]$ and
$\textrm{MMD}^2=\|\mu_P-\mu_Q\|_{\mathcal H}^2$, fully expressible in kernel
evaluations — the sample estimator · 4. choosing $\mathcal F$ = 1-Lipschitz
functions gives Wasserstein-1 (forward to §5.2.7) · 5. why MMD is attractive for
generative models: differentiable, sample-based, no adversary needed (MMD-GAN).
**Key results to state:**
$\textrm{IPM}_{\mathcal F}(P,Q)=\sup_{f\in\mathcal F}(E_P f - E_Q f)$;
$\textrm{MMD}^2(P,Q)=E_{x,x'}k(x,x')+E_{y,y'}k(y,y')-2E_{x,y}k(x,y)$;
$\mathcal F=\{$1-Lipschitz$\}\Rightarrow$ IPM $=W_1$ (link to §5.2.7).
**Diagrams:** `fig_mdl-mmd-embedding` — two point clouds mapped to their kernel
mean embeddings in feature space, with the gap $\|\mu_P-\mu_Q\|$ marked.
**Worked example(s):** compute the unbiased MMD$^2$ estimator with a Gaussian
kernel between two small 2-D samples; show it $\to 0$ as the samples come from the
same distribution.
**Exercises (draft):** (1) expand $\|\mu_P-\mu_Q\|^2$ into the three-kernel-term
estimator; (2) show MMD is a metric when the kernel is characteristic;
(3) relate the IPM function class to $W_1$ and to TV.
**Prereqs / cross-refs:** Ch1 (inner products, RKHS norm); §5.2.5 (TV is also an
IPM, over $\{f:\|f\|_\infty\le\tfrac12\}$); §5.2.7 ($W_1$ as the Lipschitz IPM).
:::

## Optimal Transport and the Wasserstein Distance
:label:`sec_mdl-optimal-transport`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** When two distributions have *disjoint or barely-overlapping
support* — exactly the situation early in GAN training — KL and JS are flat
(constant, no gradient), but the Wasserstein / earth-mover distance still varies
smoothly because it measures *how far mass must move*, not just whether supports
overlap. This is why WGAN exists.
**Outline:** 1. primal: $W_1(P,Q)=\inf_{\gamma\in\Pi(P,Q)}E_{(x,y)\sim\gamma}\|x-y\|$,
the minimum-cost coupling (transport plan) · 2. the
**Kantorovich–Rubinstein dual** $W_1(P,Q)=\sup_{\|f\|_{\textrm{Lip}}\le 1}
\big(E_P f - E_Q f\big)$ — i.e., $W_1$ is the 1-Lipschitz IPM (callback to §5.2.6)
· 3. WGAN: the critic approximates the optimal 1-Lipschitz potential $f$; the
generator descends $W_1$ · 4. enforcing the Lipschitz constraint: weight clipping
$\to$ gradient penalty (WGAN-GP, Gulrajani et al.) $\to$ spectral normalization ·
5. the disjoint-support example that motivates the whole thing.
**Key results to state:**
$W_1(P,Q)=\inf_{\gamma\in\Pi(P,Q)}\!\int\|x-y\|\,d\gamma$;
$W_1(P,Q)=\sup_{\|f\|_{\textrm{Lip}}\le1}(E_P f - E_Q f)$ (Kantorovich–Rubinstein);
1-D closed form $W_1(P,Q)=\int|F_P(t)-F_Q(t)|\,dt$.
**Diagrams:** `fig_mdl-ot-transport-plan` — two 1-D distributions with arrows
showing the optimal mass transport plan (earth-mover), and a side panel where two
disjoint spikes have constant JS/KL but linearly-varying $W_1$ as they separate.
**Worked example(s):** 1-D $W_1$ via sorted samples / the CDF formula
$W_1=\int|F_P-F_Q|$, checked against a small linear-program solve of the primal;
the disjoint-support spikes showing $W_1$ smooth where JS is flat.
**Exercises (draft):** (1) prove the 1-D $W_1=\int|F_P-F_Q|$ formula and verify in
code; (2) for two point masses at distance $d$, show JS is constant but $W_1=d$;
(3) explain why gradient penalty targets unit-norm gradients (the K–R constraint).
**Prereqs / cross-refs:** §5.2.6 (IPM/Lipschitz class); 2.4 (integrals);
3.3 (duality, optional); §5.2.10 (WGAN in the objective map); Ch6 §6.4
(OT-conditional flow matching reuses transport plans).
:::

## Fisher Divergence and Score Matching
:label:`sec_mdl-fisher-divergence`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A divergence that compares the *gradients of the log-densities*
(the "scores") rather than the densities themselves — which is the key trick that
lets diffusion models train without ever computing the intractable normalizing
constant. This is the bridge to Ch6.
**Outline:** 1. define $D_F(P\|Q)=\tfrac12 E_{x\sim P}\|\nabla_x\log p(x) -
\nabla_x\log q(x)\|^2$ · 2. the score $s(x)=\nabla_x\log p(x)$ is independent of
the normalizer $Z$, since $\nabla_x\log(\tilde p/Z)=\nabla_x\log\tilde p$ · 3.
Hyvärinen's integration-by-parts identity that removes the unknown $\nabla\log p$
from the objective (state result, defer full derivation to Ch6) · 4. relation to
KL: Fisher divergence is the "local"/derivative cousin and controls KL along
diffusion paths (de Bruijn-style identity, light).
**Key results to state:**
$D_F(P\|Q)=\tfrac12 E_P\|\nabla\log p - \nabla\log q\|^2$;
$\nabla_x\log p(x)$ does not depend on $Z=\int\tilde p$;
Hyvärinen score-matching objective $E_P[\tfrac12\|s_\theta\|^2 + \nabla\!\cdot s_\theta]$ (state, prove in Ch6).
**Diagrams:** `fig_mdl-score-field` — the score vector field $\nabla\log p$ of a
2-D density (arrows pointing uphill toward high density), contrasted with a
mismatched model's score field.
**Worked example(s):** closed-form score of a Gaussian $\nabla\log p=-(x-\mu)/\sigma^2$;
Fisher divergence between two Gaussians in closed form; confirm the normalizer
drops by adding an arbitrary constant to $\log\tilde p$.
**Exercises (draft):** (1) Fisher divergence between $\mathcal N(\mu_1,\sigma^2)$
and $\mathcal N(\mu_2,\sigma^2)$; (2) show $\nabla\log p$ is invariant to
rescaling $p$ by any constant; (3) sketch how integration by parts removes
$\nabla\log p$ from the loss.
**Prereqs / cross-refs:** 2.2–2.3 (gradient, divergence operator); §5.1
(differential-entropy caveat — scores *are* reparameterization-friendly);
**bridge to Ch6** `sec_mdl-score-matching-diffusion-flow` (denoising score
matching, diffusion).
:::

## Stein Discrepancy (Brief)
:label:`sec_mdl-stein-discrepancy`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A short coda: the kernel Stein discrepancy measures how far a
sample is from a target $P$ using *only the score of $P$* and a kernel — never
the normalizer and never samples from $P$ — making it the natural goodness-of-fit
test for unnormalized models.
**Outline:** 1. Stein's identity $E_{x\sim P}[\mathcal A_P f(x)]=0$ for the Stein
operator $\mathcal A_P f = f\,\nabla\log p + \nabla f$ · 2. kernel Stein
discrepancy = sup of $E_Q[\mathcal A_P f]$ over the RKHS unit ball (an IPM-flavored
construction, callback to §5.2.6) · 3. one-sentence uses: sample-quality
diagnostics, Stein variational gradient descent (SVGD).
**Key results to state:**
$E_{x\sim P}[\nabla\log p(x)\,f(x) + \nabla f(x)] = 0$ (Stein's identity);
KSD$(Q,P)$ uses only $\nabla\log p$ and $k$, closed-form over a sample.
**Diagrams:** none planned (text + one inline equation); optionally reuse
`fig_mdl-score-field`.
**Worked example(s):** verify Stein's identity for a 1-D Gaussian by hand.
**Exercises (draft):** (1) check Stein's identity for $\mathcal N(0,1)$ and a
linear $f$; (2) explain why KSD needs no samples from $P$.
**Prereqs / cross-refs:** §5.2.6 (RKHS/IPM), §5.2.8 (scores); SVGD references in
the main-book inference material.
:::

## The Divergence ↔ Generative-Objective Map
:label:`sec_mdl-divergence-objective-map`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The capstone table that justifies the whole section: read off,
for each modern generative model family, *which divergence its loss is secretly
minimizing* — turning a zoo of objectives into one organizing principle.
**Outline:** 1. assemble the master table (model $\to$ divergence $\to$ practical
consequence) · 2. walk three rows in detail (VAE, GAN, WGAN) tying back to the
relevant subsection · 3. close with the "choose your divergence, inherit its
failure modes" thesis (mode-dropping, vanishing gradients, blurriness).
**Key results to state (the table):**
VAE / ELBO $\to$ reverse KL (§5.2.4);
maximum likelihood / autoregressive / normalizing flows $\to$ forward KL (§5.2.4);
original GAN $\to$ Jensen–Shannon (§5.2.3);
f-GAN $\to$ chosen $f$-divergence (§5.2.3);
WGAN $\to$ Wasserstein-1 (§5.2.7);
MMD-GAN $\to$ MMD (§5.2.6);
diffusion / score-based $\to$ Fisher divergence (§5.2.8).
**Diagrams:** `fig_mdl-divergence-objective-map` — the master table rendered as a
labeled grid (rows = models, columns = divergence / projection direction /
failure mode).
**Worked example(s):** none new — this section *synthesizes*; it points each row
back to the worked example in its home subsection.
**Exercises (draft):** (1) classify a new objective (e.g., least-squares GAN) into
the table; (2) predict the failure mode of a reverse-KL vs. Wasserstein objective
on a multimodal target; (3) which rows give zero gradient on disjoint supports?
**Prereqs / cross-refs:** all of §5.2; forward to Ch6 (`sec_mdl-score-matching-diffusion-flow`);
main-book GAN / VAE / diffusion chapters.
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Bullet recap to be written once the body lands.
**Outline (planned bullets):** divergence = non-negative, zero iff equal, not
necessarily symmetric/metric · $f$-divergences (one convex generator, many
divergences) vs. integral probability metrics (one function class, sample-based)
vs. optimal transport (move mass) · forward vs. reverse KL is the practical
mode-covering/mode-seeking dial · Pinsker ties TV to KL · Wasserstein gives
gradients on disjoint supports (WGAN) · Fisher divergence drops the normalizer
(bridge to diffusion) · every generative loss is *some* divergence in disguise.
**Cross-refs:** §5.1, §5.3, Ch6.
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Recover KL and reverse-KL from the $f$-divergence template with generators
   $f(u)=u\log u$ and $f(u)=-\log u$.
2. Compute the convex conjugate $f^*$ for the $\chi^2$ generator and write down
   the explicit f-GAN critic objective.
3. Disjoint-support example: two point masses at distance $d$ — show
   Jensen–Shannon and KL are constant in $d$ while $W_1=d$ varies smoothly.
4. Prove the 1-D identity $W_1(P,Q)=\int|F_P(t)-F_Q(t)|\,dt$ and verify it against
   a sorted-sample computation in code.
5. Verify Pinsker's inequality numerically for two Gaussians using the closed-form
   KL `eq_mdl-gaussian_kl`, and show it is loose when the means are far apart.
**Cross-refs:** §5.2.2–§5.2.7.
:::

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/)
:end_tab:
