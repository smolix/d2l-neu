# Mutual Information and Representation Learning
:label:`sec_mdl-mutual-information`

::: {.callout-important title="Section status: detailed plan / ToC only — not yet written"}
This file is the **detailed outline** for §5.3. No prose, code, or figures have
been authored yet; every subsection below is a planning stub in the standard
format. The body framing for the section as a whole:

> *Self-supervised and contrastive methods (SimCLR, CPC, CLIP-style training)
> are, formally, **mutual-information maximization** — and InfoNCE is a* lower
> bound *on $I(X;Y)$. Knowing why MI is hard to estimate keeps you honest about
> what these objectives actually optimize.*

**Migration note.** §5.3.1–§5.3.2 below **absorb** the mutual-information
material currently living in §5.1 (`sec_mdl-information_theory`: joint /
conditional entropy, the MI definition `eq_mdl-mut_ent_def`, pointwise MI
`eq_mdl-pmi_def`, the Venn figure `fig_mdl-mutual_information`, and the
properties list). When this section is written, that content moves here and §5.1
keeps only a one-paragraph pointer.

**Prerequisites:** §5.1 (entropy, conditional entropy, KL, Gibbs);
§5.2 (`sec_mdl-divergences-distances`: KL as a building block, the f-GAN /
Donsker–Varadhan dual); 4.1 (joint/marginal densities, independence);
Ch3 (stochastic optimization, for the learned-critic bounds).
:::

## Mutual Information, Recapped
:label:`sec_mdl-mi-recap`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Re-introduce mutual information as the KL divergence between the
joint and the product of marginals — i.e., *how far two variables are from being
independent* — and migrate the §5.1 entropy-based identities here.
**Outline:** 1. $I(X;Y)=D_{\textrm{KL}}(P_{X,Y}\,\|\,P_X P_Y)$ (the divergence view,
connecting to §5.2) · 2. the entropy identities $I(X;Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)
=H(X)+H(Y)-H(X,Y)$ (migrated from §5.1) · 3. properties: symmetric, $\ge 0$
(Gibbs), $=0\iff$ independence · 4. pointwise MI $\textrm{pmi}(x,y)=\log\frac{p(x,y)}{p(x)p(y)}$
and one NLP example (replacing the dated "Amazon is on fire" anecdote).
**Key results to state:**
$I(X;Y)=D_{\textrm{KL}}(P_{X,Y}\|P_XP_Y)$;
$I(X;Y)=H(X)-H(X\mid Y)$;
$I(X;Y)\ge 0$ with equality iff $X\perp Y$;
$\textrm{pmi}(x,y)=\log\frac{p_{X,Y}(x,y)}{p_X(x)p_Y(y)}$.
**Diagrams:** `fig_mdl-mi-overlap` — extend the §5.1 entropy Venn diagram with
$I(X;Y)$ labeled as the overlap of $H(X)$ and $H(Y)$, $H(X\mid Y)$/$H(Y\mid X)$ as
the crescents, $H(X,Y)$ as the union.
**Worked example(s):** migrate the discrete `mutual_information` cell from §5.1;
add the closed-form Gaussian MI $I=-\tfrac12\log(1-\rho^2)$ as a continuous
ground-truth anchor.
**Exercises (draft):** (1) show $I(X;Y)=D_{\textrm{KL}}(P_{X,Y}\|P_XP_Y)$ from the
definition; (2) compute $I$ for a 2x2 joint pmf; (3) $I=-\tfrac12\log(1-\rho^2)$
for a bivariate Gaussian and its $\rho\to\pm1$ limit.
**Prereqs / cross-refs:** §5.1 (entropy chain rule, KL, Gibbs — source of migrated
material); §5.2.1 (MI as a KL/divergence instance); 4.1.9 (correlation).
:::

## Mutual Information as Nonlinear Correlation
:label:`sec_mdl-mi-nonlinear-correlation`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Correlation only sees *linear* dependence; mutual information
sees *any* dependence — and, unlike correlation, it is invariant under invertible
reparameterization of either variable.
**Outline:** 1. Pearson $\rho=0$ does not imply independence; MI $=0$ does ·
2. the $Y=X^2$ (or $Y=\cos X$) example where $\rho=0$ but $I>0$ · 3. invariance:
$I(X;Y)=I(g(X);h(Y))$ for invertible $g,h$ — callback to the §5.1
differential-entropy caveat (the offending Jacobians cancel) · 4. caveat that
this invariance is also *why* MI is hard to pin down empirically (next section).
**Key results to state:**
$\rho(X,Y)=0 \not\Rightarrow X\perp Y$, but $I(X;Y)=0\iff X\perp Y$;
$I(g(X);h(Y))=I(X;Y)$ for invertible $g,h$.
**Diagrams:** reuse `fig_mdl-mi-overlap`; optional small scatter inset showing
$\rho\approx 0$ yet visibly dependent data.
**Worked example(s):** $X\sim\mathcal N(0,1)$, $Y=X^2$ — show sample $\rho\approx0$
but estimated $I>0$; show MI unchanged under $X\mapsto 2X+3$.
**Exercises (draft):** (1) construct dependent variables with $\rho=0$;
(2) prove reparameterization invariance from the divergence form.
**Prereqs / cross-refs:** 4.1.9 (correlation, $X=Y^2$ counterexample);
§5.1.3 (reparameterization / differential-entropy caveat).
:::

## Why Mutual Information Is Hard in High Dimensions
:label:`sec_mdl-mi-hard`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The honesty section: there is *no free lunch* — certifying that
two high-dimensional variables share a lot of information requires exponentially
many samples, and every practical estimator is either badly biased or wildly
high-variance. This is why "MI maximization" objectives must be read as
*objectives*, not as MI meters.
**Outline:** 1. the statistical barrier: any distribution-free high-confidence
*lower* bound on MI from $N$ samples is itself $\le\log N$ (McAllester–Stratos) ·
2. consequence: a critic reporting "MI $=20$ nats" from a batch of 256 is
reporting at most $\log 256\approx 5.5$ nats of *certified* information · 3.
bias/variance tradeoff of the estimators (preview of §5.3.4–§5.3.5) · 4. why this
does not doom representation learning — the *objective* can still produce good
features even when the *number* is untrustworthy.
**Key results to state:**
any sample-based high-probability MI lower bound is $\le \log N$ (McAllester &
Stratos, 2020); bounded critic $\Rightarrow$ InfoNCE $\le \log N$.
**Diagrams:** none planned (text-heavy); optionally a small bias/variance
schematic deferred to §5.3.4.
**Worked example(s):** simulate two strongly-dependent Gaussians with true
$I\gg\log N$ and show every sample estimator saturates near $\log N$.
**Exercises (draft):** (1) state the $\log N$ ceiling and its consequence for
batch size; (2) explain why an *upper* bound on MI is easier than a lower bound.
**Prereqs / cross-refs:** §5.3.4–§5.3.5 (the bounds it warns about); §5.3.7
(criticisms); Ch3 (sampling/estimation, variance).
:::

## Variational Lower Bounds on Mutual Information
:label:`sec_mdl-mi-variational-bounds`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Since MI is intractable, we *lower-bound* it with a learned
auxiliary model and maximize the bound — exactly the f-GAN dual trick from §5.2,
now applied to the joint-vs-product KL. Three classic bounds, one framework
(Poole et al., 2019).
**Outline:** 1. **Barber–Agakov**: introduce a variational decoder $q(x\mid y)$;
$I(X;Y)\ge H(X)+E[\log q(x\mid y)]$ — tightest when $q=p(x\mid y)$ · 2.
**Donsker–Varadhan / MINE**: a learned critic $T$ gives
$I\ge E_{P_{XY}}[T]-\log E_{P_XP_Y}[e^{T}]$ — callback to the §5.2.3 dual of KL ·
3. **NWJ / f-bound**: the $f$-divergence-dual sibling, lower variance than DV ·
4. the bias/variance continuum unifying them (Poole et al.).
**Key results to state:**
Barber–Agakov: $I(X;Y)\ge H(X)+E_{P_{XY}}[\log q(x\mid y)]$;
Donsker–Varadhan: $I(X;Y)\ge \sup_T\{E_{P_{XY}}[T]-\log E_{P_XP_Y}[e^{T}]\}$ (MINE);
NWJ: $I(X;Y)\ge E_{P_{XY}}[T]-E_{P_XP_Y}[e^{T-1}]$.
**Diagrams:** `fig_mdl-mi-variational-bounds` — schematic of the bounds stacked
under the true MI, annotated with their bias/variance behavior as MI grows
(after Poole et al.).
**Worked example(s):** minimal MINE critic on two correlated Gaussians — tracks
the closed-form $I=-\tfrac12\log(1-\rho^2)$ at low MI, degrades (bias/variance) at
high MI.
**Exercises (draft):** (1) derive Barber–Agakov from $H(X\mid Y)=H(X)-I$ and
Gibbs; (2) show DV is the KL dual specialized to $P_{XY}$ vs $P_XP_Y$;
(3) explain why DV's $\log E[e^T]$ term causes high variance.
**Prereqs / cross-refs:** §5.2.3 (f-GAN / convex-conjugate dual — the same
machinery); §5.1 (KL, Gibbs); §5.3.3 (the $\log N$ ceiling these bounds hit).
:::

## InfoNCE and Contrastive Learning
:label:`sec_mdl-infonce`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The bound that powers modern self-supervised learning: InfoNCE
turns MI estimation into a *classification* problem — pick the one positive pair
out of a batch of $N$ — and the resulting loss is a lower bound on $I(X;Y)$ that
saturates at $\log N$, which is precisely why bigger batches help (CPC, SimCLR,
CLIP-style dual encoders).
**Outline:** 1. setup: one positive pair $(x,y^+)$ and $N-1$ negatives $y^-$;
a learned similarity/critic $f(x,y)$ scored through a softmax · 2. the InfoNCE
loss $\mathcal L_{\textrm{NCE}} = -E\big[\log\frac{e^{f(x,y^+)}}{\sum_j e^{f(x,y_j)}}\big]$
· 3. the bound $I(X;Y)\ge \log N - \mathcal L_{\textrm{NCE}}$ (van den Oord et al.,
CPC, 2018) · 4. the $\log N$ ceiling (callback to §5.3.3): the bound can never
exceed $\log N$, so batch size caps what you can certify · 5. CPC / SimCLR / CLIP
as instances; views/augmentations as the two "variables."
**Key results to state:**
$\mathcal L_{\textrm{NCE}}=-E\!\left[\log\frac{e^{f(x,y^+)}}{\sum_{j=1}^N e^{f(x,y_j)}}\right]$;
$I(X;Y)\ge \log N - \mathcal L_{\textrm{NCE}}$ (InfoNCE lower bound);
bound $\le \log N$ regardless of the critic.
**Diagrams:** `fig_mdl-infonce-pos-neg` — an anchor $x$ with one positive and
several negatives in embedding space, the softmax-over-similarities highlighted,
and the $\log N$ ceiling annotated.
**Worked example(s):** tiny InfoNCE on synthetic paired data — show the estimated
MI rises with batch size $N$ and saturates at $\log N$ even when true MI is larger.
**Exercises (draft):** (1) show $\mathcal L_{\textrm{NCE}}\ge 0$ and hence the
bound $\le\log N$; (2) relate InfoNCE to a categorical cross-entropy (callback to
§5.1); (3) why do larger batches tighten the bound?
**Prereqs / cross-refs:** §5.1 (softmax cross-entropy); §5.3.3 ($\log N$ ceiling);
§5.3.4 (InfoNCE as one member of the variational-bound family); main-book
contrastive / self-supervised chapters (SimCLR, CLIP).
:::

## The Information Bottleneck
:label:`sec_mdl-information-bottleneck`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A principled answer to "what makes a good representation":
compress the input as much as possible while keeping everything relevant to the
label — formalized as a tradeoff between two mutual informations, with a single
knob $\beta$.
**Outline:** 1. the objective $\min_{p(z\mid x)} I(X;Z) - \beta\, I(Y;Z)$ —
squeeze the representation $Z$ (low $I(X;Z)$) while staying predictive (high
$I(Y;Z)$) · 2. the $\beta$ knob sweeping the compression–prediction frontier ·
3. the deep variational IB (VIB) as a tractable bound (reuses §5.3.4 variational
MI bounds) · 4. **honest note** on the contested "compression phase" claim:
Shwartz-Ziv & Tishby (2017) reported a two-phase fit-then-compress dynamic, but
Saxe et al. (2018) showed it is largely an artifact of saturating (tanh)
activations and does not appear with ReLU, yet those nets still generalize — so
treat the compression-phase story as unsettled.
**Key results to state:**
IB objective $\min I(X;Z) - \beta\, I(Y;Z)$;
$\beta\to 0$ = maximal compression (uninformative $Z$), $\beta\to\infty$ =
maximally predictive $Z$;
deep VIB optimizes a variational bound on both terms.
**Diagrams:** `fig_mdl-ib-tradeoff` — the information plane: $I(Y;Z)$ (prediction)
vs $I(X;Z)$ (compression), with the optimal frontier swept by $\beta$ and a
marked operating point.
**Worked example(s):** sweep $\beta$ on a small VIB on a toy classification
problem; plot the resulting $(I(X;Z), I(Y;Z))$ points tracing the frontier.
**Exercises (draft):** (1) describe the optimal $Z$ as $\beta:0\to\infty$;
(2) relate the IB Lagrangian to a constrained "compress subject to accuracy"
problem (callback to Ch3 duality); (3) summarize the Saxe et al. critique in two
sentences.
**Prereqs / cross-refs:** §5.3.4 (variational bounds, for VIB); Ch3 (Lagrangian /
constrained optimization); main-book representation-learning chapters.
:::

## Limits and Criticisms of MI Estimation
:label:`sec_mdl-mi-limits`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A closing reality check: the MINE / InfoNCE estimates that
power these methods are loose and variance-heavy, and several papers argue that
the *quality of the learned critic / decoder* — not the tightness of the MI bound
— is what actually drives representation quality. Treat MI objectives as useful
training signals, not as trustworthy measurements.
**Outline:** 1. estimates are loose/high-variance (recap §5.3.3 + Poole et al.) ·
2. evidence that better representations can come from *worse* MI bounds (the bound
is a proxy) · 3. practical guidance: use MI objectives for the inductive bias they
encode, report them with skepticism · 4. pointer to ongoing debate (IB
compression phase, §5.3.6).
**Key results to state:** no new equations — synthesizes §5.3.3–§5.3.6;
restate the $\log N$ ceiling and the bias/variance tradeoff as caveats.
**Diagrams:** none planned.
**Worked example(s):** none new — references the saturating-estimator experiment
from §5.3.3 and the MINE example from §5.3.4.
**Exercises (draft):** (1) given a method that "maximizes MI," list two reasons
its reported MI may be untrustworthy; (2) design a sanity check using a
known-MI Gaussian pair.
**Prereqs / cross-refs:** §5.3.3 ($\log N$), §5.3.4 (bounds), §5.3.6 (IB debate).
:::

## Summary

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Outline (planned bullets):** $I(X;Y)=D_{\textrm{KL}}(P_{X,Y}\|P_XP_Y)$ measures
*any* dependence and is reparameterization-invariant · certifying large MI from
samples is statistically hard (the $\log N$ ceiling) · we therefore *lower-bound*
MI with learned critics/decoders (Barber–Agakov, MINE/DV, NWJ, InfoNCE) ·
InfoNCE is the contrastive-learning workhorse and saturates at $\log N$ · the
Information Bottleneck frames a good representation as a compression–prediction
tradeoff (with an unsettled "compression phase" debate) · read MI objectives as
training signals, not measurements.
**Cross-refs:** §5.1, §5.2; main-book self-supervised / contrastive chapters.
:::

## Exercises

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Draft exercise set (to be finalized with the body):**
1. Bivariate Gaussian MI $I=-\tfrac12\log(1-\rho^2)$; describe the $\rho\to\pm1$
   behavior and connect it to the $\log N$ ceiling.
2. Show InfoNCE is bounded: $I(X;Y)\ge\log N - \mathcal L_{\textrm{NCE}}$ and the
   right side is $\le\log N$.
3. Derive the Barber–Agakov bound from $I=H(X)-H(X\mid Y)$ and Gibbs'
   inequality (KL $\ge 0$).
4. For the Information Bottleneck, characterize the optimal $Z$ as $\beta$ sweeps
   $0\to\infty$.
**Cross-refs:** §5.3.1, §5.3.4–§5.3.6.
:::

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/)
:end_tab:
