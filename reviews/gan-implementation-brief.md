# GAN Chapter Rebuild — Implementation Brief

2026-08-02, revised same day after adversarial review. Companion to
`gan-chapter-review-and-proposal-2026-08-02.md`. This document specifies
each notebook closely enough that writing can proceed without re-deciding
structure. It still assumes the style guide's revision passes during
drafting; nothing here is final prose.

Conventions used below: "derive" = full derivation in the chapter;
"state+cite" = give the statement and a one-line reason, with either a
`:numref:` to the appendix section that proves it or a `:cite:` to the
literature; "PT/JAX" = one code cell per framework via `%%tab pytorch` /
`%%tab jax`. Cell IDs are assigned by `tools/add_cell_ids.py` after
drafting — never by hand. All appendix labels below were verified in
`reviews/gan-research/04-in-repo.md`. Exercises must be self-contained:
no pointers to the note or to research files; anything an exercise needs
is stated in the exercise.

---

## 0. Chapter-wide decisions

- **Files and section labels:** `index.md` (`chap_gans`), `gan.md` (16.1,
  `sec_basic_gan` — kept), `objectives.md` (16.2, new,
  `sec_gan_objectives`), `relativistic.md` (16.3, new,
  `sec_gan_relativistic`), `convergence.md` (16.4, new,
  `sec_gan_convergence`), `dcgan.md` (16.5, `sec_dcgan` — kept),
  `adversarial-losses.md` (16.6, new, `sec_gan_beyond`). Cross-references
  between chapter sections use these section labels; a writer never
  invents labels inside another writer's file — equation-level
  cross-section references are described verbally with a section
  `:numref:` and flagged in the writer's report for the integration pass.
- **Frameworks:** PyTorch + JAX (nnx). No TF/MXNet cells anywhere.
- **Notation:** the note's conventions. $p$ data, $q$ generator, critic $D$
  a *realness* logit, $\sigma(D(x))$ estimates $P(\text{real}\mid x)$,
  $\rho = p/q$, $\lambda = \log\rho$. Symbol discipline (style guide
  §9.10): $f$ is reserved for f-divergence generators (16.2); margin and
  pairing payoffs use $\ell$ (so the Dirac eigenvalues in 16.4 read
  $\pm i\,\ell'(0)$). Where R3GAN's paper convention differs (fakeness
  logit, payoff named $f$), 16.3 states the translation once. One notation
  block early in 16.1; every later section reuses it.
- **`#@save` surface:** 16.1 saves `update_D`, `update_G` (classic losses,
  PT/JAX). 16.4 saves `rpgan_loss_D`, `rpgan_loss_G`, `r1_r2_penalty`
  (PT/JAX). 16.5 consumes both sets; nothing outside ch. 16 consumes any of
  them (audited).
- **Figures:** new `tools/gen_mdl_gan_figures.py`, prefix `mdl-gan-`,
  imports the shared style. Four figures total (specs in §8); the legacy
  hand-drawn `img/gan.svg` is retired. Computed results (training curves,
  saturation curves, phase portraits, sample grids) stay as executed-cell
  plots.
- **Experimental prose rule:** quote only re-execution-stable facts —
  "covers all 25 modes / collapses to a handful", "the saturating loss makes
  no visible progress in the same budget" — never per-seed digits. Any
  numeric claim in prose must hold across three re-runs during the capture
  pilot.
- **Every section ends with `## Summary` (reconstructing the argument per
  style guide §5.6) followed by `## Exercises`.**
- **Decisions locked (Alex, 2026-08-02):** six sections; identity proved
  in-chapter; the note is not separately published (novel content lives in
  the book — the full-text novelty pass in §9.4 still runs); Pokemon with a
  sample-quality bar (pilot R6); in-notebook CIFAR-10 feature CNN with
  pretrained-backbone fallback (pilot R4 decides); file names as listed
  above. Writing is done by Fable agents, one section at a time; 4 GPUs
  available for pilots and execution; PyTorch+JAX only.

---

## 1. `index.md` — chapter opening

No code. Four paragraphs.

1. The problem, concretely: a sampler $x = G(z)$ with no density. Everything
   the book has trained so far was fit by maximum likelihood or a surrogate;
   here the quantity MLE needs does not exist. One sentence recalling
   $\text{MLE} = \arg\min_\theta D_{\mathrm{KL}}(\hat p \,\|\, p_\theta)$
   (state+cite `subsec_mdl-nll-crossentropy`).
2. What replaces it: compare samples from the model with samples from the
   data, and let the comparison itself be learned. Name the consequence up
   front: which comparison is chosen decides whether the generator receives
   a usable gradient.
3. Why this matters in 2026: adversarial losses ship inside one-step image
   generators, tokenizers, and vocoders even though stand-alone GANs are no
   longer the frontier; the instability that stalled them is now understood
   and fixable. (No numbers here; 16.6 carries the evidence.)
4. Roadmap by dependency: 16.1 one game analyzed exactly → 16.2 the space
   of games → 16.3 the modern objective → 16.4 the regularization that
   makes gradient descent find its optimum → 16.5 practice on images →
   16.6 where the loss lives now.

---

## 2. 16.1 `gan.md` — Generative Adversarial Networks

Question: what does the original GAN objective compute, and why does the
naive generator loss fail?

### Outline

`## From Likelihood to Comparison`
- MLE recap in two sentences, state+cite `subsec_mdl-nll-crossentropy`
  (MLE = forward KL) and `sec_mdl-latent-em-elbo` for the latent-variable
  route the book already knows.
- Implicit generator defined; why the density is unavailable (pushforward
  through a non-invertible map; one sentence, no measure theory).
- Likelihood ≠ sample quality: the mixture argument (derive, 3 lines):
  $0.01\,p_{\text{data}} + 0.99\,p_{\text{noise}}$ is within $\log 100$ of
  optimal log-likelihood while producing noise 99% of the time. Motivates
  judging the model by its samples.
- The two-sample view (kept from the current chapter, tightened): decide
  whether two sample sets came from one distribution; use the decision rule
  as a training signal.

`## The Log-Loss Game`
- Setup as a classification problem (label ~ Bernoulli(½), mixture marginal
  $m$), $V(D)$ defined; equation protocol per style guide.
- Derive $\sigma(D^\star) = p/(p+q)$, $D^\star = \log(p/q)$ pointwise.
  Note (used again in 16.3): this game pins $D^\star$ exactly, additive
  constant included.
- Derive $\max_D V = 2\,\mathrm{JS}(p,q) - 2\log 2$; JS definition
  state+cite `eq_mdl-js-def`. Two readings, each one paragraph: the
  entropy-Jensen-gap form (derive; the appendix does not have it) and the
  mutual-information form $\mathrm{JS}(p,q) = I(x;y)$ — drafting note:
  keep the value ($2\,\mathrm{JS} - 2\log 2$) and the divergence
  ($\mathrm{JS}$) typographically distinct; do not write "the value of the
  game is $I(x;y)$".
- Consequence: $0 \le \mathrm{JS} \le \log 2$, ceiling attained under
  disjoint supports (derive the bound; the pathology is computed in 16.2's
  separation experiment).

`## The Generator's Gradient`
- Saturating loss: gradient carries weight $\sigma(D(x'))$, which vanishes
  exactly where the generator is worst. Derive the two weights
  ($\sigma(D)$ vs. $\sigma(-D)$ — two lines each) and present the
  non-saturating loss as the same fixed point with the opposite weighting,
  not as folklore.

`## Fitting a Gaussian` (experiment)
- Same toy as today (2-D Gaussian via fixed $A, b$), rebuilt to verify the
  section's three results. Cells:
  1. PT/JAX imports (one cell each, chapter-standard).
  2. Data: $X = ZA + b$; scatter + true covariance printed (kept).
  3. Models: linear generator, 3-layer MLP critic (kept, PT/JAX).
  4. `update_D` / `update_G` `#@save` (classic losses; JAX version written
     so 16.5 can reuse it unchanged — resolve the current dcgan.md JAX
     redefinition by parameterizing the model arguments here, once).
  5. Train (non-saturating default); loss curves + sample overlay.
  6. **Verify the optimal critic:** both densities are Gaussian with known
     parameters ($q$ from the learned linear map), so $\log(p/q)$ is
     available in closed form; plot $D(x)$ against it on a grid; report the
     fit qualitatively ("the trained critic tracks the analytic log-ratio").
  7. **Quantify the fit:** multivariate Gaussian KL, stated in one display
     equation with a `:cite:` (the appendix's `eq_mdl-gaussian_kl` is
     univariate; the multivariate form is stated, not derived), computed at
     start and end of training.
  8. **Saturating vs. non-saturating A/B:** identical initialization, two
     short runs; sample overlays early and late + both loss traces. Prose
     states the observable difference, not numbers.
- Close: two point masses as the section's unresolved failure — the value is
  pinned at $\log 2$ and the gradient is zero; forward pointer to 16.2.

### Exercises (16.1)
1. Show the log-loss game identifies $D^\star$ exactly: adding a constant
   $c \ne 0$ to the optimal critic strictly decreases $V$. (Contrast
   planted for 16.3, where the pairing objective is shift-invariant.)
2. Unequal priors: with $P(y{=}1) = \alpha$, derive the optimal critic and
   show the value is the $\alpha$-skewed JS.
3. Compute the mixture-argument bound for mixture weight $\epsilon$; at
   what $\epsilon$ does the likelihood penalty exceed 1 nat?
4. Modify cell 6's grid comparison to report the sup-norm error of
   $\sigma(D)$ vs. $p/(p+q)$; where in the plane is it largest, and why?

### Slides (16.1): ~10
Cover; MLE-to-comparison (the mixture argument); the game (fig
`mdl-gan-architecture`); optimal-critic derivation (one slide, staged);
JS value + information reading; saturation weights; toy setup `@cell`;
ratio verification `@!cell`; A/B result `@!cell`; recap.

---

## 3. 16.2 `objectives.md` — Adversarial Objectives and Divergences

Question: what family of quantities can an adversarial game evaluate, and
which of them keep a gradient when supports separate?

### Outline

`## One Template, Two Choices`
- The template $d(p,q) = \sup_{T}\{E_p[a(T)] - E_q[b(T)]\}$ (note Eq. 1):
  choice of payoff, choice of critic class. Convexity in $(p,q)$ in one
  sentence. Figure `mdl-gan-template` (§8).

`## Proper Losses and Their Divergences`
- Margin objective with payoff $\ell$; Bayes-risk gap $\Delta_\ell$
  (derive, short); Proposition "every gap is an f-divergence" (derive —
  perspective argument, 6 lines; f-divergence definition state+cite
  `eq_mdl-f-div-def`).
- The loss→divergence table (logistic→JS, square→triangular, exponential→
  squared Hellinger, hinge→TV, 0–1→½TV), with two annotations: LSGAN's
  "Pearson χ²" is triangular discrimination up to scale (the mixture is the
  second argument — a reader trap worth one sentence); hinge→TV as stated
  here, with Lim & Ye cited for the max-margin view of the same loss.

`## f-Divergences from Duality`
- Variational bound state+cite `sec_mdl-f-gan-dual` / `eq_mdl-f-gan-bound`
  (proved in the appendix); what the chapter adds: the optimal critic
  $T^\star = f'(p/q)$ (derive via Fenchel–Young equality, 3 lines;
  conjugate state+cite `subsec_mdl-convex-conjugate`) and the reading that
  every unconstrained critic estimates the density ratio in some
  parameterization. Domain-of-$f^*$ → output activation, one paragraph
  (f-GAN recipe).
- Consistency check with 16.1 (the GAN row reduces to $V(D)$) as a worked
  two-line reconciliation.

`## Integral Probability Metrics`
- IPM definition state+cite `eq_mdl-ipm-def`; MMD closed form state+cite
  `eq_mdl-mmd2` — analytic *for a fixed kernel*, so the game needs no inner
  optimization; scope it in the same breath (a learned feature space
  reopens the choice — forward pointer to KID in 16.5). The $O(k^2)$ batch
  cost noted (recurs at 16.5's KID).
- $W_1$: KR duality state+cite `eq_mdl-kr-dual`; Lipschitz enforcement in
  one paragraph (clipping/penalty/spectral norm, citing the appendix's own
  WGAN passage); 1-D closed form state+cite `eq_mdl-w1-cdf`. The Gaussian
  $W_2$ (Bures–Wasserstein) is *deferred to 16.5* where it becomes FID.

`## Which Objectives Give Gradients` (experiment)
- The dichotomy, then compute it. Cells:
  1. Imports.
  2. **Separation experiment (analytic):** two unit-width Gaussians
     distance $d$ apart — the point-mass example at computable width.
     JS by fixed-grid quadrature (deterministic, capture-stable), $W_1 =
     |d|$ (1-D formula), MMD² (RBF, analytic). `d2l.plot` all three vs.
     $d$: JS saturates at $\log 2$, the other two keep slope.
  3. **One testbed, four losses:** 2-D mixture target; same
     generator/critic sizes; train under non-saturating log-loss, LSGAN,
     hinge, and MMD (no critic). What the grid demonstrates is stated up
     front: all four share the fixed point $q = p$, so with overlapping
     supports the final fits should agree while the training dynamics
     differ; the per-run diagnostic (each critic's implied density-ratio
     estimate against the analytic ratio) is what distinguishes them.
  4. **Ratio recovery:** train the KL-row f-GAN critic on a pair of known
     densities; plot $T(x)$ against $1 + \log(p/q)$ (its analytic optimum).
- Estimation rates paragraph: MMD at the parametric rate, $W_1$ at
  $n^{-1/d}$, ratio estimation hardest where the ratio is extreme —
  state+`:cite:` (Sriperumbudur et al.; Weed–Bach); no appendix target
  exists, no experiment.
- Summary cites `sec_mdl-divergence-objective-map` as the map of
  objective→divergence rather than rebuilding the table.

### Exercises (16.2)
1. Derive the Bayes-risk gap for the Brier loss from $L(\eta)=\eta(1-\eta)$.
2. Show $\Delta_{0\text{–}1} = \tfrac12 \mathrm{TV}$ and reconcile with the
   hinge row.
3. Verify LSGAN ≡ triangular discrimination numerically on random discrete
   distributions (state both normalizations in the exercise; the ratio is
   exactly 8).
4. In the separation experiment, replace the RBF kernel with a linear one;
   which failure reappears, and why (characteristic kernels)?
5. (Harder) From `eq_mdl-f-gan-bound`, derive the output activation needed
   for the JS generator and recover $V(D)$ exactly.

### Slides (16.2): ~10
Template; two knobs; loss→divergence table (staged); $T^\star=f'(p/q)$;
IPMs — MMD analytic, $W_1$ dual; separation plot `@!cell`; four-loss grid
`@!cell`; rates; objective map (cite); recap.

---

## 4. 16.3 `relativistic.md` — Relativistic Objectives

Question: what changes when the critic scores pairs instead of samples?

### Outline

`## Scoring Pairs`
- $\Phi(D) = E_{x\sim p, y\sim q}[\log\sigma(D(x)-D(y))]$; Bradley–Terry
  reading (state+cite `eq_bradley_terry` in ch. 15; one sentence on reward
  models and on preference-Elo leaderboards using the same model). Figure
  `mdl-gan-pairing` (§8).
- Two symmetries, each with its consequence: additive shift of $D$ leaves
  $\Phi$ unchanged (contrast with 16.1 exercise 1; R3GAN's equilibrium
  condition "any constant $C$"), and $\Phi$ depends on $p \otimes q$ —
  quadratic in the pair, outside 16.2's template, so none of 16.2's
  conclusions transfer automatically.

`## The Value of the Pairing Game`
- Optimal critic $D^\star = \log(p/q)$ up to a constant (derive; the
  variational stationarity computation, compressed).
- The lifting argument and the closed form, proved in full (four lines
  given 16.1: the product-space optimal critic is
  $\log\frac{p\otimes q}{q\otimes p} = \lambda(x) - \lambda(y)$, itself a
  difference critic; the swap symmetry halves the objective; 16.1's value
  formula applies on $\mathcal X \times \mathcal X$):
  $d_{\mathrm{Rp}}(p,q) = \mathrm{JS}(p\otimes q,\ q\otimes p)$, with the
  entropy form and the pair reading (a randomly ordered pair carries
  $d_{\mathrm{Rp}}$ nats about which member is real). Presentation: state
  that we have not found this identity in the literature; cite
  Jolicoeur-Martineau 2020 for the divergence property and PairGAN for the
  nearest construction.
- Properties, each with its one-line proof or cite: JS ≤ d_Rp ≤ log 2
  (marginalization + data processing), locally $2\times$JS, saturates under
  disjoint supports — so pairing does *not* fix 16.1's failure; that is
  16.4's job. This dependency is the section boundary.

`## Ranking and Mode Coverage`
- The generator's per-sample weight: rank statistic
  $E_x[\sigma(D(y)-D(x))]$ vs. the classical threshold $\sigma(D(y))$;
  why a single decision boundary suffices to please the classical weight
  (mode dropping) and cannot satisfy the rank weight. Sun et al. stated:
  the classical landscape has exponentially many mode-dropping basins, the
  relativistic one does not (cite, no proof).
- Saturating vs. non-saturating pairing: the paper states the zero-sum
  form; the reference implementation trains the generator on the
  non-saturating form. Work out the two weights (they mirror 16.1's) and
  state plainly that implementing the paper's equation literally gives the
  weaker variant. One tight paragraph plus the two-row weight table.

`## Verifying the Closed Form` (experiment)
  1. Imports.
  2. **Finite-sample-space verification:** 5 atoms; maximize $\Phi(D)$ over
     the 5 critic values with a few hundred optimizer steps; compare
     against $\mathrm{JS}(p\otimes q, q\otimes p)$ computed directly, and
     the recovered $D$ against $\log(p/q)$ up to a constant. Seconds of
     CPU; the section's theorem checked to many digits.
  3. **Weight comparison:** on the same atoms, plot the rank weight and the
     threshold weight as functions of the critic value — the mode-coverage
     argument made visible without training anything.

### Exercises (16.3)
1. Prove the additive-shift invariance of $\Phi$ and conclude the critic is
   identified only up to a constant. Why does the same argument fail for
   16.1's $V$?
2. Unequal ordering priors: if the real sample sits first with probability
   $\alpha \ne \tfrac12$, derive the value of the pairing game.
3. Derive the local expansion $d_{\mathrm{Rp}} = 2\,\mathrm{JS} +
   O(\epsilon^3)$ for $q = p(1+\epsilon h)$.
4. Extend the pairing game to $K$ negatives; show the value is the
   generalized JS of the $K{+}1$ product measures with ceiling
   $\log(K{+}1)$, and connect to InfoNCE (state+cite `sec_mdl-infonce`).
5. (Harder) Show that restricting the pair critic to differences
   $D(x)-D(y)$ loses nothing for the logistic loss, and exhibit where the
   argument uses the product structure of $p\otimes q$.

### Slides (16.3): ~9
Pairing objective + Bradley–Terry; the two symmetries; optimal critic;
lifting → closed form (staged, two slides); properties table; rank vs.
threshold weight; verification `@!cell`; saturating-vs-code lesson; recap.

---

## 5. 16.4 `convergence.md` — Gradient Penalties and Convergence

Question: why does gradient descent on a correct objective still fail, and
what restores convergence?

### Outline

`## What the Objective Cannot Buy`
- Opening dependency: 16.3 fixed the landscape's basins but not the
  disjoint-support saturation, and neither section said anything about the
  *dynamics* of alternating gradient descent. Both failures are visible on
  the smallest example.

`## The Dirac-GAN`
- $p = \delta_0$, $q_\theta = \delta_\theta$, linear critic
  $D_\psi(x) = \psi x$ (derive everything; 6 lines with 16.1's machinery).
  Continuous gradient flow: eigenvalues $\pm i\,\ell'(0)$, exact circles.
  Discrete simultaneous gradient descent: the same game spirals *outward* —
  discretization alone turns a neutral center into divergence. (This is
  the experiment's second panel, and the reason "just descend" fails even
  on the toy.)

`## Zero-Centered Penalties`
- $R_1 + R_2 = \gamma E_m \|\nabla_x D\|^2$ (derive, one line) — the same
  mixture $m$ as 16.1's game.
- Dirac-GAN with either penalty: eigenvalues
  $-\gamma/2 \pm \sqrt{\gamma^2/4 - \ell'(0)^2}$; local convergence for
  every $\gamma > 0$; critically damped at $\gamma = 2|\ell'(0)|$.
  Footnote: the R3GAN paper prints this formula without the square;
  confirmed a typo against Mescheder et al. and the paper's own Jacobian
  (evidence in `gan-research/02-literature.md` §1).
- What the penalty measures near equilibrium: the penalized game value is a
  squared dual Sobolev norm of $p-q$, i.e. linearized $W_2$ (statement +
  one-paragraph reading; $W_2$/Benamou–Brenier state+cite
  `eq_mdl-benamou-brenier`, `eq_mdl-w2`). One-centered vs. zero-centered
  table: the WGAN-GP penalty keeps a nonzero slope at $p=q$ and fails the
  Dirac test (cite Mescheder); that is the whole treatment WGAN-GP gets.

`## When One Penalty Is Not Enough`
- Scoped, not universal: R3GAN's ablation found $R_1$-alone diverging on
  StackedMNIST and 2-D toys (both objectives, γ swept 0.1–100), while
  StyleGAN2 trained with $R_1$ alone at FFHQ scale. The mechanism: near
  equilibrium either penalty suffices; far from it, smoothing only $p$
  leaves the critic free to steepen on $q$ (Roth's smoothing reading,
  compressed).

`## The R3GAN Recipe` (experiment)
- The loss in code, `#@save`, PT/JAX (~15 lines): `rpgan_loss_D`,
  `rpgan_loss_G` (non-saturating, matching the reference implementation
  and 16.3's lesson), `r1_r2_penalty`. JAX note: per-sample input
  gradients via `jax.vmap(jax.grad(...))`; PT via `autograd.grad` on the
  batch sum (the reference implementation's shape).
- The six principles (convergent objective + penalties; small LR, no
  momentum; no normalization layers; bilinear resampling; leaky ReLU; the
  backbone) and the trick-removal roadmap A→E summarized in one table;
  StackedMNIST ablation quoted (1000/1000 vs. 693; both objectives fail
  with $R_1$ alone); the compute footnote (their real budgets) quoted,
  setting up 16.5's scale discussion.
- Cells:
  1. Imports.
  2. Loss functions (`#@save`).
  3. **Dirac-GAN trajectories:** three panels — integrated ODE (circles),
     discrete simultaneous GD (outward spiral), discrete GD with penalty
     (contraction). Computed plot.
  4. **Mode-coverage A/B** (configuration fixed by pilot, 2026-08-02;
     data in `scratchpad/gan-pilots/coverage/`): 25-Gaussian grid, G
     latent-64 MLP(256,256), D MLP(256,256), γ = 1.0, Adam(β₁=0,
     β₂=0.99, lr 2e-4), batch 256, 20k steps; three configs — classic
     non-saturating GAN (no penalty), classic + $R_1{+}R_2$, RpGAN +
     $R_1{+}R_2$. Final-sample panels + covered-mode count (≥1 sample
     within 3σ). The claims the pilot supports on every seed (and the
     only ones the prose may make): the plain GAN drops several of the
     25 modes and is markedly less uniform; both penalized configs
     reliably cover all 25, with no measurable difference between them
     at this scale. The RpGAN-vs-GAN coverage gap under penalties is
     cited to R3GAN's StackedMNIST result (693 vs. 1000 modes), not
     demonstrated here — the section says why (the gap needs mode
     counts and capacity pressure a 2-D toy does not have). One more
     phrasing constraint from the pilot: no toy config diverges
     numerically, so "instability" language belongs to the Dirac-GAN
     analysis; the 25-Gaussians experiment teaches mode dropping.
     ~2–3 min/config on one GPU.

### Exercises (16.4)
1. Compute the Jacobian of the Dirac-GAN's simultaneous-GD map and show
   its spectral radius exceeds 1 for every step size when $\gamma = 0$.
2. Show $R_1 + R_2 = \gamma E_m\|\nabla D\|^2$ and explain why the mixture
   weighting matters (compare with weighting by $p$ alone).
3. Solve the penalized quadratic game of the linearization exactly and
   verify the $\gamma$-scaling of the value.
4. Re-run cell 4 with $\gamma$ 10× larger and smaller; which failure mode
   returns in each direction?
5. Sweep the step size in cell 3's discrete panel; does any step size
   stabilize the unpenalized game?

### Slides (16.4): ~10
What 16.3 left open; Dirac-GAN setup; flow vs. discrete (staged); penalties
+ eigenvalues (typo footnote); linearized-$W_2$ table (one- vs.
zero-centered); when one penalty is not enough; loss code `@cell`; phase
portraits `@!cell`; mode-coverage panels `@!cell`; StackedMNIST + roadmap
tables (quoted); recap.

---

## 6. 16.5 `dcgan.md` — Adversarial Image Generation

Question: what does the modern objective change in practice, and how is
sample quality measured?

### Outline

`## The 2015 Recipe` (compressed DCGAN)
- What DCGAN fixed (an architecture class in which the classic loss trains
  at all) and what it could not (the objective). Keep `G_block`/`D_block`
  with BN and the shape arithmetic (tightened; transposed-conv details
  remain state+cite `sec_transposed_conv`, `sec_fcn`). Pokemon data
  pipeline (PT/JAX only). Train briefly with the classic loss: the
  historical baseline and the first image result.

`## A Modern Minimal Backbone`
- R3GAN's principles scaled to the toy: bilinear resampling + conv instead
  of strided/transposed conv; leaky ReLU in both networks; no
  normalization layers (which also removes the BatchNorm special-casing
  that currently forces a JAX-specific `update_D`); Adam with
  $\beta_1 = 0$; small learning rate; horizontal-flip augmentation (one
  sentence on why small datasets need it, citing ADA); EMA of generator
  weights (state+cite the EMA implementation in `training-recipes`).
  Explicitly a reduced instance, not R3GAN's Config E; grouped-conv/
  inverted-bottleneck noted as the paper's full version and skipped.

`## Loss A/B on One Backbone` (experiment)
- Identical modern backbone; classic non-saturating loss vs.
  RpGAN + $R_1{+}R_2$ (16.4's `#@save` functions). Sample grids at fixed
  epochs + loss traces. γ is dataset-dependent (R3GAN sweeps 0.05–150
  across its benchmarks); the chapter picks one by pilot and says why a
  single number is not portable.

`## Measuring Sample Quality`
- The metrics reuse 16.2's two analytic cases, now in feature space:
  FID = $W_2^2$ between Gaussians — Bures–Wasserstein formula stated, with
  the commuting-covariance check derived in two lines and a `:cite:`
  (Dowson–Landau; Givens–Shortt) for the general case; KID = the unbiased
  MMD U-statistic (state+cite `eq_mdl-mmd2`) over *learned* features,
  which reopens the kernel choice 16.2 fixed — the scoping sentence
  planted there pays off here.
- Feature network (decision 5 in the proposal): a small CNN classifier
  trained in-notebook on CIFAR-10 (PT/JAX, ~2 min), penultimate features
  scoring the sprite runs. Off-domain by construction — the same
  structural situation as Inception-on-ImageNet scoring faces — and the
  fragility paragraph uses exactly that: feature dependence,
  ImageNet-class leakage (cite Kynkäänniemi; R3GAN reports its numbers
  without ImageNet-pretrained discriminators for this reason), the
  aliased-resizing pitfall (cite Parmar), finite-sample bias in one
  sentence. Precision/recall in one paragraph as the fidelity/coverage
  decomposition R3GAN reports.
- Compute FID and KID for both A/B runs; prose claims the *ordering*, not
  the values (risk R4).

`## Conditioning, Scale, and Scope`
- `### Conditional Generation` (sketch, no training cell — per Alex's
  review): a condition $c$ (class label, text embedding) is given to both
  networks, $G(z, c)$ and $D(x, c)$; the game becomes distribution matching
  per condition, and the optimal critic is the conditional log-ratio
  $\log\big(p(x\mid c)/q(x\mid c)\big)$ (two-line derivation deferred to
  exercise 5). The two standard mechanisms in one sentence each:
  concatenate $c$ into the inputs (`:cite:` Mirza & Osindero 2014), or the
  projection discriminator (inner product of a class embedding with the
  critic's features, `:cite:` Miyato & Koyama 2018). Observation that ties
  it to the rest of the chapter: R3GAN's CIFAR-10 and ImageNet results are
  class-conditional, and every text-to-image system in 16.6 conditions on
  prompts — the objective is unchanged, only the networks' inputs grow.
- `### Scale` — R3GAN's real budgets quoted (StackedMNIST 7 h on 8×L40 →
  FFHQ-256 weeks on 8×A6000); what changes at scale (resolution-dependent
  γ, BF16-not-FP16, EMA half-life, augmentation probability); pointer to
  the Image Models part for image applications and to ch. 17 for the
  successor model family.

### Exercises (16.5)
1. Derive Gaussian $W_2^2$ in 1-D from the quantile formula; check the
   commuting multivariate case against the Bures–Wasserstein formula.
2. Compute FID between the dataset and itself under subsampling; how large
   is the finite-sample bias at $n = 100, 1000$?
3. Swap the feature network for one trained on Fashion-MNIST; do the A/B
   orderings move? (Feature dependence, hands-on.)
4. Apply the 16.4 loss to Fashion-MNIST (carried over from the current
   chapter, now with a modern recipe).
5. Derive the optimal critic of the conditional game and show the value is
   $E_{c}\big[2\,\mathrm{JS}(p(\cdot\mid c),\, q(\cdot\mid c))\big] -
   2\log 2$; conclude that conditioning changes the networks, not the
   analysis.

### Slides (16.5): ~10
2015 recipe; what it couldn't fix; modern backbone principles; A/B grids
`@!cell`; loss traces `@!cell`; FID = Gaussian $W_2$; KID = MMD with
learned features; metric fragility; scale table (quoted); recap + forward
pointers.

---

## 7. 16.6 `adversarial-losses.md` — Adversarial Losses Beyond GANs

Question: where does the adversarial objective survive in 2026, and why
there? Prose-forward closing section with one demonstration cell.

`## The Capacity Argument` (with the section's one experiment)
- A bounded student under pointwise MSE averages over what it cannot
  represent and blurs; a critic penalizes leaving the data manifold.
  Demonstrated in ~20 lines: 1-D two-mode conditional target; an MSE
  student predicts the conditional mean (between the modes), an
  adversarially trained student commits to modes. PT/JAX, CPU-seconds.
- SDXL-Lightning's patch-FID observation summarized: the adversarial term
  buys local detail, not global layout.

`## Distillation`
- ADD, organized around its loss ablation (distillation-only FID 315.6 vs.
  adversarial-only 20.8 — the section's central number); DMD2's different
  motive (train on real data, escape the teacher's ceiling); LADD (the
  teacher as discriminator); FLUX.1-schnell as the production citation.
  Counterpoint with equal weight: MeanFlow and sCM reach one/two-step
  quality with no discriminator; the term is useful, not necessary.

`## Tokenizers, Audio, Video`
- VQGAN's patch discriminator as the standard tokenizer recipe and the
  2026 reversal (ViTok-v2 drops it at 5B for stability under scaling —
  the 2017 criticism, still live); vocoders, where adversarial training
  remains standard practice (RAF porting the relativistic loss to
  waveforms); real-time video via adversarial post-training.

`## Three Exits from Instability`
- The de-heuristicization map: (1) fix the game — convergence-analyzed
  objectives and penalties (Arjovsky–Bottou diagnosis → Mescheder → Sun →
  R3GAN); (2) constrain the critic — spectral norm as a structural
  constraint, then frozen pretrained features (Projected GAN →
  StyleGAN-T/GigaGAN → ADD's DINOv2 heads) plus feedback-controlled
  augmentation (ADA); (3) remove the game — MMD/moment matching, IMLE,
  sliced/entropic OT, and at the field level diffusion/flow as regression.
  Each exit with its cost, one sentence. Mapped onto the three axes (what
  the critic estimates / does / is constrained by). Figure `mdl-gan-exits`
  (§8).
- Exit 3's closing observation (from the literature addendum, `02-literature.md`
  §8): the game was removed three separate times and returned three times —
  GMMN's fixed kernel failed on CIFAR-10 and came back as MMD-GAN with a
  learned kernel; the sliced-Wasserstein generator paper itself added a
  discriminator to pick projection directions; diffusion removed the game
  from training and brought the discriminator back at distillation. The
  pattern, stated once without ornament: when a fixed discrepancy is too
  weak in high dimension, the recurring fix is to learn the critic. (Two
  supporting numbers if drafting wants them: GMMN's raw-pixel variant loses
  to contemporary GANs; the Sinkhorn generator's own ablation worsens as
  regularization approaches true OT — better-estimable beats theoretically
  stronger.) Citation cautions from the addendum: IMLE is arXiv-only (no
  venue attribution); "Adaptive IMLE" (AAAI 2023) is an unrelated method.

`## The Discriminator Inside Likelihood Models`
- DDO: a likelihood model read as an implicit discriminator through the
  same identity $D^\star = \log(p/q)$ that opened the chapter; MLE and
  adversarial training as two uses of one object, with the two-player
  optimization as the part that proved optional. Handoff to ch. 17 (its
  appendix foundations already exist; state+cite
  `sec_mdl-score-matching-diffusion-flow`).

### Exercises (16.6)
1. From the capacity argument, predict which of (a) super-resolution,
   (b) class-conditional 32×32 generation, (c) text rendering benefits most
   from an adversarial term; check against the section's evidence.
2. Show that DDO's implicit discriminator is Bayes-optimal when the
   likelihood model equals the data law (three lines from 16.1).
3. Modify the capacity-argument cell to three modes; does the MSE student's
   failure change qualitatively?
4. For each of the three exits, name the cost in one sentence.

### Slides (16.6): ~9
Capacity argument + demo `@!cell`; ADD ablation table; distillation family
tree; tokenizer/audio/video; GAN-free counterpoint; three exits (fig);
DDO; where ch. 17 picks up; recap.

---

## 8. Figures (`tools/gen_mdl_gan_figures.py`, prefix `mdl-gan-`)

| id | Content | Replaces |
|---|---|---|
| `mdl-gan-architecture` | Generator/discriminator loop: $z \to G \to$ fake batch vs. real batch $\to D \to$ realness logit, gradients flowing back both ways. House style. | legacy `img/gan.svg` |
| `mdl-gan-template` | The template's two knobs: payoff $(a,b)$ axis and critic-class axis, with GAN/f-GAN, MMD, $W_1$ placed; RpGAN outside the plane (quadratic). | — |
| `mdl-gan-pairing` | Single-sample classification vs. pair ranking, side by side: $D(x) \gtrless 0$ vs. $D(x) - D(y)$; the additive constant visibly canceling in the pair. | — |
| `mdl-gan-exits` | Three exits from instability as a branching schematic, each with 2–3 named methods and its cost. | — |

All other visuals are executed-cell outputs. Render-and-inspect loop per
CLAUDE.md before any figure is declared done; byte-idempotent generator.

---

## 9. Infrastructure checklist

Ordered; items 1–4 are prerequisites for drafting, the rest land with it.

1. `CHAPTER_NUMBERING` (`tools/d2l_preprocess.py`): `gan.md → [16,1]`,
   `objectives.md → [16,2]`, `relativistic.md → [16,3]`, `convergence.md →
   [16,4]`, `dcgan.md → [16,5]`, `adversarial-losses.md → [16,6]`; matching
   `_quarto.yml` order (dict order = book order; PDF pairing depends on
   it).
2. Incidental fixes riding along: `chapter_optimization/practice.md` line
   730 "Diffusion (ch. 15)" → `:numref:`chap_diffusion``; CLAUDE.md part-
   structure bullet updated for the RL split (RL 14, Deep RL 15, GANs 16,
   Diffusion 17, downstream shifted).
3. **Bibliography:** none of the chapter's post-2016 citations exist in
   `d2l.bib`. Add: Jolicoeur-Martineau 2019 + 2020, Huang et al. 2024
   (use the arXiv/proceedings title "A Modern GAN Baseline"), Mescheder
   2018, Sun 2020, Nowozin 2016, Arjovsky 2017 (+Bottou), Gulrajani 2017,
   Miyato 2018, Lim & Ye 2017, Mao 2017, Roth 2017, Heusel 2017, Bińkowski
   2018, Kynkäänniemi 2019 + 2023, Parmar 2022, Chong & Forsyth 2020,
   Sauer 2023 (ADD), Yin 2024 (DMD2), Zheng 2025 (DDO), Tong 2020
   (PairGAN), Dowson–Landau / Givens–Shortt, Metz 2017 (StackedMNIST),
   Sriperumbudur, Weed–Bach, and the exits-section set: Nagarajan &
   Kolter 2017, Li et al. 2015 (GMMN), Li et al. 2017 (MMD-GAN), Genevay
   2018 (Sinkhorn), Deshpande 2018 (sliced-W), Li & Malik 2018 (IMLE,
   arXiv-only — no venue), Sauer 2021 (Projected GAN), Kumari 2022,
   Karras 2020 (ADA), Zhao 2020 (DiffAugment), Mirza & Osindero 2014
   (conditional GAN), Miyato & Koyama 2018 (projection discriminator).
4. **Novelty follow-up (before the chapter prints the priority claim):**
   a full-text literature pass for the d_Rp identity (Google Scholar
   full-text / arXiv full-text) beyond the exhausted-budget search already
   done; record the outcome in the note's footnote. Related open decision
   (proposal §4.3): arXiv-first for the note.
5. `tools/gen_mdl_gan_figures.py` + `make figures` stamp entry; retire
   `img/gan.svg` references.
6. All cells `%%tab pytorch` / `%%tab jax`; no `#@tab` four-way tags remain
   in the chapter.
7. `tools/add_cell_ids.py` after drafting; `tools/lint_source.py` clean.
8. **Transitional constraint (review finding, 2026-08-02):** after 16.1's
   `#@save` narrowing to PT/JAX, the old four-framework `dcgan.md` still
   calls `d2l.update_D`/`update_G` from TF/MXNet tabs — `make lib` +
   TF/MXNet notebook regeneration is broken until the 16.5 rewrite lands.
   The chapter lands as one change set; do not run lib or tf/mxnet
   notebook targets in between.
9. Delete `outputs/{tensorflow,mxnet}/chapter_generative-adversarial-networks/`;
   regenerate + execute pytorch/jax through the scheduler
   (`make run-notebooks-<fw>` or targeted stamps); capture via
   `capture_outputs.py --frameworks pytorch,jax`; `make audit-outputs`
   clean before render. Capture only after "scheduler done: 0 failed".
10. Slides: six `<!-- slides -->` sections per the outlines above;
   `tools/audit_slides.py` clean; decks render for pytorch+jax only.
11. PDF tripwires: no `$`-digit adjacency, `\left(\begin{smallmatrix}`
    forms for parenthesized small matrices, no `]` in captions; `make pdfs`
    green.
12. **Final acceptance gate (Alex, 2026-08-02): before the chapter is
    declared done, the orchestrator reads all six sections end-to-end,
    in order, and verifies (a) logical coherence as one chapter — the
    argument builds monotonically, notation and voice are uniform, each
    section consumes what its predecessor established; (b) the reference
    graph points backwards — to earlier chapters and the Mathematics for
    Deep Learning part — with forward references limited to the
    deliberate hand-offs (section-closing dependencies, ch. 17, Image
    Models). A student must be able to read linearly. Findings are fixed
    before execution/capture, not after.**
13. Note corrections (Alex's `gan-notes/` draft — **pending Alex's
    approval**, listed in the proposal §3.5): Eq. 12 footnote upgrade,
    non-saturating-code remark, TV↔hinge attribution, PairGAN citation.

## 10. Compute budget

| Notebook | Configs trained | Est. wall-clock per framework |
|---|---|---|
| 16.1 | 3 short toy runs (main + A/B pair) | < 2 min |
| 16.2 | 4 toy losses + 1 critic fit + analytic plots | ~5 min |
| 16.3 | finite-space verification (CPU-seconds) | < 1 min |
| 16.4 | Dirac sim + 3 mode-coverage configs | ~7 min |
| 16.5 | DCGAN baseline + 2 A/B runs + feature net + metrics | ~12 min |
| 16.6 | 2 tiny regression students | < 1 min |

Nominal total ~27 min per framework; ceiling ~60 min if risk R1 forces the
StackedMNIST fallback. Nothing needs the heavy/multi-GPU scheduler tiers.

## 11. Risks and pilots (resolve before prose is written)

- **R1 — mode-coverage separation at toy scale. RESOLVED (pilot,
  2026-08-02; 52 runs in `scratchpad/gan-pilots/coverage/`).** The
  anticipated failure occurred: no stable RpGAN-vs-GAN gap under
  penalties at K=25 (both ceiling at 25/25) or K=64 (overlapping
  ranges; a real but non-claim-grade trend with a shrunken generator).
  The plain-GAN vs. penalized separation is solid on every seed. The
  16.4 experiment and its permissible prose claims are now fixed in §5
  cell 4; the StackedMNIST fallback is not used (it would triple the
  budget to demonstrate a result the citation carries).
- **R2 — γ portability. RESOLVED (pilot, 2026-08-02;
  `scratchpad/gan-pilots/images/PILOT_NOTES.md`).** γ = 0.1 mode-collapses;
  γ ∈ {1, 10, 100} is a stable plateau. Chapter uses **γ = 10** (plateau
  center), confirmed on a second seed. The 16.5 configuration is fixed in
  the pilot's RECOMMENDATIONS section (RpGAN + R₁R₂, 15000 steps,
  Adam(0, 0.99) lr 2e-4, batch 64, EMA half-life 500, flip augmentation,
  z-injection per the notes) and is binding for the writer. Timing note
  superseding R3's extrapolation: at this model size JAX is ~4× FASTER
  per step than PT (scale-dependent; the R3 overhead finding applied to
  its smaller conv probe) — framework-timing prose stays qualitative.
- **R3 — JAX input-gradient penalties. RESOLVED (pilot, 2026-08-02;
  code + verification in `scratchpad/gan-pilots/penalty/`).** Signature:
  `r1_r2_penalty(critic, real, fake) -> (r1, r2)`, per-sample unscaled
  squared gradient norms; caller applies `(gamma/2)*(r1+r2).mean()`.
  PT: `autograd.grad` on the batch sum; JAX: `jax.vmap(jax.grad(...))`
  with the nnx module closed over — composes with `nnx.jit` and
  second-order parameter gradients, verified (linear exact case,
  finite differences, cross-framework to 1.6e-5). Consequence for 16.5:
  penalty overhead on conv critics at 64×64 is ~2–3× in PT but ~9–10× in
  JAX, so the JAX image A/B runs several times longer than the PT one —
  fine at Pokemon scale; the 16.5 writer should set expectations in the
  timing text and must NOT introduce lazy regularization to compensate
  (R3GAN's appendix rejects it: convergence failure on toys).
- **R4 — feature-metric stability. RESOLVED (pilot, 2026-08-02).** FID and
  KID rank the penalized-relativistic run above the collapsed classic run
  unanimously across 3 feature-CNN seeds and both n = 500/2000; the
  real-vs-real floor sits two orders of magnitude below the gap. The
  in-notebook CIFAR-10 feature CNN stands (Alex's decision 5); no
  pretrained fallback needed. Prose quotes orderings, not values.
- **R5 — quadrature determinism in 16.2.** Fixed grid, not sampling, so
  the separation curve is identical across captures.
- **R6 — image A/B viability on ~800 sprites. RESOLVED (pilot,
  2026-08-02).** The classic no-penalty config collapses completely by
  step 15000 (all 64 samples identical) while the penalized relativistic
  config stays diverse and seed-stable with no memorization signal
  (train/held-out critic gap ≈ 0). The A/B contrast at image scale is
  therefore *stronger* than the 25-Gaussians toy's and needs no hedging.
  Quality bar: grids `runs/r2_g10_s0/grid_ema_final.png`,
  `runs/r2_g10_s1_confirm/grid_ema_final.png` (pass, per pilot and
  orchestrator inspection — diverse, creature-like, clean silhouettes;
  sent to Alex, veto window open), vs. `runs/r6_A_s0/grid_ema_final.png`
  (the collapse). Pokemon stands unless Alex vetoes.
- **R7 — 16.6 demo stability.** The two-mode regression A/B must show the
  mean-vs-modes contrast on every seed; fix all seeds and keep the target
  variance large relative to noise.
