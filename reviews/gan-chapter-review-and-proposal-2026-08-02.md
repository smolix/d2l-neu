# GAN Chapter (ch. 16): Review and Rebuild Proposal

2026-08-02, revised same day after adversarial review. Prepared for Alex's
review before any implementation. The companion document
`gan-implementation-brief.md` specifies each notebook cell-by-cell; this one
records the assessment, the design, and its rationale.

Evidence base (full reports in `reviews/gan-research/`):

| File | Scope |
|---|---|
| `01-courses.md` | How 11 top courses teach GANs, 2024–26 (Stanford, Berkeley, MIT, CMU, ETHZ, Tsinghua, Cornell, …) |
| `02-literature.md` | R3GAN, Jolicoeur-Martineau, Mescheder, Sun et al.; where adversarial objectives are used 2024–26; evaluation practice; verification and novelty checks against `gan-notes/gan-objectives.tex` |
| `03-texts-tutorials.md` | Bishop 2024, Murphy PML-AT, Prince UDL, Tomczak, Foster, Goodfellow's tutorial, Weng; failure modes of GAN chapters |
| `04-in-repo.md` | Exact `:label:` inventory of usable appendix material; ch. 16 dependency surface; conventions |

Throughout, "the note" means `gan-notes/gan-objectives.tex` ("Adversarial
Objectives in Closed Form"), which supplies the chapter's mathematical spine.

---

## 1. Assessment of the current chapter

Two sections (`gan.md`, `dcgan.md`), essentially unchanged since ~2019.

What is worth keeping: the two-sample-test opening (no textbook uses it,
though Stanford CS236 builds its lecture on the same statistical frame), the
2-D Gaussian toy problem (every course surveyed uses one), the Pokemon
dataset (small, distinctive, trains in under a minute), and dcgan.md's
transposed-convolution shape arithmetic (matches the book's
derive-the-shapes habit).

What fails against the book's own standard:

1. No derivation of the optimal discriminator or the Jensen–Shannon value of
   the game. The non-saturating loss is asserted to share the optimum,
   not shown. This is the one derivation every serious course and text does.
2. The toy example never computes anything the theory predicts. It fits a
   Gaussian and shows a scatter plot; it does not check the discriminator
   against the known density ratio or measure the fit.
3. Nothing after 2015 appears: no instability analysis, no WGAN/IPM material,
   no gradient penalties, no evaluation metrics, no relativistic objectives.
4. Zero cross-references into the math appendix, although
   `sec_mdl-divergences-distances` was written with GANs named in its intro
   and already proves the f-GAN bound, Kantorovich–Rubinstein duality, and
   the MMD closed form (see §4 below).
5. Four framework tabs, contrary to the Advanced-part PyTorch+JAX policy.
   The audit confirms the chapter is self-contained (`update_D`/`update_G`
   and the `pokemon` dataset have no outside consumers), so the drop is
   mechanical.
6. Exercises: one open question plus two prompts, none derivation-based.

Prose quality is pre-style-guide throughout ("world's lamest example", "we
will spare you another spiel").

## 2. What the research established

Four findings shape the design; the rest is detail for the brief.

**The teaching core shrank, and what survived matches the note's spine.**
Across 2024–26 courses the stable core is: implicit generator contrasted with
likelihood models; the minimax game; optimal discriminator → JS; the
non-saturating loss; instability and mode collapse on a 2-D toy. WGAN and FID
are near-core. The GAN zoo, the trick bestiary, and f-GAN duality (outside
Stanford CS236) have been dropped. Courses now frame GANs three ways:
historical milestone, a rung on the divergence ladder, or — the growing one —
the adversarial loss as a live tool (Kaiming He ends his lecture at VQ-GAN
tokenizers; Berkeley's homework is 40% VQGAN; MIT assigns diffusion-into-GAN
distillation).

**Nobody teaches the modern result.** R3GAN appears in exactly one course, as
a student-seminar reading (MIT 6.S978). No textbook covers it (all predate
it). No conference tutorial 2023–26 treats GANs at all. No source connects
MLE → KL → adversarial training as one derivation chain, and none states the
closed-form value of the relativistic game. The chapter would not be
summarizing a known synthesis; it would be the first pedagogical treatment.

**The 2026 field position is specific enough to teach.** Training a GAN from
scratch as the frontier generative model is over; the adversarial loss as a
finishing step is standard practice (ADD/SDXL-Turbo, FLUX.1-schnell,
DMD2, real-time video, vocoders, VQGAN-family tokenizers). One ablation
summarizes why: in ADD, one-step distillation without the adversarial term
gives FID 315.6; the adversarial term alone gives 20.8. The
counter-evidence is equally concrete: MeanFlow and sCM reach one/two-step
quality with no discriminator, ViTok-v2 removes the GAN loss at 5B
parameters because it prevents stable scaling, and DDO (ICML 2025
spotlight) obtains discriminator-style gains with no two-player training at
all. A 2025–26 line (R3GAN → GAT → CAT, FID 1.56 one-step ImageNet-256) has
made pure GANs competitive again. The chapter teaches both results and the
unresolved question between them.

**The note survives scrutiny, with four corrections.** All five rows of its
proper-loss table and the d_Rp = JS(p⊗q, q⊗p) identity verify numerically to
machine precision. The R3GAN Eq. (12) missing square is confirmed a typo in
the camera-ready (checked four ways). Two claims need fixing before the
chapter inherits them: R3GAN's released code trains the generator with the
non-saturating form (the note says it uses none), and the TV↔hinge identity
is the note's own, not Lim & Ye's. PairGAN (arXiv:2002.08621) constructs the
same pair mixture but computes a different quantity; citing it locates the
new identity precisely. Novelty confidence after inspecting the full
citation graph of Jolicoeur-Martineau 2020 and source-grepping the primary
papers: ~80%. A full-text literature pass has not yet been done and is
scheduled in the brief before the chapter prints the priority claim.

## 3. Design

### 3.1 Principles

1. **One spine: closed-form values of adversarial games.** Every objective in
   the chapter gets the same treatment — what does the game evaluate at the
   optimal critic, and what does that closed form predict about training? The
   organizing question ("which quantity, and does its gradient survive?") is
   asked once, in §16.1, and answered five times.
2. **The critic estimates a density ratio; the rest is parameterization.**
   This single fact (D* = log p/q for the log-loss game, T* = f′(p/q) for
   f-GANs, the same log-ratio for the relativistic game) is what unifies the
   chapter and what connects it back to MLE: maximum likelihood minimizes a
   KL whose log-ratio it can evaluate; adversarial training estimates the
   ratio it cannot.
3. **Code verifies theorems.** Each notebook computes something the section
   proved: the recovered density ratio against ground truth, the closed-form
   game value against a numerically maximized one, the predicted
   saturation/convergence behavior against observed training runs.
4. **2026 content on merit, not survey-frequency.** The zoo, the trick
   bestiary, StyleGAN internals, CycleGAN/pix2pix, InfoGAN, BiGAN: out
   (contrast material only, one line each where theory explains what replaced
   them). R3GAN's recipe, the penalties' geometry, evaluation-metric
   closed forms, and the distillation/tokenizer role of adversarial losses:
   in.
5. **Cite the appendix, derive only what is new here.** The book already
   proves the f-GAN bound, KR duality, MMD, forward/reverse KL, MLE=KL, and
   InfoNCE. The chapter states these with `:numref:` and spends its
   derivation budget on what the appendix lacks: the value of the log-loss
   game, the Bayes-risk-gap view of losses, the relativistic closed form,
   and the penalty analysis.
6. **Images appear as demonstrations, not as the subject.** One image
   notebook (16.5). Serious image modeling stays in the Image Models part;
   the section says so explicitly and points forward.

### 3.2 Proposed structure

Seven files: `index.md` plus six sections. PyTorch + JAX only. (The earlier
draft of this proposal packed the relativistic objective and the
convergence/penalty analysis into one section; review flagged that it
answered two questions and carried a third of the chapter. They are now
separate, which also matches the spine's own order: value of the game first,
then the geometry that makes gradient descent find it.)

**index.md — chapter opening (no code).** The problem: a sampler with no
density, so maximum likelihood — the tool the whole book has used — does not
apply. What replaces the likelihood is a learned comparison between generated
and real samples. Scope (population-level objectives; small experiments;
images later) and roadmap by dependency.

**16.1 `gan.md` — Generative Adversarial Networks.** From MLE to the
log-loss game. MLE = forward-KL minimization (cite
`subsec_mdl-nll-crossentropy`); why likelihood is unavailable for implicit
models and why good likelihood ≠ good samples (the mixture argument);
the two-sample view; the game; the optimal discriminator and the value
2 JS − 2 log 2 with the entropy-Jensen-gap and mutual-information readings;
the non-saturating generator loss derived from the gradient, not asserted.
Experiment: the 2-D Gaussian toy, upgraded to verify — recovered density
ratio vs. ground truth, saturating-vs-non-saturating A/B, fit quantified
with the closed-form Gaussian KL. Closes on the failure that drives the
chapter: disjoint supports pin the value at log 2 with zero gradient.

**16.2 `objectives.md` — Adversarial Objectives and Divergences.** The field
guide. The margin template (note Eq. 1); every proper loss yields a
Bayes-risk gap, every gap is an f-divergence (the loss→divergence table:
logistic→JS, square→triangular/LSGAN, hinge→TV); Fenchel route to
variational f-divergences (cite `sec_mdl-f-gan-dual`, restate the
optimal-critic pattern T* = f′(p/q)); IPMs — MMD (cite `sec_mdl-ipm-mmd`,
analytic for a fixed kernel) and W₁ (cite `sec_mdl-optimal-transport`);
the disjoint-support dichotomy and the opposing estimation rates.
Experiments: a separation experiment (two narrow Gaussians at increasing
distance; JS saturates while W₁ and MMD keep slope — the point-mass example
at computable width); one shared low-dimensional testbed trained under four
losses with a per-run density-ratio diagnostic; an f-GAN critic recovering
the analytic ratio. The appendix's objective→divergence map
(`sec_mdl-divergence-objective-map`) is cited as the summary artifact rather
than rebuilt.

**16.3 `relativistic.md` — Relativistic Objectives.** The pairing objective;
ranking/Bradley–Terry reading (cross-ref `eq_bradley_terry` in ch. 15 and,
in one sentence, preference-Elo evaluation of generative models); the two
symmetries (additive shift, dependence on the product measure); optimal
critic is the same log-ratio; the closed form d_Rp = JS(p⊗q, q⊗p) via the
lifting argument, proved in full in-chapter and presented as this book's
result with PairGAN cited as nearest prior art; properties (JS ≤ d_Rp ≤
log 2, locally twice JS, still saturates on disjoint support); why ranking
removes the mode-dropping landscape (rank weight vs. threshold weight; Sun
et al. stated); saturating vs. non-saturating pairing, including the
paper-vs-reference-code discrepancy as a worked lesson. Experiment:
numerical verification of the closed form on a finite sample space.

**16.4 `convergence.md` — Gradient Penalties and Convergence.** What the
objective alone cannot buy. Zero-centered penalties R₁ + R₂ = γ E_m‖∇D‖²;
the Dirac-GAN analysis in full — continuous flow circles, discrete
simultaneous gradient descent spirals outward, either penalty contracts
(with the corrected eigenvalue formula and the critically damped
γ = 2|ℓ′(0)|); the linearized-W₂ reading of the penalized game (cite
`eq_mdl-benamou-brenier`); the one-centered/zero-centered contrast table
(the WGAN-GP penalty keeps a nonzero slope at p = q and fails the Dirac
test — its one lesson here); when one penalty is not enough (R3GAN's
ablation, scoped: StyleGAN2 trained with R₁ alone at FFHQ scale, the toys
and StackedMNIST did not). Ends with the R3GAN recipe: the loss in ~15
lines, the six principles, the trick-removal roadmap A→E, the StackedMNIST
ablation quoted. Experiments: Dirac trajectories (three panels: ODE,
discrete, discrete+penalty); the 25-Gaussians mode-coverage A/B.

**16.5 `dcgan.md` — Adversarial Image Generation.** DCGAN compressed to its
historical role (the 2015 recipe: what it fixed, what it could not fix), then
the modern minimal recipe from R3GAN's principles (bilinear resampling, leaky
ReLU, no normalization in G/D, Adam without momentum, light augmentation,
EMA cross-ref `training-recipes`), trained on Pokemon with the 16.3/16.4
loss. The section's experiment is an A/B on identical backbones: 2015 loss
vs. RpGAN+R₁R₂. Evaluation connects back to the chapter's own closed forms:
FID is the Gaussian W₂² (Bures–Wasserstein formula, stated with a
commuting-case check and cited); KID/CMMD is the unbiased MMD estimator, now
with a learned feature space — which reopens the kernel choice 16.2 fixed.
Metric fragility (feature dependence, ImageNet-class leakage, aliased
resizing, finite-sample bias) and benchmark hygiene (R3GAN's refusal of
ImageNet-pretrained discriminators) get a compact treatment. Pointer
forward: image applications in the Image Models part; R3GAN's real compute
budgets quoted so the reader can place the toy on the scale.

**16.6 `adversarial-losses.md` — Adversarial Losses Beyond GANs (closing).**
Where the objective lives in 2026 and why. The capacity argument — a bounded
student under MSE averages and blurs; a critic penalizes leaving the data
manifold — demonstrated in one short cell (two-mode regression, MSE student
vs. adversarial student). Distillation (ADD's ablation, DMD2's
escape-the-teacher motive, FLUX.1-schnell in production), tokenizers (VQGAN
role and the ViTok-v2 reversal), audio (where adversarial training remains
standard practice), real-time video. The counter-programme (MeanFlow, sCM,
DDO) stated with equal weight. The three exits from instability — fix the
game (Mescheder → R3GAN), constrain the critic (spectral norm → frozen
pretrained features), remove the game (MMD/IMLE → diffusion as regression)
— mapped onto the note's three axes. The closing subsection, "The
Discriminator Inside Likelihood Models": DDO reads a likelihood model as an
implicit discriminator through the same identity D* = log p/q that opened
the chapter. Handoff to ch. 17.

### 3.3 What is deliberately excluded, and why

- **The variant zoo** (InfoGAN, BiGAN, BEGAN, SAGAN, progressive growing,
  StyleGAN internals, CycleGAN/pix2pix): architecture history that theory
  now explains away or that belongs with image applications. CycleGAN gets
  one sentence in 16.6 as an adversarial-loss application; StyleGAN appears
  only as the baseline R3GAN strips.
- **The trick bestiary** (feature matching, minibatch discrimination,
  historical averaging, TTUR, unrolling, virtual BN): each existed to patch
  a failure the chapter now derives; R3GAN's ablation is the evidence they
  are removable. Mentioned collectively, once.
- **Weight clipping and WGAN-GP as recipes**: the KR constraint is taught;
  the one-centered penalty appears only in 16.4's contrast table.
- **Full f-divergence generality** (measurable-selection remarks, the
  α-family): the appendix carries it.
- **Conditional GAN training**: the image notebook stays unconditional
  (conditioning is orthogonal to the objective and would double the
  experiment budget). Per Alex's review, the *idea* is sketched in 16.5: a
  condition fed to both networks turns the game into per-condition
  distribution matching, with the concatenation and projection-discriminator
  mechanisms each in a sentence and the observation that R3GAN's CIFAR-10/
  ImageNet results and all text-to-image distillation are conditional.

### 3.4 Relation to the rest of the book

The chapter becomes the missing consumer of appendix material that already
anticipates it (details and exact labels in the brief): MLE=KL
(`sec_mdl-maximum_likelihood`), JS and the divergence gallery, f-GAN bound
(`sec_mdl-f-gan-dual`), MMD (`sec_mdl-ipm-mmd`), KR/W₁
(`sec_mdl-optimal-transport`), W₂/Benamou–Brenier (dynamics chapter),
InfoNCE (`sec_mdl-infonce`, exercise-level), Bradley–Terry (ch. 15
`regularized.md`), EMA (`training-recipes`). The deep-RL `regularized.md`
citation pattern (name the appendix proposition, use its properties, do not
re-derive) is the model. Diffusion (ch. 17) receives the handoff; the Image
Models part receives image applications.

### 3.5 The note and the chapter

The chapter adapts the note's §1–§7 at textbook register (motivation-first,
smaller inferential steps, code interleaved) and at Tier-2 prerequisites
(state duality results, cite the appendix for proofs). Three corrections
flow back into the note regardless of the chapter work: the Eq. (12) footnote
becomes a flat statement; the saturation remark acknowledges the
non-saturating reference implementation; the TV↔hinge attribution is
reworded; PairGAN is cited at the theorem footnote. The InfoNCE/K-negatives
extension and the separability proposition become exercises rather than
chapter text.

### 3.6 Infrastructure (summary; checklist in the brief)

Framework tabs `%%tab pytorch` / `%%tab jax` throughout (RL-chapter
pattern); delete `outputs/{tensorflow,mxnet}/chapter_generative-adversarial-networks/`;
`CHAPTER_NUMBERING` gains entries for the four new files; new
`tools/gen_mdl_gan_figures.py` (prefix `mdl-gan-`) replaces the legacy
hand-drawn `gan.svg`; ~20 new `d2l.bib` entries (none of the chapter's
post-2016 citations exist there today); slides rewritten per section; two
incidental fixes ride along (`chapter_optimization/practice.md` line 730
hardcoded "ch. 15"; CLAUDE.md part-structure bullet stale since the RL
split).

Compute: every notebook CPU-tolerable except 16.5's image A/B; the chapter
re-executes in ~30 GPU-minutes per framework, or up to ~60 if the
mode-coverage pilot forces the StackedMNIST fallback (brief §10). Prose
quotes only re-execution-stable quantities (mode counts as thresholds, not
exact seeds; loss curves qualitatively).

## 4. Open decisions for Alex

1. **Section count.** Six sections as proposed; or five by folding 16.6 into
   16.5 (keeps the closing material but subordinates it to the image
   section); or five by re-merging 16.3+16.4 (rejected by review as
   two-questions-in-one, but listed for completeness).
2. **The new identity's presentation.** State d_Rp = JS(p⊗q, q⊗p) with the
   full lifting proof in-chapter (four lines given 16.1's machinery, as now
   planned), or state and cite the note. In-chapter proof proposed.
3. **Publication order for the note.** The chapter will print a priority
   claim ("we have not found this identity in the literature"). Posting
   `gan-notes` to arXiv first would give the chapter a stable citation and
   timestamp the claim. Related: the remaining full-text literature pass
   (brief §8) runs either way.
4. **Dataset for 16.5.** Keep Pokemon (continuity, cheap, distinctive) or
   switch to FFHQ-64 thumbnails (matches R3GAN's benchmark, less whimsical)?
   Pokemon proposed.
5. **Feature network for the 16.5 metrics.** Proposed: a small CNN
   classifier trained in-notebook on CIFAR-10 (PT/JAX, ~2 min) whose
   penultimate features score the sprite runs — off-domain features by
   construction, which is the same structural situation as Inception-on-
   ImageNet scoring faces, and the fragility discussion uses that fact.
   Alternative: a pretrained downloaded backbone (realistic, heavier
   dependency, less self-contained).
6. **File naming.** `objectives.md`, `relativistic.md`, `convergence.md`,
   `adversarial-losses.md` as proposed. Flag: `objectives.md` and
   `adversarial-losses.md` are near-synonyms as stems; alternatives for the
   latter: `beyond-gans.md`, `adversarial-training.md`. (Renaming
   `gan.md`/`dcgan.md` is not proposed; URL stability.)

## 5. Decisions (Alex, 2026-08-02)

All six resolved in review:

1. Six sections. 2. Identity proved in-chapter. 3. No separate arXiv
posting of the note — novel content lives in the book. 4. Pokemon, with a
quality bar: the generated samples must look good ("cute"), enforced via
pilot R6; escalate to FFHQ-64 only if the demo fails that bar. 5. Feature
network trained in-notebook (CIFAR-10 CNN); switch to a pretrained
downloaded backbone only if it scores poorly in pilot R4. 6. File names as
proposed.

One addition from review: conditional GANs must be sketched, not merely
name-checked — now a short subsection in 16.5 (see §3.3 and the brief).
