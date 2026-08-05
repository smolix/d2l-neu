# Chapter Overview — chapter_generative-adversarial-networks

Best external match by far: Stanford CS236 (Ermon & Grover) Homework 3 (2018, verified
by direct PDF fetch), which independently derives the same optimal-discriminator,
projection-discriminator, and Wasserstein/Lipschitz identities as gan.md, conditional.md,
and objectives.md, each paired with a small Fashion-MNIST coding task. Goodfellow's NeurIPS
2016 tutorial (arXiv:1701.00160, confirmed via full-text fetch) contributes three explicit,
solved exercises — optimal discriminator, an oscillating toy game anticipating the
Dirac-GAN, and the non-saturating/KL equivalence. The hardest sections (relativistic.md,
convergence.md) have no course-homework tradition at all; their best "sources" are the
original papers' own toy examples and ablation tables (Mescheder et al. 2018,
Jolicoeur-Martineau 2019/2020, Sun-Fang-Schwing 2020, R3GAN 2024) — a genuine finding, not
a search failure. University of Toronto's CSC321 CycleGAN assignment (Grosse, 2018,
verified by direct PDF fetch) is an excellent, previously-unused match for
conditional.md's translation section. MIT 6.S191, suggested as a source, does **not**
currently have a dedicated GAN-training lab (checked directly): its generative-modeling
lecture pairs with a VAE-based debiasing lab, not a GAN notebook — the labs are
CNN/MNIST classification and VAE debiasing only. Berkeley CS294-158's GAN homework
(Spring 2020, HW4) exists but its problem-level content could not be verified beyond
video/README summaries, so it is not cited as provenance. UDL (Prince) ch. 15 Problems
15.2–15.6 map cleanly onto gan.md/objectives.md's divergence content; 15.6 is already the
book's own cited source for conditional.md's truncation exercise. Overall the chapter's
existing 39 exercises are exceptional: only 2 need even a formatting rewrite (inline-lettered
sub-parts), none need dropping, so external material is used almost entirely to *add*
depth rather than replace anything.

---

## chapter_generative-adversarial-networks/gan.md — Generative Adversarial Networks

**Topic:** The original GAN minimax objective: the log-loss game, its optimal discriminator
(the log density ratio), the value at optimum (Jensen–Shannon divergence), and the
saturating-vs-non-saturating generator loss, verified end-to-end on a 2-D Gaussian.

**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — every exercise names a
concrete derivation or numerical check with a stated comparison target; the prior style
review flagged no clarity or formatting issues in this file.

**External sources found:**
- Stanford CS236 (Ermon & Grover), Homework 3, Problem 2.1–2.2, 2018 — derives
  D\*=p_data/(p_data+p_θ) and shows the logits equal the log density ratio, by the same
  pointwise-optimization argument this section uses — verified via direct PDF fetch of
  `raw.githubusercontent.com/Subhajit135/CS236_DGM/master/hw/CS236_Homework_3.pdf`.
- Stanford CS236, Homework 3, Problem 1.1, 2018 — asks students to show the minimax
  generator loss's gradient vanishes as D(G(z))→0, the saturation fact this section proves
  via the sigmoid weights.
- Stanford CS236, Homework 3, Problem 1.2, 2018 — a short coding task: implement the
  non-saturating loss and train a GAN on Fashion-MNIST for one epoch, checking that
  samples become "roughly recognizable."
- Stanford CS236, Homework 3, Problem 2.3, 2018 — shows that the *sum* of the minimax and
  non-saturating losses equals KL(p_θ‖p_data) exactly at the optimal critic — a cleaner
  exact identity than this section's non-saturating-alone inequality.
- Goodfellow, "NIPS 2016 Tutorial: Generative Adversarial Networks" (arXiv:1701.00160),
  Exercise 1 (§7.1/8.1: derive the optimal discriminator) and Exercise 3 (§7.3/8.3: show the
  non-saturating loss is equivalent in expectation to approximate KL minimization under an
  optimal discriminator) — confirmed via full-text (ar5iv) fetch.
- Prince, *Understanding Deep Learning*, ch. 15, Problem 15.2 — "relate the loss … to the
  Jensen–Shannon distance" — this section's central identity, posed with no derivation
  scaffolding.

**Proposed problem set** (7 problems):
1. [conceptual] **Shift Invariance of the Critic.** Show that shifting the optimal critic
   D\*+b for b≠0 strictly lowers V, by examining the pointwise objective's unique maximizer.
   *Provenance:* original.
1. [conceptual] **Unequal Priors, Skewed Divergence.** Derive the optimal critic and the
   value of the game under an unbalanced label prior P(y=1)=α, and identify the resulting
   skewed divergence.
   *Provenance:* original.
1. [conceptual] **Likelihood Under a Noise Mixture.** Generalize the mixture bound to
   p̃=εp_data+(1−ε)p_noise, find the ε at which the likelihood penalty reaches one nat, and
   report the corresponding noise fraction.
   *Provenance:* original.
1. [conceptual] **Two Generator Weights.** Differentiate the saturating and non-saturating
   losses and evaluate both weights at D=±3 to quantify the ~20× gap in step size.
   *Provenance:* original.
1. [short-code] **Critic Error Across a Grid.** Extend the mixture verification from
   sampled points to a fixed grid and report the sup-norm error of σ(D) against the
   analytic posterior, explaining where and why it is largest.
   *Provenance:* original.
1. [conceptual] **Exact KL from Combined Losses.** Show that the *sum* of the minimax loss
   E_q[log σ(-D)] and the non-saturating loss −E_q[log σ(D)], evaluated at D\*=λ, equals
   KL(q‖p) exactly — tighter than this section's inequality (:eqref:`eq_gan_ns_value`) for
   the non-saturating loss alone. State which two losses a practitioner would need to sum
   in code to obtain this exact identity.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problem 2.3 (2018) (overlap
   high — same identity and optimal-critic substitution, framed here against the section's
   own bound rather than derived from scratch).
1. [short-code] **Log-2 Calibration on Real Images.** Reuse `update_D`/`update_G` from this
   section, but replace the 2-D Gaussian with a small batch of flattened, downsampled
   (e.g. 16×16) grayscale Fashion-MNIST images and MLP generator/critic networks of
   comparable size. Train for a short fixed budget and report whether the discriminator's
   per-sample loss approaches log 2 as it did for the Gaussian; note any systematic gap and
   suggest why a non-Gaussian, higher-dimensional target might close more slowly or
   plateau higher.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problem 1 (coding task: train a
   non-saturating GAN on Fashion-MNIST) (overlap medium — same dataset and loss, different
   success criterion: this section's log-2 calibration rather than visual recognizability).

---

## chapter_generative-adversarial-networks/objectives.md — Adversarial Objectives and Divergences

**Topic:** Generalizing the log-loss game along two axes — the classification loss (giving
f-divergences via the Bayes-risk gap, and the f-GAN duality) and the critic class (giving
integral probability metrics: MMD and Wasserstein-1) — and comparing which retain a
gradient once supports separate.

**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — dense, well-scaffolded
derivations plus one clean numerical falsification (linear-kernel MMD); no clarity or
formatting issues flagged.

**External sources found:**
- Stanford CS236, Homework 3, Problem 4.1–4.2, 2018 — derives
  KL(N(θ,ε²)‖N(θ₀,ε²))=(θ−θ₀)²/2ε² and shows it (and its gradient) blows up as ε→0 for
  θ≠θ₀ — the same disjoint-support pathology this section demonstrates by quadrature for
  JS/MMD/W1.
- Stanford CS236, Homework 3, Problem 4.3–4.4, 2018 — shows no discriminator minimizes the
  raw linear-critic objective on two point masses, and that restricting the critic to
  1-Lipschitz functions restores a minimizer — the Kantorovich–Rubinstein fact this
  section states but does not re-derive from a degenerate case.
- Nowozin, Cseke & Tomioka, "f-GAN: Training Generative Neural Samplers using Variational
  Divergence Minimization" (NeurIPS 2016, arXiv:1606.00709) — the duality this section's
  own Exercise 5 already re-derives for the JS generator; correctly cited in the text.
- Prince, *Understanding Deep Learning*, ch. 15, Problem 15.4 — compute KL, reverse KL, JS,
  and Wasserstein distance between two shifted uniform distributions over separation
  a∈[−3,3] — essentially this section's own separation experiment, for uniforms instead
  of Gaussians, posed as a hand/plot exercise rather than in code.
- Prince, ch. 15, Problem 15.3 — set up the earth-mover's-distance primal linear program
  (an 8×16 constraint matrix) for two four-point discrete distributions — a complementary,
  LP-based route to Wasserstein distance that this section only presents via duality.
- No verified course homework treats the "proper loss ⇒ f-divergence via the Bayes-risk
  gap" construction as its own problem — CS236 jumps straight from the minimax loss to
  Wasserstein/f-GAN without this unifying lemma. A real gap in the external tradition,
  already filled well by this section's own Exercises 1–3.

**Proposed problem set** (7 problems):
1. [conceptual] **Brier Loss's Bayes Risk.** Derive the Brier loss's conditional Bayes risk
   and confirm the table's square-row value.
   *Provenance:* original.
1. [conceptual] **Zero-One Loss and Total Variation.** Derive Δ₀₋₁=½TV(p,q) and explain why
   the hinge row is exactly twice this from the Bayes risks alone.
   *Provenance:* original.
1. [short-code] **LSGAN's Triangular Discrimination.** Verify numerically on random
   discrete distributions that LSGAN's gap equals ⅛ of the triangular discrimination.
   *Provenance:* original.
1. [short-code] **A Blind Spot for MMD.** Swap in a linear kernel and a scale-only
   perturbation to show MMD can fail where JS and W1 still detect a difference, and
   connect the failure to characteristic kernels.
   *Provenance:* original.
1. [conceptual] **f-GAN Duality for JS.** Derive the conjugate of the (doubled) JS
   generator, its domain, and the reparameterization recovering the original GAN's value
   function.
   *Provenance:* original.
1. [conceptual] **Lipschitz Bounds for IPMs.** Using the two point-mass construction of
   this section's separation experiment, show that the unconstrained linear objective
   sup_T{E_p[T]−E_q[T]} is unbounded (take T(x)=cx, c→∞), and that restricting T to the
   1-Lipschitz ball both bounds the supremum and attains it at exactly the separation
   distance d. State in one sentence what this adds to the Kantorovich–Rubinstein fact
   already used in the text.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problems 4.3–4.4 (overlap
   medium — same degenerate-critic argument, reframed in this section's IPM/Lipschitz-ball
   language rather than the homework's raw-discriminator framing).
1. [short-code] **KL Divergence Under Separation.** Extend this section's
   `js_1d`/separation-sweep cell with a fourth curve: the analytic
   KL(N(0,1)‖N(d,1)) (closed form, already available from :numref:`sec_basic_gan`'s
   Gaussian KL formula) over the same grid of separations, plotted on a log scale since it
   is unbounded. Report at what separation KL first exceeds JS's ceiling log 2, and state
   why KL cannot be the value of any game in this section's Bayes-risk-gap family.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problem 4.1–4.2 (overlap
   medium — same divergence and limit, applied to this section's own separation-sweep code
   rather than derived symbolically for a fixed ε).

---

## chapter_generative-adversarial-networks/relativistic.md — Relativistic Objectives

**Topic:** The relativistic pairing objective (RpGAN): scoring a real–fake pair by the
difference of critic scores, its shift-invariance, its reduction (by "lifting") to the
classical log-loss game on the product space, the value JS(p⊗q, q⊗p), and how the
rank-based generator weight it induces removes many mode-dropping minima.

**Current exercises:** 6; disposition: keep 5, rewrite 1, drop 0 — Exercise 6 packs three
lettered sub-parts (a)–(c) inline in one dense paragraph, the "crammed inline lettering"
formatting defect the prior style review flagged; its content (general concave payoffs,
the two-sided lifted game, the least-squares case) is excellent and should be kept, only
reformatted as a nested list.

**External sources found:**
- Jolicoeur-Martineau, "The relativistic discriminator: a key element missing from
  standard GAN" (arXiv:1807.00734, ICLR 2019) — the paper this section builds on;
  confirmed to match the section's citation. Its own experiments (RaGAN-GP vs. WGAN-GP
  sample quality) are training-scale, not adaptable at this section's toy scale.
- Jolicoeur-Martineau, "On Relativistic f-Divergences" (ICML 2020) — proves the general
  concave-payoff nonnegativity/vanishing-at-q=p result this section's own Exercise 6
  already explores; no additional problem beyond what the section already poses.
- Sun, Fang & Schwing, "Towards a Better Global Loss Landscape of GANs" (NeurIPS 2020) —
  proves the nⁿ−n! bad-local-minima count for the classical objective and its absence for
  the relativistic one, cited in this section's text; its finite-sample counting argument
  is small enough to reproduce directly, which the section does not currently exercise.
- Huang, Gokaslan, Kuleshov & Tompkin, "The GAN is dead; long live the GAN!" (R3GAN,
  arXiv:2501.05441, NeurIPS 2024) — already the section's citation for the rank-weight
  table and the paper-vs-code discrepancy; no homework-style material beyond ablation
  tables already used in convergence.md/dcgan.md.
- No good external exercise tradition: no Stanford CS236 or Berkeley CS294-158 homework
  was found covering relativistic/pairwise discriminators at all — both courses' GAN
  homework predates or omits this line of work. External material here is necessarily
  paper-derived rather than course-derived.

**Proposed problem set** (7 problems):
1. [conceptual] **Shift Invariance, Proved.** Prove Φ(D+b)=Φ(D) and contrast with why the
   same shift strictly lowers the classical value function.
   *Provenance:* original.
1. [short-code] **Shift Invariance, Checked Numerically.** Add b=5 to the recovered
   five-atom critic and confirm Φ is unchanged to machine precision.
   *Provenance:* original.
1. [conceptual] **Unequal Ordering Priors.** Derive the optimal pair critic and the
   resulting skewed divergence when the real-first presentation has probability α≠½.
   *Provenance:* original.
1. [conceptual] **Local Expansion to Second Order.** Expand d_Rp and JS to O(ε²) under a
   local perturbation q=p(1+εh) and confirm the factor-of-two ratio.
   *Provenance:* original.
1. [conceptual] **From Pairs to InfoNCE.** Generalize to K real samples and one fake sample
   in a random order, deriving the generalized JS divergence and its InfoNCE connection.
   *Provenance:* original.
1. [conceptual] **General Concave Payoffs.** (Reformatted as a nested `1.` sub-list rather
   than an inline (a)/(b)/(c) paragraph — content unchanged.) Show that restricting an
   unrestricted two-sided pair game to difference critics costs nothing under the logistic
   payoff but strictly loses under the least-squares payoff.
   *Provenance:* original.
1. [short-code] **Counting Mode-Dropping Minima.** Extend the five-atom setup to n=3
   "real" atoms and enumerate all nⁿ=27 assignments of 3 generated draws to real atoms
   (with repetition allowed); for each assignment, compute the best-response critic and the
   resulting value under both the classical objective and Φ (reusing the gradient-ascent
   code from the verification cell), and count how many assignments are strict local
   minima under each. Compare the count against Sun–Fang–Schwing's nⁿ−n!=21 bound for the
   classical objective, and report whether any assignment is a strict local minimum under
   the relativistic objective.
   *Provenance:* adapted from Sun, Fang & Schwing, "Towards a Better Global Loss Landscape
   of GANs" (NeurIPS 2020) (overlap medium — reproducing the qualitative claim by
   brute-force enumeration at trivial scale, not their proof technique).

---

## chapter_generative-adversarial-networks/convergence.md — Gradient Penalties and Convergence

**Topic:** Why a correct equilibrium doesn't imply convergent training — the Dirac-GAN's
circular continuous-time dynamics vs. outward-spiraling discrete updates — and how
zero-centered gradient penalties R1/R2 restore local convergence and (near equilibrium)
measure a linearized Wasserstein-2 distance; tested via the R3GAN recipe on 25 Gaussians.

**Current exercises:** 5; disposition: keep 4, rewrite 1, drop 0 — Exercise 4 packs four
lettered sub-questions (a)–(d) inline in one paragraph, the same formatting anti-pattern
flagged in the style review; its content (reconciling coverage with off-mode mass,
predicting damping-regime failures) is strong and should be kept, only reformatted.

**External sources found:**
- Mescheder, Geiger & Nowozin, "Which Training Methods for GANs do actually Converge?"
  (arXiv:1801.04406, ICML 2018) — the paper this entire section builds on; confirmed to
  match the Dirac-GAN construction, eigenvalue results, and the claim that alternating
  (rather than simultaneous) updates also fail to converge for the unpenalized game — a
  claim the text states but the section's own exercises do not empirically test.
- Goodfellow, NIPS 2016 Tutorial (arXiv:1701.00160), Exercise 2 (§7.2/8.2) — a toy example
  asking the reader to run simultaneous gradient descent on a simple two-player game and
  observe oscillation rather than convergence, predating and pedagogically anticipating
  the Dirac-GAN by a year; confirmed via full-text fetch.
- Huang, Gokaslan, Kuleshov & Tompkin, "The GAN is dead; long live the GAN!" (R3GAN,
  arXiv:2501.05441) — source of the StackedMNIST and FFHQ-256 ablation tables already used
  verbatim in this section's text; no further homework-style material beyond what is
  already cited.
- No good external exercise tradition: no Stanford CS236, Berkeley CS294-158, or other
  course homework treats the Dirac-GAN, gradient penalties, or GAN training dynamics as a
  standalone problem (CS236's WGAN-GP problem, usable as an objectives.md source, does not
  touch local-convergence analysis or the Dirac-GAN). Provable non-convergence of a
  *correct* objective has essentially no course-homework tradition outside the original
  papers — a real finding, not a search failure.

**Proposed problem set** (6 problems):
1. [conceptual] **Eigenvalues of the Discrete Map.** Compute the Jacobian of the discrete
   simultaneous update at equilibrium and show its spectral radius exceeds 1 for every
   step size.
   *Provenance:* original.
1. [conceptual] **Mixture Weighting in the Penalty.** Derive the R1+R2 sum and construct a
   separated-support counterexample showing why R1 alone can leave the critic's gradient
   on generated samples uncontrolled.
   *Provenance:* original.
1. [conceptual] **A Sobolev-Norm Interpretation.** Prove the ray-then-direction
   optimization giving the dual Sobolev-norm value, and confirm its 1/γ scaling.
   *Provenance:* original.
1. [conceptual] **Coverage Versus Off-Mode Mass.** (Reformatted as a nested `1.` sub-list
   rather than an inline (a)–(d) paragraph — content unchanged.) Reconcile full mode
   coverage with a small on-mode fraction, quantify the off-mode bin's KL contribution, and
   predict the failure modes of γ=10 vs. γ=0.1.
   *Provenance:* original.
1. [short-code] **Sweeping the Step Size.** Sweep η over four values at γ=0 and confirm no
   step size stabilizes the unpenalized game.
   *Provenance:* original.
1. [short-code] **Alternating Updates on the Dirac-GAN.** Modify the `trajectory` function
   from the phase-portrait cell to perform *alternating* rather than simultaneous gradient
   steps (update ψ from the current θ, then update θ from the new ψ) for γ=0 and for the
   two penalized values of γ already used in the text (0.3 and 1.0). Plot the resulting
   trajectories alongside the simultaneous ones and report whether alternation changes the
   qualitative outcome — divergence at γ=0, convergence at γ>0 — or only the rate.
   *Provenance:* adapted from Mescheder, Geiger & Nowozin (ICML 2018), who state that
   alternating updates also fail to converge for the unpenalized game (overlap medium —
   the section asserts this result in prose; the exercise asks the reader to verify it in
   the section's own code). The "toy game, simultaneous vs. sequential descent" framing
   echoes Goodfellow's NIPS 2016 Tutorial, Exercise 2 (overlap low).

---

## chapter_generative-adversarial-networks/dcgan.md — Adversarial Image Generation

**Topic:** Building a DCGAN-era image generator and a modern R3GAN-style minimal backbone,
comparing the classic loss (which collapses under a shared initialization) against the
penalized relativistic loss (which trains stably), and evaluating both with FID/KID from a
feature network trained inside the notebook.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — every exercise specifies
an exact protocol, dataset, and comparison; Exercise 2 is already explicitly adapted from
Chong & Forsyth (2020) in the section's own text, and the remaining five are equally
concrete (quantile-function FID derivation, cross-dataset feature-network ablation,
initialization ablation, memorization-check extensions).

**External sources found:**
- Heusel, Ramsauer, Unterthiner et al., "GANs Trained by a Two Time-Scale Update Rule
  Converge to a Local Nash Equilibrium" (arXiv:1706.08500) — the FID paper; confirmed,
  already correctly cited for :eqref:`eq_gan_fid`.
- Binkowski, Sutherland, Arbel & Gretton, "Demystifying MMD GANs" (arXiv:1801.01401) — the
  KID paper; confirmed, already correctly cited for :eqref:`eq_gan_kid`.
- Chong & Forsyth, "Effectively Unbiased FID and Inception Score and where to find them"
  (arXiv:1911.07023, CVPR 2020) — already the section's own citation and the direct source
  of Exercise 2's extrapolation-to-1/n→0 protocol; confirmed to match.
- University of Toronto CSC321 (Roger Grosse), Programming Assignment 4: DCGAN + CycleGAN,
  Winter 2018 — a verified DCGAN coding assignment (32×32 emoji generation) specifying
  exact conv/deconv architectures with stride-2, kernel-4 layers and asking students to
  derive the padding needed to exactly halve resolution — a clean warm-up this section's
  own architecture never poses as a question. Verified by direct PDF fetch of
  `cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf`.
- Stanford CS231n, Assignment 3, Question 5 ("Generative Adversarial Networks") —
  implements vanilla GAN, LSGAN, and DCGAN on MNIST in a graded notebook; existence and
  topic confirmed across multiple course years via the official assignment pages and
  mirrored student repos, though exact rubric text could not be retrieved, so it is noted
  here but not used as direct provenance for a new problem.
- MIT 6.S191 (suggested source), checked directly: the current public lab repository
  (`github.com/MITDeepLearning/introtodeeplearning`) pairs its generative-modeling
  *lecture* with a VAE-based facial-detection-debiasing *lab*, not a GAN-training lab; no
  dedicated DCGAN/GAN notebook exists in `lab1`/`lab2`/`lab3`/`xtra_labs`. A finding, not
  an omission: the suggested source lacks the exercise tradition expected of it.

**Proposed problem set** (7 problems):
1. [conceptual] **FID for Univariate Gaussians.** Derive the 1-D Wasserstein-2
   quantile-function formula and show it reduces to (μ_p−μ_q)²+(σ_p−σ_q)² before checking
   the commuting-covariance multivariate case.
   *Provenance:* original.
1. [short-code] **Finite-Sample Bias in FID/KID.** Split the held-out sprites and measure
   how the real-vs-real FID/KID floor moves with n, extrapolating to 1/n→0.
   *Provenance:* adapted from Chong & Forsyth, "Effectively Unbiased FID and Inception
   Score" (CVPR 2020) — already the section's own cited source (overlap high; already
   adopted).
1. [short-code] **Feature Networks and Ranking Stability.** Retrain the feature CNN on
   Fashion-MNIST instead of CIFAR-10 and check whether the FID/KID *values* and *ordering*
   of the two training arms survive the swap.
   *Provenance:* original.
1. [extended] **Modern Recipe on Fashion-MNIST.** Adapt the modern backbone to
   single-channel Fashion-MNIST, train once with the classic loss and once with
   RpGAN+R1+R2 at matched budget, and compare sample grids.
   *Provenance:* original.
1. [short-code] **Initialization and Mode Collapse.** Rerun the classic-loss arm with
   framework-default initialization instead of 𝒩(0,0.02²) and reconcile the result with
   the fixed-minima argument of :numref:`sec_gan_relativistic`.
   *Provenance:* original.
1. [short-code] **Screening for Generator Copying.** Extend the nearest-neighbor
   memorization check to admit mirrored training images as candidates, and repeat the
   search in raw pixel space instead of feature space.
   *Provenance:* original.
1. [conceptual] **Padding for Exact Halving.** For a convolution with kernel size K=4 and
   stride S=2 — the configuration used by this section's own `G_block`/`D_block` — derive
   the padding P that makes each layer exactly halve (discriminator) or double (generator)
   the spatial resolution, using the standard output-size formula ⌊(H+2P−K)/S⌋+1. Confirm
   your answer against the shapes printed by this section's own generator/discriminator
   parameter-count cell.
   *Provenance:* adapted from University of Toronto CSC321, Programming Assignment 4,
   "Implement the Discriminator of the DCGAN," Part 1 Question 1 (2018) (overlap high —
   same derivation, applied to this section's own layer configuration instead of the
   assignment's).

---

## chapter_generative-adversarial-networks/conditional.md — Conditional Generation

**Topic:** Conditioning both networks on a label, deriving the per-condition value
E_c[2JS(p(·|c),q(·|c))]−2log2 and the projection discriminator from a Bayes-rule
decomposition, training a class-conditional CIFAR-10 generator, and separating
fidelity/diversity/alignment as distinct evaluation axes; concluding with pix2pix/CycleGAN
as image-conditioned generation.

**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the longest and densest
exercise set in the chapter; every item names an exact deliverable (a derivation, a
numerical verification recipe, or a comparison with a stated budget), confirmed by the
prior style review to have no clarity issues despite legacy "1."-repeated numbering.

**External sources found:**
- Stanford CS236, Homework 3, Problem 3.1, 2018 — derives the optimal conditional critic's
  logit as a bilinear form h\*(x,y)=yᵀ(Aφ(x)+b) under a Gaussian-mixture-in-feature-space
  assumption — a different route to essentially the same projection-discriminator
  structure this section derives via Bayes' rule and shared softmax posteriors.
- Stanford CS236, Homework 3, Problem 3.2, 2018 — a matching coding task: implement and
  train a conditional GAN with this projection-style discriminator on Fashion-MNIST for one
  epoch.
- Miyato & Koyama, "cGANs with Projection Discriminator" (arXiv:1802.05637, ICLR 2018) —
  the paper this section's own :eqref:`eq_gan_cond_projection` is drawn from; confirmed,
  correctly cited.
- Brock, Donahue & Simonyan, "Large Scale GAN Training for High Fidelity Natural Image
  Synthesis" (BigGAN, arXiv:1809.11096) — source of the truncation trick already used in
  this section's Exercise 5; confirmed, correctly cited.
- University of Toronto CSC321 (Roger Grosse), Programming Assignment 4, Part 2: CycleGAN,
  Winter 2018 — a verified assignment training CycleGAN with and without the
  cycle-consistency loss and asking students to report and explain the qualitative
  difference — a direct hands-on complement to this section's own conceptual Exercise 7
  (which only asks students to classify losses and reason about the failure mode, not run
  it). Verified by direct PDF fetch (same document as dcgan.md's CSC321 source above).
- Prince, *Understanding Deep Learning*, ch. 15, Problem 15.6 (truncation-trick
  proportion-of-mass calculation) is already the source cited elsewhere in this catalog for
  this section's own Exercise 5; not re-listed here to avoid double-counting.

**Proposed problem set** (8 problems):
1. [conceptual] **Separate Feature Maps for Posteriors.** Determine the critic's form when
   the real/generated class posteriors require different feature maps, and whether a
   single shared-map projection head can still express it.
   *Provenance:* original.
1. [short-code] **Verifying the Per-Condition JS.** Compute E_c[2JS]−2log2 by quadrature on
   a two-class Gaussian toy and compare against a trained critic's converged per-sample
   loss.
   *Provenance:* original.
1. [conceptual] **Across-Slice Versus Within-Slice Imbalance.** Contrast how label-marginal
   imbalance (which only discounts the value) differs from within-game imbalance (which
   changes the optimal critic), and predict the effect on a rare class's alignment.
   *Provenance:* original.
1. [short-code] **Per-Class Distributional Diagnostics.** Extend the collapse diagnosis
   from per-class feature variance to per-class MMD or precision/recall, and compare the
   resulting class ranking to the alignment table's.
   *Provenance:* original.
1. [short-code] **Sweeping the Truncation Trick.** Implement latent truncation at several
   τ and report whether alignment rises and within-class variance falls as BigGAN found.
   *Provenance:* original.
1. [short-code] **Concatenation Versus Projection Critics.** Implement a
   concatenation-based critic, retrain both critics at a reduced budget, and compare
   alignment and feature distances.
   *Provenance:* original.
1. [conceptual] **Classifying Adversarial Losses in Translation.** List each system's
   (pix2pix, CycleGAN) adversarial losses, classify each critic as conditional or marginal,
   and explain how near-zero cycle losses can still permit an unintended pairing.
   *Provenance:* original.
1. [short-code] **Reproducing the Cycle-Consistency Ablation.** Using two label-defined
   CIFAR-10 subsets already available to this section's data pipeline (for instance, one
   vehicle class as domain X and a visually related one as domain Y), train a small
   CycleGAN-style pair of generators and marginal critics with and without the
   cycle-consistency loss, at a short, fixed iteration budget. For a handful of held-out
   source images, display both the direct translation and the round-trip reconstruction
   under each arm, and report whether removing the cycle loss produces translations that
   still fool the marginal critics while visibly failing to preserve the source image's
   identity.
   *Provenance:* adapted from University of Toronto CSC321, Programming Assignment 4,
   Part 2 ablation ("train the CycleGAN with/without the cycle-consistency loss … explain
   the difference") (overlap medium — same ablation logic and comparison, adapted from the
   assignment's emoji dataset to this section's own CIFAR-10 pipeline).

---

## chapter_generative-adversarial-networks/adversarial-losses.md — Adversarial Losses Beyond GANs

**Topic:** Why adversarial losses persist alongside diffusion/flow models — the capacity
argument that pointwise regression predicts a possibly-invalid conditional mean under
multimodal targets — surveyed across distillation, tokenizers, and vocoders, ending with
three responses to instability and a likelihood model as its own implicit discriminator.

**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — every exercise has a
stated deliverable and, where checkable, a comparison target; the style review found no
clarity issues.

**External sources found:**
- Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion Distillation" (ADD,
  arXiv:2311.17042) — source of the one-step distillation/adversarial/both FID ablation
  (315.6/20.8/20.6) already quoted verbatim in this section's text; confirmed.
- The DMD2, LADD, and ViTok/ViTok-v2 citations already used in the text were spot-checked
  for correctness of the qualitative claims attributed to them (adversarial-only training
  surpassing its teacher; dropping the adversarial loss at 5B-parameter scale) rather than
  searched fresh, since the text's own claims already carry full citations.
- No good external exercise tradition: no Stanford CS236, Berkeley CS294-158, or other
  course homework treats "pointwise regression vs. adversarial loss under a multimodal
  conditional target" as a standalone problem. The capacity argument is a
  research-literature synthesis (this section's own contribution, drawing on the ADD
  ablation and the SDXL-Lightning patch-FID observation) rather than an established
  homework topic, and the section's five exercises already constitute close to the entire
  available exercise tradition on this specific framing.
- University of Toronto CSC321's CycleGAN assignment (cited under conditional.md above)
  touches a related but distinct point — a reconstruction loss compensating for an
  underdetermined adversarial mapping — not reused here to avoid double-counting.

**Proposed problem set** (6 problems):
1. [conceptual] **Ranking Tasks by Ambiguity.** Rank super-resolution, class-conditional
   generation, and text rendering by how much an adversarial term should help a
   capacity-limited pointwise student, and check the ranking against the section's
   patch-FID evidence.
   *Provenance:* original.
1. [conceptual] **DDO as an Optimal Critic.** Show in three lines that DDO's
   D(x)=log(q_θ/q_ref) equals the optimal critic of :numref:`sec_basic_gan` once q_θ=p.
   *Provenance:* original.
1. [short-code] **A Three-Band Toy Regression.** Extend the mode offset to three unequally
   likely values and report what the squared-error student now predicts.
   *Provenance:* original.
1. [short-code] **Giving the Student a Latent Bit.** Add a latent input to the two-band
   student and compare which loss gives it an incentive to use that bit.
   *Provenance:* original.
1. [conceptual] **Costs of Three Responses.** State in one sentence each what regularized
   dynamics, frozen-feature critics, and fixed discrepancies/regression targets give up.
   *Provenance:* original.
1. [short-code] **A Three-Arm Capacity Ablation.** Add a third student to the two-band
   experiment that minimizes a weighted sum of the squared-error loss and the
   non-saturating adversarial loss (reusing both training loops already defined in this
   section), sweeping the squared-error weight over a few values (e.g. 0, 0.1, 1, 10). For
   each arm, report the section's own two summary distances (mean distance to the nearest
   mode, mean distance to the conditional mean), and compare the qualitative pattern to
   ADD's one-step ablation (distillation-only / adversarial-only / both), where adding the
   pointwise term barely changes FID once the adversarial term is present.
   *Provenance:* adapted from Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion
   Distillation" (arXiv:2311.17042) (overlap medium — same three-arm ablation logic,
   applied to this section's own toy regression problem rather than image FID).
