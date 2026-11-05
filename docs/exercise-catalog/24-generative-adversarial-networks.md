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
62 exercises (2026-11 tightening; 8 to 10 per section: gan 10,
objectives 9, relativistic 8, convergence 9, dcgan 10, conditional 8,
adversarial-losses 8) carry `:exlabel:` ids, listed per section below in source order;
external material is used almost entirely to add depth rather than replace anything.

---

## chapter_generative-adversarial-networks/gan.md — Generative Adversarial Networks

**Topic:** The original GAN minimax objective: the log-loss game, its optimal discriminator
(the log density ratio), the value at optimum (Jensen–Shannon divergence), and the
saturating-vs-non-saturating generator loss, verified end-to-end on a 2-D Gaussian.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [conceptual] **Likelihood versus samples** (`ex_gan-likelihood-versus-samples`). Generalize the noise-mixture bound to weights eps and 1-eps, write the shortfall exactly as an expectation of a function of p_noise/p_data, show it is at most log(1/eps), and convert it to bits per dimension for 32x32 color images (would a two-decimal table show it?).
   *Provenance:* original (carried over from "Likelihood Under a Noise Mixture", extended with the bits-per-dimension and shrunk-Gaussian parts).
1. [conceptual] **Excess loss and the additive constant** (`ex_gan-log-loss-game-identifies`). Prove V(D*)-V(D)=2E_m[Bernoulli-KL(eta||sigma(D))], conclude V(D*+b)<V(D*) for b!=0, and expand to second order in b to get the -b^2/2 * int pq/(p+q) coefficient (in [0,1/2]) and what it says about how sharply finite samples fix the constant.
   *Provenance:* original (carried over from "Shift Invariance of the Critic", extended with the exact excess-loss identity).
1. [conceptual] **Unequal priors** (`ex_gan-unequal-priors`). Derive D*_alpha = lambda + log(alpha/(1-alpha)) and show max V_alpha + H_b(alpha) is the label mutual information; relate unequal real/generated batch sizes to alpha, the best response and the non-saturating weight, and ask whether q=p stays stationary.
   *Provenance:* original (carried over from "Unequal Priors, Skewed Divergence").
1. [conceptual] **Supports of measure zero** (`ex_gan-measure-zero-supports`). Show JS=log 2 for a line-supported generator and for the empirical distribution of the training points, then use Kantorovich-Rubinstein duality to bound the value change of a kappa-Lipschitz critic by kappa*W1(p-hat,p) and say why the restricted game stays informative.
   *Provenance:* book-original.
1. [short-code] **A dead critic** (`ex_gan-dead-critic`). Give the critic tanh units (hidden layers of 5 and 3), rerun the far-start non-saturating arm over five seeds, inspect hidden activations and input gradients of a stalled run, and decide which reading of a critic loss of log 2 (best response to q=p, or a critic that cannot separate the samples) fits.
   *Provenance:* book-original.
1. [short-code] **Two time scales** (`ex_gan-two-time-scales`). Rerun the Gaussian training for three (lr_D, lr_G) settings over five seeds, report median and range of the final KL, compare spread with between-setting differences, and say against which critic the generator optimizes when it adapts faster.
   *Provenance:* book-original.
1. [short-code] **Classifier two-sample test** (`ex_gan-classifier-two-sample-test`). Show the Bayes classifier accuracy is (1+TV)/2, bound the accuracy against the trained generator with Pinsker and the printed KL, measure a fresh critic on 2,000 held-out samples per class, and find the test size that rejects q=p at the 5 percent level.
   *Provenance:* book-original.
1. [short-code] **Class coverage of the garment generator** (`ex_gan-class-coverage`). Label 1,000 generated Fashion-MNIST samples with a trained LeNet, compare the class histogram with the uniform one, repeat for generators trained 5 and 40 epochs, and discuss how forced class assignment biases the histogram and how it relates to mode dropping.
   *Provenance:* book-original.
1. [conceptual] **Generator weights and their divergences** (`ex_gan-generator-weight-divergences`). Show saturating plus non-saturating loss equals KL(q||p) at the best-response critic (weight 1), derive the weight-to-f-divergence map rho^2 f''(rho)=w(log rho), identify the divergences of the standard weights, and show the non-saturating f is strictly convex with f(1)=0.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problem 2.3 (2018) (overlap high on the sum-of-losses identity, extended to the general weight-to-divergence correspondence); carried over from "Exact KL from Combined Losses" and "Two Generator Weights".
1. [extended] **Finite samples and critic capacity** (`ex_gan-finite-sample-critic`). Train critics of widths 4 to 1024 on n=100 to 10000 points against q=p, plot the train/fresh loss gap against parameter count, place the 337-parameter critic of make_nets, and apply the result to the roughly 530,000-parameter garment critic on 60,000 images.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/objectives.md — Adversarial Objectives and Divergences

**Topic:** Generalizing the log-loss game along two axes — the classification loss (giving
f-divergences via the Bayes-risk gap, and the f-GAN duality) and the critic class (giving
integral probability metrics: MMD and Wasserstein-1) — and comparing which retain a
gradient once supports separate.

**Current exercises:** 9; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (9 problems):
1. [short-code] **Convexity and parameterization** (`ex_objectives-convexity-parameterization`). Check by second differences that JS is convex along a mixture path, plot JS in a location parameter for an unbalanced two-mode target to find local minima, and say why this does not contradict convexity in q.
   *Provenance:* book-original.
1. [short-code] **More rows of the table** (`ex_objectives-verify-numerically-that-lsgan`). Derive the Brier loss Bayes risk eta(1-eta) and its gap (1/8 int (p-q)^2/(p+q), factor confirmed on Dirichlet draws), then the exponential and zero-one rows (Hellinger and TV/2 gaps, optimal exponential critic lambda/2).
   *Provenance:* original (carried over from "Brier Loss's Bayes Risk", "Zero-One Loss and Total Variation" and "LSGAN's Triangular Discrimination").
1. [conceptual] **Jensen-Shannon conjugate** (`ex_objectives-variational-bound-eqref-eq`). Derive the conjugate f*(t)=-log(2-e^t) of the JS generator and its domain, show T=log(2 sigma(D)) gives D*=lambda and the bound 2 log 2 + V(D), and examine what the -softplus activation excludes.
   *Provenance:* original (carried over from "f-GAN Duality for JS").
1. [short-code] **MMD gradient range** (`ex_objectives-mmd-gradient-range`). Differentiate the Gaussian MMD^2 in the separation and show the slope peaks at sqrt(s^2+2); plot the slope for the three-bandwidth kernel of mmd2_loss, find where it falls below 1% of its maximum, and compare with a single kernel s=4.
   *Provenance:* book-original.
1. [conceptual] **Lipschitz hinge critics** (`ex_objectives-hinge-lipschitz`). Show the hinge game equals E_p[D]-E_q[D]-2 for |D|<=1, that clipping never lowers V, hence the unconstrained game is the total-variation IPM shifted by -2, and how restricting to kappa-Lipschitz critics changes the value.
   *Provenance:* book-original.
1. [short-code] **Plateaus and critic lag** (`ex_objectives-plateaus-lag`). Compare a logistic critic with generator loss -D against the non-saturating entry over three seeds by first step with JS<0.3, explain via per-sample weights, then test critic lag against minibatch noise with a 4x critic batch or five critic steps.
   *Provenance:* book-original.
1. [short-code] **Estimation and dimension** (`ex_objectives-estimation-and-dimension`). Measure empirical W1 (optimal assignment) and unbiased MMD^2 estimation error against n and dimension on shifted Gaussians and compare slopes with the quoted rates; interpret W1 between two samples of the same Gaussian in dimension 50.
   *Provenance:* book-original.
1. [conceptual] **Divergences a loss can reach** (`ex_objectives-divergences-a-loss-can-reach`). Prove the symmetry-to-loss direction of Nguyen-Wainwright-Jordan Corollary 3: from a symmetric f-divergence build a concave, symmetric, nonnegative Bayes risk L and the margin loss reaching D_f.
   *Provenance:* book-original.
1. [extended] **Separated initializations** (`ex_objectives-separated-initializations`). Train logistic, least-squares, hinge (plain and spectrally normalized) and MMD games on a one-dimensional problem from separated starts mu_0 in {2,5,10,20}, tabulate steps to JS<0.01 over three seeds, and explain which runs stall and why.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/relativistic.md — Relativistic Objectives

**Topic:** The relativistic pairing objective (RpGAN): scoring a real–fake pair by the
difference of critic scores, its shift-invariance, its reduction (by "lifting") to the
classical log-loss game on the product space, the value JS(p⊗q, q⊗p), and how the
rank-based generator weight it induces removes many mode-dropping minima.

**Current exercises:** 8; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (8 problems):
1. [conceptual] **Shift invariance in training** (`ex_relativistic-prove-shift-invariance-ev`). Show the functional derivative integrates to zero so the critic mean is conserved in the ascent loop, that a scalar output bias receives no gradient, what then sets the score level, and how a classical critic's D>0 fraction behaves.
   *Provenance:* original (carried over from "Shift Invariance, Proved" and "Shift Invariance, Checked Numerically").
1. [conceptual] **Local expansion** (`ex_relativistic-local-expansion-let-e`). Expand the relativistic divergence and JS to second order under q=p(1+eps v), giving eps^2/4 E v^2 and eps^2/8 E v^2, and confirm the locally factor-of-two relation d_Rp = 2 JS.
   *Provenance:* original (carried over from "Local Expansion to Second Order").
1. [conceptual] **Two Gaussians** (`ex_relativistic-two-gaussians`). For translated unit Gaussians, show the pair score is Gaussian, write JS in the same form and find the c with JS(p,N(c mu,I)) = d_Rp, and evaluate both by quadrature to relate the ratio to the local limit and to saturation.
   *Provenance:* book-original.
1. [short-code] **Zero-sum and non-saturating weights** (`ex_relativistic-zero-sum-weights`). Evaluate the zero-sum and non-saturating generator weights on the five atoms with the recovered critic, compare where each places its largest weight and the factor between their q-averages, and prove the q-average of w_Rp equals int PQ/(P+Q) and is at most 1/2.
   *Provenance:* book-original.
1. [conceptual] **Separability of the pair critic** (`ex_relativistic-separability-choice-loss-replace`). For a general concave payoff, show restricting the pair critic to differences costs nothing for the logistic payoff but, for least squares, the pair-space optimum tanh((lambda(a1)-lambda(a2))/2) is not additively separable.
   *Provenance:* original (carried over from "General Concave Payoffs").
1. [conceptual] **More negatives** (`ex_relativistic-more-negatives`). Generalize to K real samples and one generated sample in random order: softmax posterior, the K+1-hypothesis lifting argument with divergence d_K bounded by log(K+1), and the InfoNCE connection.
   *Provenance:* original (carried over from "From Pairs to InfoNCE").
1. [conceptual] **Gap to twice Jensen-Shannon** (`ex_relativistic-gap-twice-js`). Prove d_Rp = 2 JS - I(a1;a2) via conditional independence and the chain rule, get JS <= d_Rp from data processing, and check the identity on the five-atom example from the 25-atom distribution.
   *Provenance:* book-original.
1. [short-code] **Random re-pairing** (`ex_relativistic-random-repairing`). Extend the bad-minima enumeration to n=4 (256 assignments), then model a critic that best-responds to each random pairing and count suboptimal strict local minima of the permutation-averaged loss for n=3 and trace n=2.
   *Provenance:* book-original (extends the enumeration of the section, which follows Sun, Fang & Schwing, NeurIPS 2020; carried over from "Counting Mode-Dropping Minima" in spirit).

---

## chapter_generative-adversarial-networks/convergence.md — Gradient Penalties and Convergence

**Topic:** Why a correct equilibrium doesn't imply convergent training — the Dirac-GAN's
circular continuous-time dynamics vs. outward-spiraling discrete updates — and how
zero-centered gradient penalties R1/R2 restore local convergence and (near equilibrium)
measure a linearized Wasserstein-2 distance; tested via the R3GAN recipe on 25 Gaussians.

**Current exercises:** 9; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (9 problems):
1. [short-code] **Discrete-time stability** (`ex_convergence-discrete-simultaneous-update-dirac`). Show the simultaneous Dirac-GAN map has eigenvalues 1 +- i h l'(0) at gamma=0 (no step size converges), prove convergence iff h < gamma/l'(0)^2 in the underdamped regime, and check the threshold at gamma=0.3 by trajectories.
   *Provenance:* original (carried over from "Eigenvalues of the Discrete Map" and "Sweeping the Step Size").
1. [conceptual] **Penalized best response** (`ex_convergence-penalized-best-response`). Show the penalized best response for the Dirac-GAN has maximizer psi* ~ -l'(0) theta/(2 gamma) and value 2 l(0) + l'(0)^2 theta^2/(4 gamma), compare with the Sobolev-norm view, and find the generator's gradient near theta=0.
   *Provenance:* original (carried over from "A Sobolev-Norm Interpretation" and "Mixture Weighting in the Penalty").
1. [conceptual] **Penalizing one side only** (`ex_convergence-one-sided-penalty`). With critics D_c(x)=-c x^2, show a one-sided R1 penalty leaves the critic objective unbounded and the non-saturating gradient diverging in c (Huang et al. 2024), and that subtracting R1+R2 gives a finite c* < 1/(8 gamma).
   *Provenance:* book-original.
1. [short-code] **The penalty weight as a length** (`ex_convergence-penalty-length`). Show the penalty weight scales as a squared length under data rescaling, compute the damping of a mode-width component at gamma=1 and 10, and rerun the gamma=10 25-Gaussians training for two and four times as many steps.
   *Provenance:* book-original.
1. [short-code] **Alternating updates** (`ex_convergence-alternating-updates`). Linearize the alternating Dirac-GAN update (determinant one, eigenvalues on the unit circle iff h l'(0) <= 2) and test by a 10^5-step simulation at h in {0.2,1,3,5}.
   *Provenance:* adapted from Mescheder, Geiger & Nowozin (ICML 2018), who state that alternating updates also fail to converge for the unpenalized game (carried over from "Alternating Updates on the Dirac-GAN").
1. [short-code] **The local value** (`ex_convergence-prove-eqref-eq-gan`). Derive the first-order condition (|l''(0)| m + gamma L_m) D = l'(0)(p-q)/2 for the maximizing critic, then compare penalized_value ratios across omega and gamma with the low-pass weight 1/(1+4 gamma omega^2) and find where it fails.
   *Provenance:* book-original.
1. [short-code] **One-centered penalty dynamics** (`ex_convergence-one-centered-dynamics`). Analyze the WGAN-GP (one-centered penalty) Dirac-GAN flow: no stationary point with psi!=0, one-sided limits at psi->0, the affine Jacobian vs the penalized one, and trajectories at gamma=1 and 3.
   *Provenance:* book-original.
1. [short-code] **Momentum in a rotating game** (`ex_convergence-momentum-rotating-game`). Linearize the penalized Dirac-GAN with beta-averaged momentum as a 4-dimensional map, compute the largest stable step size for beta in {0,0.5,0.9} at gamma=0.3 and 1, and relate it to R3GAN's Adam beta1=0.
   *Provenance:* book-original.
1. [short-code] **Non-saturating Dirac dynamics** (`ex_convergence-non-saturating-dynamics`). Analyze the non-saturating Dirac flow: r^2 is non-increasing so the approach cannot be linear, r(t)^2 ~ 8/t from the turn-averaged rate, the discrete cycle at (r*)^2 = 2h and its stability, and a simulation.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/dcgan.md — Adversarial Image Generation

**Topic:** Building a DCGAN-era image generator and a modern R3GAN-style minimal backbone,
comparing the classic loss (which collapses under a shared initialization) against the
penalized relativistic loss (which trains stably), and evaluating both with FID/KID from a
feature network trained inside the notebook.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [conceptual] **Checkerboards from stride** (`ex_dcgan-checkerboards-from-stride`). Find every kernel size (at most 6) and padding for which a stride-2 transposed convolution exactly doubles the side, count input positions per output for kernels 3 and 4 (uneven overlap), and explain why bilinear upsampling plus a 3x3 convolution cannot create a periodic pattern.
   *Provenance:* adapted from University of Toronto CSC321, Programming Assignment 4, "Implement the Discriminator of the DCGAN," Part 1 Question 1 (2018) (carried over from "Padding for Exact Halving", extended with the checkerboard counting).
1. [conceptual] **Penalty weight and resolution** (`ex_dcgan-penalty-weight-resolution`). Show how the penalty weight must rescale under pixel upsampling and under a change of pixel range, and which factor of the Karras et al. gamma = 0.0002 r^2/B rule the argument explains.
   *Provenance:* book-original.
1. [short-code] **A scale-free overfitting statistic** (`ex_dcgan-scale-free-overfitting-statistic`). Show r_v of Karras et al. 2020 is invariant under D -> aD + c while the gap and r_t are not, find which transformations leave E[sign(D_train)] meaningful for a pairing critic, and compute r_v and r_t for the three arms.
   *Provenance:* book-original.
1. [conceptual] **FID across dimensions** (`ex_dcgan-one-dimension-squared-wasserstein`). Derive the one-dimensional squared Wasserstein-2 for Gaussians, evaluate FID for scaled and shifted Gaussians in d dimensions, and compare a 10% spread error in 128 versus 2,048 dimensions.
   *Provenance:* original (carried over from "FID for Univariate Gaussians").
1. [short-code] **Finite-sample bias** (`ex_dcgan-measure-finite-sample-bias`). Measure real-versus-real FD and MMD^2 against n (five splits each), fit FD against 1/n and extrapolate, and repeat for the RpGAN + R1 + R2 generator to see whether rankings at finite n survive as n -> infinity.
   *Provenance:* adapted from Chong & Forsyth, "Effectively Unbiased FID and Inception Score" (CVPR 2020) (overlap high; carried over from "Finite-Sample Bias in FID/KID").
1. [short-code] **Precision and recall** (`ex_dcgan-precision-recall`). Implement k-nearest-neighbor ball precision and recall (Kynkaanniemi et al. 2019) on the CNN features for every generator and for training images, vary K, and find the largest number of collapsed images J for which recall is trivial.
   *Provenance:* book-original.
1. [short-code] **Initialization and collapse** (`ex_dcgan-rerun-arm-classic-loss`). Define a feature-space diversity measure, record it every 500 steps for the unpenalized arm under fan-in and normal initializations (std 0.01, 0.02, 0.05), and quantify the seed variation of the collapse step.
   *Provenance:* original (the earlier "Initialization and Mode Collapse"; rewritten with the diversity statistic).
1. [conceptual] **Moments behind FID and KID** (`ex_dcgan-moments-fid-kid`). Show FID sees only means and covariances, expand the cubic KID kernel to show which moments it compares, compute FID and MMD^2 on one-dimensional distributions differing in higher moments, and ask whether the KID kernel is characteristic.
   *Provenance:* book-original.
1. [short-code] **Checking for copies** (`ex_dcgan-checking-for-copies`). Find each sample's nearest training image in feature space, compare median nearest-neighbor distances with held-out sprites, inspect the closest pairs, and discuss whether the test detects recombined memorized parts.
   *Provenance:* book-original.
1. [conceptual] **Leaking augmentations** (`ex_dcgan-augmentation-leaks`). Show that augmenting only real images makes the optimal generator match the flipped mixture (fraction facing right), and prove that differentiable augmentation of both sides identifies q=p for flip probability rho != 1/2 but not at 1/2.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/conditional.md — Conditional Generation

**Topic:** Conditioning both networks on a label, deriving the per-condition value
E_c[2JS(p(·|c),q(·|c))]−2log2 and the projection discriminator from a Bayes-rule
decomposition, training a class-conditional CIFAR-10 generator, and separating
fidelity/diversity/alignment as distinct evaluation axes; concluding with pix2pix/CycleGAN
as image-conditioned generation.

**Current exercises:** 8; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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
1. [short-code] **Two slices with known densities** (`ex_conditional-verify-decomposition-eqref-eq`). Compute the conditional game value by quadrature on two Gaussian slices, check it against the loss of a small trained critic, and repeat with swapped labels.
   *Provenance:* original (carried over from "Verifying the Per-Condition JS").
1. [conceptual] **Separate feature maps** (`ex_conditional-projection-derivation-modeled-both`). Derive the critic when the two class posteriors need different feature maps (a projection head over concatenated features), show K dimensions suffice to represent any D(x,c), and say when the projection form restricts the critic.
   *Provenance:* original (carried over from "Separate Feature Maps for Posteriors").
1. [short-code] **The critic as a classifier** (`ex_conditional-critic-classifier`). Express the optimal critic score differences through the real and generated label posteriors, then measure the critic-as-classifier accuracy on test images with and without per-class offsets and explain why it falls toward chance as the generator improves.
   *Provenance:* book-original.
1. [short-code] **Per-class distances** (`ex_conditional-diagnostics-cell-detects-within`). Discuss the rank problem of a 100-sample covariance for per-class FD, compute per-class MMD^2 (generated vs real, real vs real), compare its ranking with alignment, and classify the failure modes of a conditional generator.
   *Provenance:* original (carried over from "Per-Class Distributional Diagnostics").
1. [short-code] **Truncated latents** (`ex_conditional-biggan-s-truncation-trick`). Compute the in-box probability and norm of truncated latents and sweep the truncation threshold on the EMA generator, reporting alignment, within-class variance and Frechet distance.
   *Provenance:* adapted from Simon Prince, Understanding Deep Learning, Problem 15.6 (carried over from "Sweeping the Truncation Trick").
1. [short-code] **A rare class** (`ex_conditional-rare-class`). Bound the value cost of ignoring a rare slice (2 eps log 2), then train with natural versus class-balanced real batches on a 50-image class against a full-set reference and report per-class metrics.
   *Provenance:* original (carried over from "Across-Slice Versus Within-Slice Imbalance").
1. [conceptual] **Auxiliary classifier bias** (`ex_conditional-auxiliary-classifier-bias`). Show the auxiliary classification loss is entropy plus KL, and that minimizing the entropy term pushes the generator toward easy-to-classify samples.
   *Provenance:* book-original.
1. [short-code] **The critic at a fixed budget** (`ex_conditional-implement-concatenation-critic-one`). Sweep the penalty weight at a fixed 4,000-step budget and explain the CIFAR-10 versus sprite scale, then replace the projection head by a concatenation critic and compare over two seeds.
   *Provenance:* original (carried over from "Concatenation Versus Projection Critics").

---

## chapter_generative-adversarial-networks/adversarial-losses.md — Adversarial Losses Beyond GANs

**Topic:** Why adversarial losses persist alongside diffusion/flow models — the capacity
argument that pointwise regression predicts a possibly-invalid conditional mean under
multimodal targets — surveyed across distillation, tokenizers, and vocoders, ending with
three responses to instability and a likelihood model as its own implicit discriminator.

**Current exercises:** 8; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (8 problems):
1. [conceptual] **Distortion of a perfect sampler** (`ex_adversarial-losses-perfect-sampler`). Show a perfect conditional sampler has twice the minimum squared error, convert to a PSNR loss in decibels, and rank the exact sampler, the conditional mean and the band on the two-band data (Blau-Michaeli trade-off).
   *Provenance:* book-original.
1. [short-code] **Mean and median** (`ex_adversarial-losses-mean-and-median`). Find conditional mean and median for three-offset and two-band variants (including P(b=+1)=3/4 and 1/2), say whether each is a valid output, and train the squared-error student.
   *Provenance:* original (carried over from "A Three-Band Toy Regression").
1. [short-code] **Pointwise plus adversarial** (`ex_adversarial-losses-pointwise-plus-adversarial`). Add kappa times squared error to the adversarial student, sweep kappa over {0.1,1,10,100} with two seeds, and analyze ADD's noised-teacher target.
   *Provenance:* adapted from Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion Distillation" (arXiv:2311.17042) (carried over from "A Three-Arm Capacity Ablation").
1. [conceptual] **Phase and the conditional mean** (`ex_adversarial-losses-phase`). Show the conditional mean of a frame with uniform phase shrinks the amplitude by sin a / a, that DFT magnitudes do not depend on phase so magnitude losses do not select a valid waveform, and analyze concatenated frames.
   *Provenance:* book-original.
1. [conceptual] **Frozen critic features** (`ex_adversarial-losses-frozen-features`). Derive the optimal head and the divergence measured by a critic that sees only frozen features, which generators minimize it (a minimizer q != p for phi(x)=x_1), and what follows when the generator is also restricted.
   *Provenance:* book-original.
1. [conceptual] **DDO's implicit critic** (`ex_adversarial-losses-ddo-plays-log-loss`). Show DDO's parameterized critic is optimal if q_theta=p and that its gradient at theta = reference is half the negative log-likelihood gradient.
   *Provenance:* original (carried over from "DDO as an Optimal Critic").
1. [short-code] **A latent bit** (`ex_adversarial-losses-latent-bit`). Give the student a binary latent, compare squared-error and adversarial training, test the IMLE nearest-sample loss, and ask which band frequencies a uniform binary latent can represent for P(b=+1)=3/4.
   *Provenance:* original (carried over from "Giving the Student a Latent Bit").
1. [short-code] **Multi-round refinement** (`ex_adversarial-losses-multi-round-refinement`). Derive the minimizer of the practical DDO loss over all densities (proportional to q_ref^(1-1/beta) p^(1/beta), normalized for one alpha) and the recursion for r_n = log(q_n/p) across rounds.
   *Provenance:* book-original.
