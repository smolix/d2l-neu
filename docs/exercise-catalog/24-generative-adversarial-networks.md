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
68 exercises (2026-11 revision; 10 per section except 8 in
adversarial-losses.md) carry `:exlabel:` ids, listed per section below in source order;
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
1. [conceptual] **Likelihood versus samples** (`ex_gan-likelihood-versus-samples`). Generalize the noise-mixture bound to weights eps and 1-eps, write the shortfall exactly, convert it to bits per dimension for 32x32x3 images, and compare against a contrast-shrunk Gaussian model with the same likelihood gap.
   *Provenance:* original (carried over from "Likelihood Under a Noise Mixture", extended with the bits-per-dimension and shrunk-Gaussian parts).
1. [conceptual] **Excess loss and the additive constant** (`ex_gan-log-loss-game-identifies`). Prove V(D*)-V(D)=2E_m[Bernoulli-KL(eta||sigma(D))], so D* is the unique maximizer; apply it to D*+b and expand to second order to get the -b^2/2 * int pq/(p+q) coefficient and what it says about finite-sample determination of the constant.
   *Provenance:* original (carried over from "Shift Invariance of the Critic", extended with the exact excess-loss identity).
1. [conceptual] **Unequal priors** (`ex_gan-unequal-priors`). Derive the optimal critic D*_alpha = lambda + log(alpha/(1-alpha)) and the skewed-divergence value for P(y=1)=alpha; relate unequal real/generated batch sizes to alpha and to the non-saturating weight.
   *Provenance:* original (carried over from "Unequal Priors, Skewed Divergence").
1. [conceptual] **Supports of measure zero** (`ex_gan-measure-zero-supports`). Show JS=log 2 for a line-supported generator and for the empirical distribution of the training points, then use Kantorovich-Rubinstein duality to bound the value gap of Lipschitz critics by L*W1(p-hat,p).
   *Provenance:* book-original.
1. [short-code] **Critic error over the plane** (`ex_gan-verification-above-compared-critic`). Evaluate |sigma(D)-eta| on a grid, report the largest error and its location, and estimate the m-weighted error and the excess V(D*)-V(D) to explain why the largest grid errors cost the critic almost nothing.
   *Provenance:* original (carried over from "Critic Error Across a Grid", extended with the excess-loss estimate).
1. [short-code] **A dead critic** (`ex_gan-dead-critic`). Swap in a small tanh critic, rerun the far-start non-saturating arm over five seeds, inspect saturated hidden units and input gradients, and explain why a stalled run reports loss log 2 and how to tell a constant critic from a best response against q=p.
   *Provenance:* book-original.
1. [short-code] **Two time scales** (`ex_gan-two-time-scales`). Compare (lr_D, lr_G) settings over five seeds by median and range of the final KL, and check which conditions of the two time-scale convergence theorem the constant learning rates violate.
   *Provenance:* book-original.
1. [short-code] **Classifier two-sample test** (`ex_gan-classifier-two-sample-test`). Show the Bayes classifier accuracy equals (1+TV)/2, bound it with Pinsker, estimate it by Monte Carlo against a trained critic, and find the test size needed to reject q=p at the 5 percent level.
   *Provenance:* book-original.
1. [conceptual] **Generator weights and their divergences** (`ex_gan-generator-weight-divergences`). Show saturating plus non-saturating loss equals KL(q||p) at the best-response critic, derive the weight-to-f-divergence map rho^2 f''(rho)=w(log rho), identify the divergences of the four weights, and show the non-saturating f is strictly convex with f(1)=0.
   *Provenance:* adapted from Stanford CS236 Homework 3, Problem 2.3 (2018) (overlap high on the sum-of-losses identity, extended to the general weight-to-divergence correspondence); carried over from "Exact KL from Combined Losses" and "Two Generator Weights".
1. [extended] **Finite samples and critic capacity** (`ex_gan-finite-sample-critic`). Train critics of widths 4 to 1024 on n=100 to 10000 points against q=p, plot the train/fresh loss gap against parameter count, locate the ratio where the loss falls below log 2, and say what a critic loss below log 2 can no longer tell apart.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/objectives.md — Adversarial Objectives and Divergences

**Topic:** Generalizing the log-loss game along two axes — the classification loss (giving
f-divergences via the Bayes-risk gap, and the f-GAN duality) and the critic class (giving
integral probability metrics: MMD and Wasserstein-1) — and comparing which retain a
gradient once supports separate.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [short-code] **Convexity and parameterization** (`ex_objectives-convexity-parameterization`). Check by second differences that JS is convex along a mixture path, then show the loss is non-convex in a location parameter for an unbalanced two-mode target by locating local minima.
   *Provenance:* book-original.
1. [short-code] **More rows of the table** (`ex_objectives-verify-numerically-that-lsgan`). Derive the Brier loss Bayes risk and its divergence (factor confirmed on Dirichlet draws), then the exponential and zero-one rows, including the LSGAN triangular-discrimination check.
   *Provenance:* original (carried over from "Brier Loss's Bayes Risk", "Zero-One Loss and Total Variation" and "LSGAN's Triangular Discrimination").
1. [conceptual] **Symmetric gaps** (`ex_objectives-symmetric-gaps`). Show every margin-game Bayes risk is symmetric so the value is a symmetric f-divergence, which excludes forward and reverse KL, and state what change would admit asymmetric divergences.
   *Provenance:* book-original.
1. [conceptual] **Jensen-Shannon conjugate** (`ex_objectives-variational-bound-eqref-eq`). Derive the conjugate of the JS generator, its domain, the optimal critic and the reparameterization recovering the original GAN value function.
   *Provenance:* original (carried over from "f-GAN Duality for JS").
1. [short-code] **MMD gradient range** (`ex_objectives-mmd-gradient-range`). Differentiate the Gaussian MMD^2 in the separation, find where the gradient peaks and how fast it decays, and compare the summed multi-bandwidth kernel of the testbed with a single kernel.
   *Provenance:* book-original.
1. [conceptual] **Lipschitz hinge critics** (`ex_objectives-hinge-lipschitz`). Show the unconstrained hinge game is the total-variation IPM and saturates, and how a Lipschitz (spectrally normalized) critic changes this.
   *Provenance:* book-original.
1. [short-code] **Plateaus and critic lag** (`ex_objectives-plateaus-lag`). Add a logistic-critic entry with a different generator loss, compare against the non-saturating run over three seeds by first step below a divergence threshold, and explain the lag via per-sample weights and critic scores.
   *Provenance:* book-original.
1. [short-code] **Estimation and dimension** (`ex_objectives-estimation-and-dimension`). Measure empirical W1 (optimal assignment) and MMD estimation error against sample size and dimension on shifted Gaussians and compare with the quoted rates.
   *Provenance:* book-original.
1. [conceptual] **Divergences a loss can reach** (`ex_objectives-divergences-a-loss-can-reach`). Prove the symmetry-to-loss direction of Nguyen-Wainwright-Jordan Corollary 3 by constructing a nonnegative margin loss for a symmetric f-divergence.
   *Provenance:* book-original.
1. [extended] **Separated initializations** (`ex_objectives-separated-initializations`). Train logistic, least-squares and hinge (spectrally normalized) games on a one-dimensional problem started from separated supports and compare which recover.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/relativistic.md — Relativistic Objectives

**Topic:** The relativistic pairing objective (RpGAN): scoring a real–fake pair by the
difference of critic scores, its shift-invariance, its reduction (by "lifting") to the
classical log-loss game on the product space, the value JS(p⊗q, q⊗p), and how the
rank-based generator weight it induces removes many mode-dropping minima.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [conceptual] **Shift invariance in training** (`ex_relativistic-prove-shift-invariance-ev`). Show the functional derivative integrates to zero, so the critic mean is conserved in the ascent loop, and that a scalar output bias receives no gradient.
   *Provenance:* original (carried over from "Shift Invariance, Proved" and "Shift Invariance, Checked Numerically").
1. [conceptual] **Local expansion** (`ex_relativistic-local-expansion-let-e`). Expand the relativistic divergence and JS to second order under q=p(1+eps v) and confirm the factor of two, including the entropy form.
   *Provenance:* original (carried over from "Local Expansion to Second Order").
1. [conceptual] **Two Gaussians** (`ex_relativistic-two-gaussians`). For translated unit Gaussians, show the pair score is Gaussian, express the divergence in closed form, and evaluate it against JS by quadrature to relate the ratio to the local limit and to saturation.
   *Provenance:* book-original.
1. [short-code] **Zero-sum and non-saturating weights** (`ex_relativistic-zero-sum-weights`). Evaluate the zero-sum and non-saturating generator weights on the five atoms with the recovered critic and compare where each places its largest weight.
   *Provenance:* book-original.
1. [short-code] **Atoms outside the generator's support** (`ex_relativistic-atoms-outside-support`). Move generator atoms off the common support, check which critic scores stay finite, and track the ascent loop over increasing iteration counts.
   *Provenance:* book-original.
1. [conceptual] **Separability of the pair critic** (`ex_relativistic-separability-choice-loss-replace`). For a general concave payoff, show restricting the pair critic to differences costs nothing for the logistic payoff but strictly loses for least squares.
   *Provenance:* original (carried over from "General Concave Payoffs").
1. [conceptual] **More negatives** (`ex_relativistic-more-negatives`). Generalize to K real samples and one generated sample in random order: softmax posterior, generalized Jensen-Shannon divergence, and the InfoNCE connection.
   *Provenance:* original (carried over from "From Pairs to InfoNCE").
1. [conceptual] **Gap to twice Jensen-Shannon** (`ex_relativistic-gap-twice-js`). Prove the mutual-information bounds on the relativistic divergence with the chain rule and data processing, and check them on the five-atom example.
   *Provenance:* book-original.
1. [short-code] **Diagonal versus all-pairs estimates** (`ex_relativistic-diagonal-all-pairs`). Derive the variances of the diagonal and all-pairs estimators of the pairing objective and confirm them by simulation on the five atoms.
   *Provenance:* book-original.
1. [short-code] **Random re-pairing** (`ex_relativistic-random-repairing`). Extend the bad-minima enumeration to 4 atoms (256 assignments) and to a critic that best-responds per random pairing, counting suboptimal strict local minima.
   *Provenance:* book-original (extends the enumeration of the section, which follows Sun, Fang & Schwing, NeurIPS 2020; carried over from "Counting Mode-Dropping Minima" in spirit).

---

## chapter_generative-adversarial-networks/convergence.md — Gradient Penalties and Convergence

**Topic:** Why a correct equilibrium doesn't imply convergent training — the Dirac-GAN's
circular continuous-time dynamics vs. outward-spiraling discrete updates — and how
zero-centered gradient penalties R1/R2 restore local convergence and (near equilibrium)
measure a linearized Wasserstein-2 distance; tested via the R3GAN recipe on 25 Gaussians.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [short-code] **Discrete-time stability** (`ex_convergence-discrete-simultaneous-update-dirac`). Compute the Jacobian of the simultaneous Dirac-GAN update, show no step size converges unpenalized, and give the convergence condition for the penalized game in the underdamped regime.
   *Provenance:* original (carried over from "Eigenvalues of the Discrete Map" and "Sweeping the Step Size").
1. [conceptual] **Penalized best response** (`ex_convergence-penalized-best-response`). Compute the penalized best-response value for the Dirac-GAN, show it gives the generator a quadratic gradient, and compare with the Sobolev-norm view.
   *Provenance:* original (carried over from "A Sobolev-Norm Interpretation" and "Mixture Weighting in the Penalty").
1. [conceptual] **Penalizing one side only** (`ex_convergence-one-sided-penalty`). Show a one-sided penalty leaves the critic objective unbounded and the non-saturating loss gradient diverging, reproducing an observation of Huang et al. 2024.
   *Provenance:* book-original.
1. [short-code] **Batch-mixing critics** (`ex_convergence-batch-mixing-critics`). Build a batch-centered critic and show the per-sample penalty implementation mis-reports its input gradient.
   *Provenance:* book-original.
1. [short-code] **The penalty weight as a length** (`ex_convergence-penalty-length`). Show the penalty weight scales as a squared length, compute the damping of a grid-spacing component, and rerun the 25-Gaussians training for longer.
   *Provenance:* book-original.
1. [short-code] **Alternating updates** (`ex_convergence-alternating-updates`). Linearize the alternating Dirac-GAN update, show the Jacobian determinant is one and eigenvalues stay on the unit circle, and test by simulation.
   *Provenance:* adapted from Mescheder, Geiger & Nowozin (ICML 2018), who state that alternating updates also fail to converge for the unpenalized game (carried over from "Alternating Updates on the Dirac-GAN").
1. [conceptual] **The local value** (`ex_convergence-prove-eqref-eq-gan`). Expand the pairing objective to second order, derive the first-order condition for the maximizing critic and its one-dimensional CDF solution (the linearized Wasserstein-2 value).
   *Provenance:* book-original.
1. [short-code] **One-centered penalty dynamics** (`ex_convergence-one-centered-dynamics`). Analyze the WGAN-GP (one-centered penalty) Dirac-GAN flow, find its one-sided limits and non-convergence.
   *Provenance:* book-original.
1. [short-code] **Momentum in a rotating game** (`ex_convergence-momentum-rotating-game`). Linearize the penalized Dirac-GAN with beta-averaged momentum and relate the result to R3GAN's Adam beta1=0.
   *Provenance:* book-original.
1. [short-code] **Non-saturating Dirac dynamics** (`ex_convergence-non-saturating-dynamics`). Analyze the non-saturating Dirac flow: the decay rate along the flow, why the approach cannot be linear, and the averaged angular dynamics.
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
1. [conceptual] **Checkerboards from stride** (`ex_dcgan-checkerboards-from-stride`). Find every kernel/padding combination that exactly doubles the side, and count input positions per output for kernels 3 and 4 to explain checkerboard artifacts.
   *Provenance:* adapted from University of Toronto CSC321, Programming Assignment 4, "Implement the Discriminator of the DCGAN," Part 1 Question 1 (2018) (carried over from "Padding for Exact Halving", extended with the checkerboard counting).
1. [conceptual] **Penalty weight and resolution** (`ex_dcgan-penalty-weight-resolution`). Show how the penalty weight must rescale under pixel upsampling, so part of its dataset and resolution dependence is a units effect.
   *Provenance:* book-original.
1. [short-code] **A scale-free overfitting statistic** (`ex_dcgan-scale-free-overfitting-statistic`). Compute the r_v and r_t overfitting statistics of Karras et al. 2020 for the three arms so the critic gaps are comparable across scales.
   *Provenance:* book-original.
1. [conceptual] **FID across dimensions** (`ex_dcgan-one-dimension-squared-wasserstein`). Derive the one-dimensional squared Wasserstein-2 for Gaussians, evaluate FID for shifted and spread features, and compare dimension dependence.
   *Provenance:* original (carried over from "FID for Univariate Gaussians").
1. [short-code] **Finite-sample bias** (`ex_dcgan-measure-finite-sample-bias`). Measure real-versus-real FD and KID against sample size, extrapolate to infinite n, and repeat for generator comparisons.
   *Provenance:* adapted from Chong & Forsyth, "Effectively Unbiased FID and Inception Score" (CVPR 2020) (overlap high; carried over from "Finite-Sample Bias in FID/KID").
1. [short-code] **Precision and recall** (`ex_dcgan-precision-recall`). Implement k-nearest-neighbor ball precision and recall (Kynkaanniemi et al. 2019) on the CIFAR-CNN features for every generator.
   *Provenance:* book-original.
1. [short-code] **Initialization, latent injection, and collapse** (`ex_dcgan-rerun-arm-classic-loss`). Rewritten: define a feature-space diversity measure, record it every 500 steps for the unpenalized arm under different initializations and latent injection, and tie collapse to each choice.
   *Provenance:* original (the earlier "Initialization and Mode Collapse"; rewritten with the diversity statistic).
1. [conceptual] **Moments behind FID and KID** (`ex_dcgan-moments-fid-kid`). Show FID sees only means and covariances, expand the cubic KID kernel to see which moments it compares, and test on one-dimensional distributions that differ in higher moments.
   *Provenance:* book-original.
1. [short-code] **Dependence on the feature network** (`ex_dcgan-retrain-feature-network-this`). Retrain the feature CNN on Fashion-MNIST, rescore every generator, and test whether the ordering of similar models is stable.
   *Provenance:* original (carried over from "Feature Networks and Ranking Stability").
1. [conceptual] **Leaking augmentations** (`ex_dcgan-augmentation-leaks`). Show that augmenting only real images makes the optimal generator match the flipped mixture, and analyze when differentiable augmentation of both sides avoids the leak.
   *Provenance:* book-original.

---

## chapter_generative-adversarial-networks/conditional.md — Conditional Generation

**Topic:** Conditioning both networks on a label, deriving the per-condition value
E_c[2JS(p(·|c),q(·|c))]−2log2 and the projection discriminator from a Bayes-rule
decomposition, training a class-conditional CIFAR-10 generator, and separating
fidelity/diversity/alignment as distinct evaluation axes; concluding with pix2pix/CycleGAN
as image-conditioned generation.

**Current exercises:** 10; the problem set below lists them in source order with their `:exlabel:` ids (2026-11 revision; the earlier keep/rewrite/drop dispositions are superseded).

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

**Proposed problem set** (10 problems):
1. [short-code] **Two slices with known densities** (`ex_conditional-verify-decomposition-eqref-eq`). Compute the conditional game value by quadrature on two Gaussian slices and check it against the loss of a small trained critic.
   *Provenance:* original (carried over from "Verifying the Per-Condition JS").
1. [conceptual] **Separate feature maps** (`ex_conditional-projection-derivation-modeled-both`). Derive the critic when the two class posteriors need different feature maps, show it is a projection head over concatenated features, and count the dimensions needed.
   *Provenance:* original (carried over from "Separate Feature Maps for Posteriors").
1. [short-code] **The critic as a classifier** (`ex_conditional-critic-classifier`). Express the optimal critic scores through the real and generated label posteriors and explain why the critic-based label classifier falls to chance as the generator improves.
   *Provenance:* book-original.
1. [short-code] **Per-class distances** (`ex_conditional-diagnostics-cell-detects-within`). Per-class FD and KID between generated and real features, including the rank problem of a 100-sample covariance.
   *Provenance:* original (carried over from "Per-Class Distributional Diagnostics").
1. [short-code] **Truncated latents** (`ex_conditional-biggan-s-truncation-trick`). Compute the in-box probability and norm of truncated latents and sweep the truncation threshold on the EMA generator, reporting alignment, precision and recall.
   *Provenance:* adapted from Simon Prince, Understanding Deep Learning, Problem 15.6 (carried over from "Sweeping the Truncation Trick").
1. [conceptual] **Mismatched label marginals** (`ex_conditional-suppose-label-marginal-is`). Use the unequal-priors result to show which skewed divergence a mismatched condition marginal r(c) optimizes.
   *Provenance:* book-original.
1. [short-code] **A rare class** (`ex_conditional-rare-class`). Bound the value cost of ignoring a rare slice, then train with natural versus class-balanced real batches on a 50-image class.
   *Provenance:* original (carried over from "Across-Slice Versus Within-Slice Imbalance").
1. [conceptual] **Auxiliary classifier bias** (`ex_conditional-auxiliary-classifier-bias`). Show the auxiliary classification loss is entropy plus KL, and that minimizing the entropy term pushes the generator toward easy-to-classify samples.
   *Provenance:* book-original.
1. [short-code] **The critic at a fixed budget** (`ex_conditional-implement-concatenation-critic-one`). Sweep the penalty weight at a fixed 4,000-step budget, then replace the projection head by a concatenation critic.
   *Provenance:* original (carried over from "Concatenation Versus Projection Critics").
1. [conceptual] **Identifiability in translation** (`ex_conditional-translation-identifiability`). Show a marginal critic without L1 admits an input-ignoring generator in pix2pix, and that CycleGAN's cycle loss leaves an invertible ambiguity.
   *Provenance:* original (carried over from "Classifying Adversarial Losses in Translation" and "Reproducing the Cycle-Consistency Ablation").

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
1. [conceptual] **Distortion of a perfect sampler** (`ex_adversarial-losses-perfect-sampler`). Show a perfect conditional sampler has twice the minimum squared error, convert it to a PSNR loss in decibels, and apply it to the two-band data (Blau-Michaeli trade-off).
   *Provenance:* book-original.
1. [short-code] **Mean and median** (`ex_adversarial-losses-mean-and-median`). Compare conditional mean and median on three-offset and two-band variants and check whether each is a valid output.
   *Provenance:* original (carried over from "A Three-Band Toy Regression").
1. [short-code] **Pointwise plus adversarial** (`ex_adversarial-losses-pointwise-plus-adversarial`). Add a weighted pointwise loss to the adversarial student, sweep its weight, and analyze ADD's distillation target.
   *Provenance:* adapted from Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion Distillation" (arXiv:2311.17042) (carried over from "A Three-Arm Capacity Ablation").
1. [conceptual] **Phase and the conditional mean** (`ex_adversarial-losses-phase`). Show the conditional mean of a vocoder frame with uniform phase shrinks the amplitude, and that magnitude-only losses do not select a valid waveform.
   *Provenance:* book-original.
1. [conceptual] **Frozen critic features** (`ex_adversarial-losses-frozen-features`). Derive the optimal head and the divergence measured by a critic that sees only frozen features, and which generators minimize it.
   *Provenance:* book-original.
1. [conceptual] **DDO's implicit critic** (`ex_adversarial-losses-ddo-plays-log-loss`). Show DDO's parameterized critic represents the optimal critic exactly and that its gradient at theta = reference is half the negative log-likelihood gradient.
   *Provenance:* original (carried over from "DDO as an Optimal Critic").
1. [short-code] **A latent bit** (`ex_adversarial-losses-latent-bit`). Give the student a binary latent, compare squared-error and adversarial training, and test the IMLE nearest-sample loss.
   *Provenance:* original (carried over from "Giving the Student a Latent Bit").
1. [short-code] **Multi-round refinement** (`ex_adversarial-losses-multi-round-refinement`). Derive the minimizer of the practical DDO loss over densities, its normalization, and its behavior across rounds.
   *Provenance:* book-original.
