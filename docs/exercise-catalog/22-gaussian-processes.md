# Chapter Overview — chapter_gaussian-processes (gp-intro, gp-priors, gp-inference)

The best external match by far is Rasmussen & Williams' own GPML textbook (free at
gaussianprocess.org/gpml): its Ch.2 (Regression) and Ch.4 (Covariance Functions)
exercises are written by the same author who wrote this chapter's prose, so
notation and pedagogy line up almost exactly. Cambridge's 4F13 coursework
(Rasmussen & Kok) is the second-best match — it is Rasmussen's own applied
problem set, using the GPML MATLAB toolbox on the same SE/periodic/product
kernels this chapter builds. GPSS (Sheffield) Lab 1 is the best match for
*code-first* exercises (GPy instead of from-scratch NumPy, but same
lengthscale/variance/kernel-identification pedagogy) and even uses the same
Mauna Loa CO2 series d2l references elsewhere. The Distill article *A Visual
Exploration of Gaussian Processes* has no formal exercises but its
interactive-demo sequence is the closest external analogue to gp-intro's own
running is-this-prior-reasonable narration. Iain Murray's MLPR tutorial 7 covers the same ground (lengthscale
intuition, kernel combination) but its page returned HTTP 410 at fetch time in
2026 — content below is paraphrased from search-index snippets only, not a
verified live render, and is used for context, not adopted verbatim. All three
existing exercise sets are clean and well-posed (per the prior style review);
the main gap is *balance*: gp-inference's 6 exercises are all short-code (zero
pencil-and-paper), while gp-intro has code-free equations that no exercise
asks the reader to actually implement. External material is used mainly to
fill those two gaps, not to replace anything.

---

## chapter_gaussian-processes/gp-intro.md — Introduction to Gaussian Processes

**Topic:** Motivating GPs as distributions over functions; RBF kernel intuition (amplitude, length-scale); scalar-observation posterior updating by hand; epistemic vs. aleatoric uncertainty.

**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the prior style review found no defects and no clarity issues across all seven items (each has a concrete question and, where relevant, a worked numeric callback verified against the file body). This is the strongest-reviewed file in the chapter; we only add.

**External sources found:**
- Görtler, Kehlbeck & Deussen (Univ. of Konstanz), *A Visual Exploration of Gaussian Processes*, Distill, 2019 — no numbered exercises, but a deliberate interactive sequence: manipulate a bivariate Gaussian's mean/covariance, drag a conditioning handle to see a slice become a new Gaussian, click to sample GP-prior functions under different kernels, then toggle kernel sums/products. This is the closest external analogue to gp-intro's own running is-this-prior-reasonable narration. — https://distill.pub/2019/visual-exploration-gaussian-processes/
- Rasmussen & Williams, *GPML*, Ch. 2 (Regression), Exercises 2.9.1 and 2.9.3 — replicate the random-function sampling of the book's Fig. 2.2 from a covariance function and a chosen input grid, and separately derive/sample the Brownian-bridge kernel obtained by conditioning a Wiener process on `f(1)=0`. Both ask the reader to sample from equations the chapter has already stated, the same spirit as gp-intro's un-coded worked example. — https://gaussianprocess.org/gpml/chapters/RW2.pdf
- Gaussian Process Summer School (Sheffield), Lab 1 (2018), Exercises 1(a)–(b) — plot the covariance function `k(x,0)` while sweeping length-scale, then again while sweeping variance, to separate the rate-of-variation effect from the vertical-scale effect — directly mirrors gp-intro's own length-scale/amplitude sweep narrative, but as a coding task rather than a set of pre-rendered figures. — https://github.com/gpschool/gpss18/blob/master/labs/GPSS_Lab1_2018.ipynb
- Univ. of Edinburgh, Iain Murray, MLPR Tutorial 7 (Gaussian processes) — asks students to sketch three typical draws from a GP prior for a given kernel, and separately reasons about length-scale as the typical spacing between a function's turning points. (Verified via search-index snippets only; the live page returned HTTP 410 at fetch time.) — https://www.inf.ed.ac.uk/teaching/courses/mlpr/2018/tut/tut7_questions.html

**Proposed problem set** (8 problems, our reference format):

1. [conceptual] **Epistemic Versus Observation Uncertainty.** State the distinction between epistemic (reducible) and observation/aleatoric (irreducible) uncertainty in one or two sentences each. Then describe one GP regression scenario where the two are comparable in size and one where they differ by an order of magnitude, saying which part of the section's credible-interval plot would visibly change in each case.
   *Provenance:* original (section's own Exercise 1, reworded to name an explicit deliverable).
1. [conceptual] **Function Properties Beyond Rate and Amplitude.** Besides rate of variation (length-scale) and vertical scale (amplitude), name at least two further high-level properties a function might have (e.g., periodicity, smoothness, symmetry, boundedness) and give one real-world example signal for each that a length-scale/amplitude-only kernel would fail to capture well.
   *Provenance:* original (section's own Exercise 2).
1. [conceptual] **Stationarity Assumption Critique.** The RBF kernel assumes covariance decays with distance in input space alone, regardless of location. Give one real dataset (from the times/spatial-locations/pixels examples the section already uses) where this is a reasonable assumption, and one where it plainly is not, justifying each in 2–3 sentences.
   *Provenance:* original (section's own Exercise 3).
1. [conceptual] **Gaussian Closure Warm-Up.** Answer, with a one-line justification each: is a sum of two Gaussian variables Gaussian? A product? If `(a,b)` is jointly Gaussian, is `a|b` Gaussian? Is `a` alone Gaussian? These are the scalar building blocks behind the GP closure properties tested more fully in gp-priors.
   *Provenance:* original (section's own Exercise 4).
1. [conceptual] **Two-Point Posterior by Hand.** Repeat the section's own worked update (observing `f(x_1)=1.2` with `k(x,x_1)=0.9`), but now add a second observation `f(x_2)=1.4` with `k(x,x_2)=0.8`. Compute the new mean and 95% credible interval for `f(x)` by hand and state whether uncertainty went up or down relative to the one-point case.
   *Provenance:* original (section's own Exercise 5).
1. [conceptual] **Noise Versus Length-Scale Confound.** Argue whether increasing your estimate of the observation-noise variance should push your estimate of the ground-truth length-scale up or down, using the section's own credible-interval reasoning (not code).
   *Provenance:* original (section's own Exercise 6).
1. [conceptual] **Saturating Predictive Uncertainty.** As a test point moves far from all training data, the predictive uncertainty grows but eventually stops increasing rather than diverging. Explain why, in terms of what `k(x,x)` alone determines about the prior variance.
   *Provenance:* original (section's own Exercise 7).
1. [short-code] **Sampling the RBF Prior.** The section states the RBF kernel formula and the joint-Gaussian definition of a GP prior but never codes them (this is the only file in the chapter with no executable cells). Write under 15 lines of NumPy that builds the mean vector and RBF covariance matrix on a grid of `x` points, draws 5 prior samples via `np.random.multivariate_normal`, and reproduces the qualitative length-scale sweep (`ℓ=0.1,0.5,2,5,10`) the section describes only in images. Deliverable: one figure with 5 sampled functions for two contrasting length-scales.
   *Provenance:* inspired by GPSS Lab 1, Ex. 1(a)–(b) and the Distill prior-sampling demo (overlap low; both externally use their own libraries/toolkits, this problem restricts itself to the equations already printed in gp-intro.md).

---

## chapter_gaussian-processes/gp-priors.md — Gaussian Process Priors

**Topic:** Weight-space to function-space view of GP priors; deriving the RBF kernel as an infinite basis-function limit; the neural-network kernel and stationary vs. non-stationary behavior.

**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the review found no defects and described all five as direct, well-posed math questions with clear expected answers, the strongest disposition in the chapter. We add 2 to exploit content in the section that is stated but never exercised (the neural-network kernel has a full closed-form formula but zero code or follow-up question).

**External sources found:**
- Rasmussen & Williams, *GPML*, Ch. 4 (Covariance Functions), Exercise 4.5.2 — labeled a computer exercise: write code to draw samples from the neural-network covariance function (their eq. 4.29, the same arcsin-form kernel this section derives) in 1-D and 2-D, and explain the sample shape when `var(u_0)=0` vs. non-zero. This is essentially a ready-made assignment for a formula gp-priors.md states but leaves uncoded. — https://gaussianprocess.org/gpml/chapters/RW4.pdf
- Rasmussen & Williams, *GPML*, Ch. 4, Exercise 4.5.6 — show the squared-exponential process is infinitely mean-square (MS) differentiable and that the Ornstein–Uhlenbeck (OU) process is not MS differentiable, connecting kernel smoothness at the origin to sample-path roughness. — https://gaussianprocess.org/gpml/chapters/RW4.pdf
- Cambridge 4F13 Probabilistic Machine Learning (Rasmussen & Kok, Michaelmas 2017), Coursework 1, parts (c)–(d) — train a GP with a periodic kernel and compare its error bars to an SE fit; then build a product kernel `{covProd,{covPeriodic,covSEiso}}`, sample near-noise-free functions from it, and explain why a small jitter term must be added to the covariance matrix before Cholesky. — https://mlg.eng.cam.ac.uk/teaching/4f13/1718/cw/coursework1.pdf
- Gaussian Process Summer School (Sheffield), Lab 1 (2018), Exercises 2(a)–(b) and 3 — reason about why sums and elementwise products of valid covariance functions are themselves valid, then match six unlabeled sample-function plots to the kernel (RBF/Matérn/periodic/Brownian/linear) that generated them. — https://github.com/gpschool/gpss18/blob/master/labs/GPSS_Lab1_2018.ipynb
- Univ. of Edinburgh, Iain Murray, MLPR Tutorial 7 — a question on combining two kernels by addition, reasoning informally that variances/covariances of independent Gaussian processes add under summation. (Verified via search-index snippets only; the live page returned HTTP 410 at fetch time — used for external-tradition context, not adopted into a problem below.) — https://www.inf.ed.ac.uk/teaching/courses/mlpr/2018/tut/tut7_questions.html

**Proposed problem set** (7 problems, our reference format):

1. [short-code] **OU Versus RBF Sample Comparison.** Implement the Ornstein–Uhlenbeck kernel `k_OU(x,x')=exp(-|x-x'|/ℓ)` alongside the section's existing `rbfkernel` function, draw sample prior functions from each at matched length-scale, and describe in 2–3 sentences how the OU samples' visual roughness differs from the RBF samples'.
   *Provenance:* original (section's own Exercise 1, tagged short-code since it requires implementing a new kernel function against the section's own sampling code).
1. [conceptual] **Amplitude Scaling Effect.** Derive how the RBF kernel's amplitude `a²` enters the covariance of sampled function values, and state precisely what changes (and what does not) in the distribution over functions as `a` grows.
   *Provenance:* original (section's own Exercise 2).
1. [conceptual] **Sum of Two GPs.** For `u(x) = f(x) + 2g(x)` with `f ~ GP(m_1,k_1)` and `g ~ GP(m_2,k_2)` independent, show `u` is a GP and give its mean and covariance functions in closed form.
   *Provenance:* original (section's own Exercise 3).
1. [conceptual] **Input-Dependent Scaling of a GP.** For `g(x) = a(x) f(x)` with `f ~ GP(0,k)` and `a(x)=x²`, show `g` is a GP, give its mean and covariance functions, and sketch (by hand) what a sample from `g` looks like compared to a sample from `f`.
   *Provenance:* original (section's own Exercise 4).
1. [conceptual] **Product of Two GPs.** For `u(x) = f(x) g(x)` with `f ~ GP(m_1,k_1)` and `g ~ GP(m_2,k_2)` independent, determine whether `u` is a GP; if so, give its mean and covariance, and if not, say precisely which closure property fails.
   *Provenance:* original (section's own Exercise 5).
1. [short-code] **Sampling the Neural-Network Kernel.** The section derives the neural-network kernel `k(x,x') = (2/π) arcsin(...)` in closed form but never codes or samples it. Implement it, draw 5 prior samples in 1-D, and reproduce the section's own observation that samples look distinctly non-stationary near the origin compared to an RBF sample — then explain, from the formula, why setting `var(u_0)=0` removes that special point.
   *Provenance:* adapted from GPML Ex. 4.5.2 (overlap high; near-identical ask — implement the same closed-form kernel and explain the `var(u_0)=0` special case — cite GPML on adoption).
1. [conceptual] **Mean-Square Differentiability of SE and OU.** Using the fact that MS-differentiability at `τ=0` requires the kernel to be twice differentiable there, show `k_SE(τ)=a²exp(-τ²/2ℓ²)` has a well-defined second derivative at `τ=0` while `k_OU(τ)=exp(-|τ|/ℓ)` does not (it has a corner). Connect this to why the OU samples from Problem 1 look rougher than the RBF samples.
   *Provenance:* adapted from GPML Ex. 4.5.6 (overlap medium; original is a general MS-differentiability proof for arbitrary stationary kernels, here scoped down to a direct SE-vs-OU second-derivative check at the origin — cite GPML on adoption).

---

## chapter_gaussian-processes/gp-inference.md — Gaussian Process Inference

**Topic:** Closed-form GP regression posterior; learning kernel hyperparameters by maximizing the marginal likelihood; from-scratch NumPy implementation versus GPyTorch; epistemic vs. aleatoric uncertainty decomposition; cubic cost of exact inference.

**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — the review flagged Exercise 4 (timing 10k/20k/40k points) as resource-heavy but not a clarity defect, and found all six items well-posed with concrete deliverables. The imbalance we correct is structural, not qualitative: all 6 existing exercises are short-code and none are pencil-and-paper, despite the section containing rich untested derivations (the marginal-likelihood decomposition, the jitter argument, the block-matrix predictive-variance formula).

**External sources found:**
- Rasmussen & Williams, *GPML*, Ch. 2 (Regression), Exercise 2.9.4 — prove that the GP predictive variance at a fixed test point can never increase as more training data is added, via the partitioned-matrix-inverse identity, using the same predictive-variance expression this section states. — https://gaussianprocess.org/gpml/chapters/RW2.pdf
- Rasmussen & Williams, *GPML*, Ch. 2, Exercise 2.9.2 — show the feature-space predictive covariance `φ(x*)ᵀA⁻¹φ(x*')` is compatible with the kernel-space expression the book gives elsewhere; noted as an alternative to the adopted problem below, not separately adopted here to avoid redundant linear-algebra proofs in one section. — https://gaussianprocess.org/gpml/chapters/RW2.pdf
- Cambridge 4F13 Probabilistic Machine Learning (Rasmussen & Kok, Michaelmas 2017), Coursework 1, parts (a), (b), (d) — fit an SE-kernel GP with the GPML toolbox and show 95% error bars; show that re-initializing hyperparameters finds a different local optimum of the marginal likelihood; and justify why a small jitter term (`1e-6*eye(n)`) must be added to the covariance matrix before a Cholesky decomposition — the same jitter this section's prose already discusses. — https://mlg.eng.cam.ac.uk/teaching/4f13/1718/cw/coursework1.pdf
- Gaussian Process Summer School (Sheffield), Lab 1 (2018), Exercises 4–7 — manually tune, then marginal-likelihood-optimize, an RBF fit on toy data; extend predictions beyond the training range and discuss uncertainty growth; and build a sum-of-kernels (Bias+Linear) model for a real CO2 time series, closely paralleling this section's own from-scratch-then-GPyTorch progression. — https://github.com/gpschool/gpss18/blob/master/labs/GPSS_Lab1_2018.ipynb

**Proposed problem set** (8 problems, our reference format):

1. [short-code] **Hyperparameter Sensitivity Sweep.** Skip the marginal-likelihood optimization step and instead manually try a grid of length-scales and noise variances on the from-scratch model. Report, with plots, what happens to the fit and credible set for a large length-scale, a small length-scale, a large noise variance, and a small noise variance.
   *Provenance:* original (section's own Exercise 1).
1. [short-code] **Local Optima in Marginal Likelihood.** Re-run the hyperparameter optimization from two deliberately mismatched starting points (large length-scale + large noise; small length-scale + small noise) and report whether the optimizer converges to different solutions, with the final marginal-likelihood value at each.
   *Provenance:* original (section's own Exercise 2).
1. [short-code] **Extrapolating Beyond the Training Range.** Re-run prediction with `test_x = np.linspace(0, 10, 1000)` (training data stays on `[0,5]`) and report how the 95% credible set behaves beyond `x=5`, whether it still covers the true function there, and how the picture changes if you plot only aleatoric (observation-noise) uncertainty in that region.
   *Provenance:* original (section's own Exercise 3).
1. [short-code] **Cubic Cost in Practice.** Re-run the from-scratch fit at 10,000, 20,000, and 40,000 training points, measuring wall-clock time for training and for prediction separately; compare the empirical scaling to the theoretical `O(n^3)`/`O(n^2)` counts. Readers on limited hardware may substitute 1,000/2,000/4,000 and report the scaling exponent from a log-log fit instead of raw times.
   *Provenance:* original (section's own Exercise 4; added a low-compute fallback given the review's resource-heavy flag).
1. [short-code] **Kernel Choice in GPyTorch.** Re-run the GPyTorch example with a Matérn kernel, then a spectral-mixture kernel, in place of the RBF kernel. Report which one is easier to fit (fewer iterations to a stable marginal likelihood) and which extrapolates better at long range versus short range.
   *Provenance:* original (section's own Exercise 5).
1. [short-code] **Epistemic-Only GPyTorch Plot.** Redo the GPyTorch prediction plot showing only the latent-function (epistemic) credible set rather than the observation-space one, and confirm it visually matches the from-scratch epistemic-only plot earlier in the section.
   *Provenance:* original (section's own Exercise 6).
1. [conceptual] **Why Jitter Fixes Conditioning.** Using the eigendecomposition `K = U Λ Uᵀ`, show that the condition number of `K` is `λ_max/λ_min`, and that adding jitter `εI` changes it to `(λ_max+ε)/(λ_min+ε)`. State, for a covariance matrix with `λ_min` near `1e-9` and `λ_max` near `1`, whether `ε=1e-6` (the value the section uses) meaningfully improves conditioning.
   *Provenance:* adapted from Cambridge 4F13 CW1, part (d) (overlap medium; the original asks for a one-line justification, here expanded into a full eigenvalue derivation using this section's own `K + σ²I` object — cite 4F13 on adoption).
1. [conceptual] **Predictive Variance Cannot Increase.** Using the partitioned-matrix inverse identity, prove that the predictive variance `S = K(x_*,x_*) - K(x_*,X)[K(X,X)+σ²I]⁻¹K(X,x_*)` at a fixed test point can only decrease or stay the same as a new training point is added to `X` — i.e., epistemic uncertainty is monotonically non-increasing in the data.
   *Provenance:* adapted from GPML Ex. 2.9.4 (overlap high; same claim, same partitioned-matrix-identity hint, restated in this section's own notation — cite GPML on adoption).

---

## Summary

Sources, in order of usefulness: (1) GPML Ch. 2 & 4 exercises — same author, same notation, free online, direct textbook match; (2) Cambridge 4F13 Coursework 1 (Rasmussen & Kok) — Rasmussen's own applied problem set on the same kernels; (3) GPSS Sheffield Lab 1 — best code-first analogue (GPy vs. NumPy), covers the same lengthscale/variance/kernel-ID ground and even a CO2 dataset the book alludes to; (4) Distill's *A Visual Exploration of Gaussian Processes* — no formal exercises, but its interactive-demo sequence is the closest external match to gp-intro's own style; (5) Iain Murray's MLPR Tutorial 7 — same topics, but its page 410'd at fetch time, so it's used only as corroborating context (search-index snippets), never adopted into a numbered problem.

No section lacks an external exercise tradition — GP regression/covariance functions is one of the most consistently taught ML topics, and Rasmussen having written both this chapter's chief external source (GPML) and its second-best (4F13) is an unusually tight match.

Totals: 3 sections, all reviewed with disposition keep (18 of 18 existing exercises kept, 0 rewritten, 0 dropped) + 6 new problems proposed, for 24 total problems across the chapter (gp-intro 8, gp-priors 7, gp-inference 8). New problems: 2 short-code (RBF/NN-kernel sampling), 4 conceptual (MS-differentiability, jitter/conditioning, predictive-variance monotonicity — the last two directly closing gp-inference's 0-conceptual gap).
