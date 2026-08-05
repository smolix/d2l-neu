# Chapter Overview — chapter_mdl-probability-statistics

Best external sources, in order of how directly reusable they proved: (1) Wasserman's
*All of Statistics* — via its own numbered exercises (verified through a third-party
solutions mirror, parsiad.ca) — matched nearly every section, from Ch. 2/4 groundwork
(independence, Markov/Chebyshev, Chernoff-type bounds) through Ch. 8-10 (bootstrap, MLE,
hypothesis testing); (2) Stanford CS229 (both the classic public-course PS1 and the
Summer 2020 PS1), the best match for the ML-flavored sections (naive Bayes, MLE,
exponential-family GLMs); (3) CMU 10-601's Naive Bayes homework, a full text-classification
assignment nearly isomorphic to the book's own MNIST pipeline; (4) Wainwright's
*High-Dimensional Statistics* Ch. 2 (22 titled exercises) and Vershynin's
*High-Dimensional Probability* Ch. 2, the two standard sources for concentration; (5) MIT
18.05, clean verified in-class problems for CDF/density/variance basics. Harvard Stat 110's
strategic-practice PDFs are blocked outright by the university's bot protection (Akamai
403 on every direct and proxied fetch attempt); their topics are cited at title-confirmation
level only, not as adopted-with-quote adaptations. Two real coverage gaps emerged:
Bayesian computation (MCMC/importance sampling/VI) and naive Bayes/MLE both sit outside
Stat 110/18.05/6.041's syllabi entirely — a finding, not a failure — and are better served
by Gutmann's freely available "Pen and Paper Exercises in Machine Learning" and by
CS229/CMU 10-601, respectively. Six of the seven files' existing exercise sets were already
rated defect-free by the prior style review, and that held up: they are direct, checkable,
and several carry real citations (Koller & Friedman, McDiarmid, Dwork et al.). Dispositions
below skew heavily toward "keep"; drops are reserved for exercises redundant with an
in-text proof or under-specified against the "no bare explain-why" rule.

---

## chapter_mdl-probability-statistics/mdl-random-variables.md — Random Variables

**Topic:** Densities and CDFs for continuous variables, mean/variance, Markov/Chebyshev,
joint/marginal/conditional densities and independence, conditional expectation, covariance
and correlation, change of variables.
**Current exercises:** 7; disposition: keep 4, rewrite 1, drop 2 — the set is already
strong and direct (prior review found no clarity issues); the two drops are narrow,
somewhat isolated calculations (a Laplace-density moment calc that duplicates machinery
used more richly elsewhere, and a second proof of an identity the text already proves
twice), and the one rewrite exists because none of the 7 pairs with the section's own
three code cells (Riemann-sum density check, Cauchy-divergence check, change-of-variables
log-normal check) despite all being pencil-and-paper.

**External sources found:**
- MIT 18.05 (Orloff & Bloom), Class 5 In-Class Problems, Spring 2022, Problems 2-3 —
  given a density $cx^2$ on $[0,2]$ or a CDF $y^2/9$ on $[0,b]$, find the normalizing
  constant/endpoint, derive the CDF or density, and compute an interval probability —
  https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class05a_pset_sol.pdf
- Harvard Stat 110 (Blitzstein), Strategic Practice 7, Fall 2011 — joint, conditional,
  and CDF problems for continuous pairs (title/topic confirmed only; direct fetch blocked
  by the host's bot protection) —
  https://projects.iq.harvard.edu/files/stat110/files/strategic_practice_and_homework_7.pdf
- Harvard Stat 110, Strategic Practice 8, Fall 2011 — covariance and correlation practice
  problems (title/topic confirmed only) —
  https://projects.iq.harvard.edu/files/stat110/files/strategic_practice_and_homework_8.pdf
- Wasserman, *All of Statistics*, Ch. 2, Exercises 5, 7, 12 — independence as joint-density
  factorization; the CDF of $\min(X,Y)$ for independent $X,Y$; density factorization
  $\Leftrightarrow$ independence — verified via solutions mirror
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_2_solutions.pdf
- Wasserman, *All of Statistics*, Ch. 4, Exercises 1, 3 — Chebyshev's bound compared
  against the *exact* tail of an exponential; a Chernoff-type bound for a Bernoulli sample
  mean that beats Chebyshev for large $n$ — verified via
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_4_solutions.pdf

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **Density and CDF from a Power-Law Tail.** Verify $p(x)=1/x^2$ on
   $x\ge1$ is a valid density, then compute $P(X>2)$ and the full CDF $F(x)$, checking
   that $F$ is continuous and increases to $1$.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **Chebyshev's Bound Against a Finite Sample.** Using $\mu=1,\sigma=2$,
   bound the population probability $P(X>9)$ via Chebyshev, then explain why observing
   $25\%$ of a *finite* sample above $9$ is evidence against the population claim without
   being logically impossible absent the sample size.
   *Provenance:* original (existing exercise 3, retitled; no change).
3. [conceptual] **Joint Density, Marginals, and an Independence Check.** For
   $p_{XY}(x,y)=x+y$ on $[0,1]^2$, verify unit integral, find both marginals and
   $\textrm{Cov}(X,Y)$, and decide independence.
   *Provenance:* original (existing exercise 4, retitled; no change).
4. [short-code] **Chebyshev Versus the Exact Tail.** For $\bar X_n$, the mean of $n$
   i.i.d. $\textrm{Bernoulli}(p)$ draws, plot both the Chebyshev bound on
   $P(|\bar X_n-p|\ge\varepsilon)$ and the *exact* binomial tail probability against $n$ on
   a log scale (as the section's own :numref:`sec_mdl-concentration-generalization` later
   does for a fair coin); report at what $n$ the exact tail first beats the Chebyshev
   bound by an order of magnitude.
   *Provenance:* adapted from Wasserman, *All of Statistics*, Ch. 4, Exercises 1 & 3
   (overlap low — we replace their exponential-tail and Chernoff-bound derivations with a
   numerical demo built only from the section's own inequality and a binomial sample).
5. [short-code] **Sampling a New Density by Inversion.** Given the CDF $F(y)=y^2/9$ on
   $[0,3]$, derive $p(y)=F'(y)$, implement $F^{-1}$, generate $10^5$ samples via the
   section's own inverse-transform-sampling recipe, and overlay a histogram on the
   derived density.
   *Provenance:* adapted from MIT 18.05, Class 5 In-Class Problems, Problem 3 (overlap
   med — we reuse their CDF verbatim but extend it into the section's own
   inverse-transform-sampling code pattern, which their problem does not ask for).
6. [conceptual] **Covariance Blind to a Deterministic Relationship.** For $X$ uniform on
   $\{-1,0,1\}$ and $Y=|X|$, compute $\textrm{Cov}(X,Y)$, confirm it is zero despite $Y$
   being a deterministic function of $X$, and explain what correlation misses here.
   *Provenance:* original (existing exercise 7, retitled; no change).

---

## chapter_mdl-probability-statistics/mdl-distributions.md — Distributions

**Topic:** A reference gallery of eleven named discrete/continuous distributions, the
exponential family, and the Beta/Gamma/Dirichlet conjugate priors.
**Current exercises:** 10; disposition: keep 7, rewrite 1, drop 2 — this is another set
the prior review found defect-free, and it already touches nearly every subtopic (sums of
Bernoullis, softmax gradients, memorylessness, the Poisson CLT limit, Laplace MLE,
categorical-as-exponential-family, MVN geometry, Beta-Bernoulli conjugacy, MVN
conditioning); the two drops are mechanical algebra checks with limited independent
insight (a binomial-from-multinomial reduction, and an MVN eigen/linear-map restatement
of a worked example already in the text), and the rewrite turns a pencil-and-paper Beta
posterior computation into a short numerical convergence check.

**External sources found:**
- Stanford CS229, Summer 2020, Problem Set #1, Q3 — show the Poisson pmf is an
  exponential-family member (identify $b(y),\eta,T(y),a(\eta)$), find the canonical
  response function, and derive the GLM stochastic-gradient-ascent update —
  https://cs229.stanford.edu/summer2020/ps1.pdf
- Stanford CS229, Summer 2020, Problem Set #1, Q4 — prove $\mathbb E[Y;\eta]=\partial
  a/\partial\eta$ and $\textrm{Var}(Y;\eta)=\partial^2a/\partial\eta^2$ for a scalar
  exponential family, then show the resulting NLL is convex (Hessian PSD) — same URL;
  this is essentially our own exponential-family moment proposition, derived
  independently.
- MIT 6.041/6.431 (Bertsekas & Tsitsiklis), Fall 2010/2013 — problem sets and recitation
  drills on the Bernoulli process, Poisson, and the binomial-to-Poisson approximation
  (topic confirmed via OCW listings, not individually fetched) —
  https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/363f55b9e3f3950690bc22e9e8b521bf_MIT6_041F10_assn06_sol.pdf
- Harvard Stat 110, Strategic Practice 3, Fall 2011 — Bernoulli/Binomial problems
  including conditioning and Simpson's paradox (title/topic confirmed only) —
  https://projects.iq.harvard.edu/files/stat110/files/strategic_practice_and_homework_3.pdf
- Harvard Stat 110, Strategic Practice 5, Fall 2011 — Poisson distribution problems
  (title/topic confirmed only) —
  https://projects.iq.harvard.edu/files/stat110/files/strategic_practice_and_homework_5.pdf
- Harvard Stat 110, Strategic Practice 6, Fall 2011 — Exponential distribution problems
  (title/topic confirmed only) —
  https://projects.iq.harvard.edu/files/stat110/files/strategic_practice_and_homework_6.pdf

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **Variance of a Difference of Binomials.** For independent
   $X,Y\sim\textrm{Binomial}(16,1/2)$, find the standard deviation of $X-Y$.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **Softmax Gradient Is a Residual.** Show the categorical NLL's gradient
   with respect to the logits is $\hat{\mathbf p}-\mathbf y$, and explain why this form
   keeps softmax classification well behaved.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [conceptual] **Memorylessness Forces the Exponential.** Prove the converse of
   memorylessness (a multiplicative survival function forces $G(t)=e^{-\lambda t}$), then
   show the minimum of independent exponentials is again exponential.
   *Provenance:* original (existing exercise 4, retitled; no change).
4. [short-code] **Poisson as a CLT Limit, Simulated.** Argue $(X-\lambda)/\sqrt\lambda$
   becomes approximately Gaussian as $\lambda\to\infty$ by splitting $\lambda$ into unit
   Poisson pieces, then confirm numerically by histogramming standardized Poisson draws
   at increasing $\lambda$ against $\mathcal N(0,1)$.
   *Provenance:* original (existing exercise 5, retitled and its hint promoted to an
   explicit simulation deliverable).
5. [conceptual] **Laplace MLE Is the Median.** Show the maximum-likelihood location under
   a Laplace model minimizes $\sum_i|x_i-\mu|$ and hence is the sample median, contrasting
   with the Gaussian's mean.
   *Provenance:* original (existing exercise 6, retitled; no change).
6. [conceptual] **Categorical Distribution in Exponential-Family Form.** Write the
   categorical over $K$ classes in exponential-family form, identify $T(\mathbf x)$
   (one-hot) and the natural parameters, and verify $\nabla A(\boldsymbol\eta)=\mathbb
   E[T]$ recovers the softmax.
   *Provenance:* original (existing exercise 7, retitled); CS229 PS1 Q4 derives the same
   moment identity abstractly for a generic scalar family and is a useful pencil-and-paper
   companion, noted above rather than adopted verbatim.
7. [short-code] **Beta-Bernoulli Posterior, Simulated to Convergence.** Starting from a
   uniform $\textrm{Beta}(1,1)$ prior and the "HHHTHTTHHHHHT" sequence, give the posterior
   and its mean; then simulate additional flips at increasing $n$ and plot the posterior
   mean converging to the MLE frequency $n_H/(n_H+n_T)$.
   *Provenance:* rewrite of existing exercise 9 (original pencil-and-paper claim, now
   paired with a numerical convergence plot).

---

## chapter_mdl-probability-statistics/mdl-maximum-likelihood.md — Maximum Likelihood

**Topic:** The MLE principle, NLL as cross-entropy/MSE, MAP and regularization, Fisher
information/Cramér-Rao and asymptotic normality, latent variables/EM/ELBO.
**Current exercises:** 10; disposition: keep 8, rewrite 1, drop 1 — another defect-free
set per the prior review, spanning exponential/Gaussian/categorical MLE, ridge and lasso
from Gaussian/Laplace priors, MAP$\to$MLE limits, Fisher information, the asymptotic-
normality simulation, and the GMM M-step; the one drop restates the Gaussian-NLL-equals-
MSE identity the text already proves and numerically labels, and the rewrite widens a
single-parameter Fisher-information exercise into the full $(\mu,\sigma)$ matrix case,
which is both more informative and better matched to an available external solution.

**External sources found:**
- Stanford CS229 (public course), Problem Set #1, Q4 — derive the naive-Bayes joint
  log-likelihood, show the MLE-parameter formulas match the standard counting rule, and
  prove the decision rule is linear — a clean pencil-and-paper MLE-to-classifier pattern
  matching this section's Gaussian-NLL and coin derivations —
  https://see.stanford.edu/materials/aimlcs229/problemset1.pdf
- Stanford CS229, Summer 2020, Problem Set #1, Q1(c)-(d) — for Gaussian discriminant
  analysis, derive the MLEs of $\phi,\mu_0,\mu_1,\Sigma$ by maximizing the log-likelihood
  and show the resulting posterior is a logistic sigmoid —
  https://cs229.stanford.edu/summer2020/ps1.pdf
- Wasserman, *All of Statistics*, Ch. 9, Exercise 2 — the MLE of $\textrm{Uniform}(a,b)$
  is $(\min_iX_i,\max_iX_i)$, the classic *non-regular* example (support depends on the
  parameter) that this section's regularity-conditions paragraph explicitly excludes —
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_9_solutions.pdf
- Wasserman, *All of Statistics*, Ch. 9, Exercise 5 — the Poisson MLE is the sample mean,
  with Fisher information $I(\lambda)=1/\lambda$ — direct companion to this section's
  coin/Bernoulli treatment — same URL as above.
- Wasserman, *All of Statistics*, Ch. 9, Exercises 7-8 — Fisher information and a
  delta-method CI for a two-proportion difference; the full Fisher information *matrix*
  $I(\mu,\sigma)=-(n/\sigma^2)\,\textrm{diag}(1,2)$ for a Gaussian's mean and variance
  jointly — same URL.

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **Exponential MLE from One and Many Observations.** From a single
   observation $x=3$ of $\alpha e^{-\alpha x}$, find the MLE of $\alpha$; generalize to a
   sample and show $\hat\alpha=1/\bar x$.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **Gaussian Mean MLE via the NLL.** For a Gaussian sample with known unit
   variance, show the MLE of the mean is the sample average by minimizing the Gaussian
   NLL.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [conceptual] **Categorical Cross-Entropy Gradient.** For a $K$-class softmax
   classifier, show the categorical NLL is the cross-entropy and its gradient with
   respect to the logits is $\hat{\mathbf p}-\mathbf y$.
   *Provenance:* original (existing exercise 4, retitled; no change).
4. [conceptual] **Ridge Regression from a Gaussian Prior.** Show a Gaussian prior on the
   weights recovers ridge regression and read off the weight-decay strength in terms of
   the noise and prior variances.
   *Provenance:* original (existing exercise 5, retitled; no change).
5. [conceptual] **MAP Converges to MLE.** Prove
   $\hat{\boldsymbol\theta}_{\textrm{MAP}}\to\hat{\boldsymbol\theta}_{\textrm{MLE}}$ as the
   prior variance $\tau^2\to\infty$, and separately as $n\to\infty$ for fixed $\tau$.
   *Provenance:* original (existing exercise 7, retitled; no change).
6. [short-code] **Fisher Information Matrix for a Gaussian's Mean and Variance.** Extend
   the single-parameter Fisher-information computation to the joint $(\mu,\sigma)$ case,
   deriving $I(\mu,\sigma)=-(n/\sigma^2)\,\textrm{diag}(1,2)$; then simulate many
   Gaussian samples, compute the empirical covariance of $(\hat\mu,\hat\sigma)$ across
   replications, and confirm it matches $I(\mu,\sigma)^{-1}/n$ for both parameters (not
   just the mean).
   *Provenance:* adapted from Wasserman, *All of Statistics*, Ch. 9, Exercise 8 (overlap
   med — we adopt their joint-information target and add the section's own simulation
   pattern, which their solution only codes for a scalar reparameterization).
7. [short-code] **EM for a Two-Component Gaussian Mixture, Coded End to End.** Derive the
   GMM M-step for $\mu_k$ and $\pi_k$ (as the existing exercise asks), then implement
   EM on synthetic two-component data and confirm the log-likelihood is non-decreasing
   across iterations, as the section's own EM proposition guarantees.
   *Provenance:* original (existing exercise 10, retitled, with its derivation extended
   into a runnable implementation and a monotonicity check).

---

## chapter_mdl-probability-statistics/mdl-bayesian-computation.md — Bayesian Computation

**Topic:** Approximating posterior averages when the posterior is known only up to a
constant: importance sampling, Metropolis MCMC, the Laplace approximation, and
mean-field variational inference.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — every existing
exercise (prior-variance sensitivity, a deliberately bad importance proposal, a
Metropolis step-size sweep against ESS, a full-covariance VI upgrade, and a bimodal
posterior stress test) is concrete, checkable, and free of clarity issues per the prior
review. This file's actual defect is bibliographic (zero cross-references, zero citations,
and no Discussions section at all — the only file in the group missing one), not
exercise quality; that is a formatting note for the reform, not a reason to touch the
exercises.

**External sources found:**
- Gutmann, *Pen and Paper Exercises in Machine Learning* (arXiv:2206.13446), Exercise 9.1
  "Importance sampling to estimate tail probabilities" (based on Robert & Casella, 2010,
  Exercise 3.5) — https://arxiv.org/abs/2206.13446
- Gutmann, same collection, Exercise 9.6 "Rejection sampling" (based on Robert & Casella,
  2010, Exercise 2.8), Exercise 9.8 "Basic Markov chain Monte Carlo inference," Exercise
  9.9 "Bayesian Poisson regression," and Exercise 9.10 "Mixing and convergence of
  Metropolis-Hastings MCMC" — same URL.
- Gutmann, same collection, Exercises 10.1-10.2 "Mean field variational inference I/II"
  and 10.3-10.4 "Variational posterior approximation I/II" — same URL.
- Harvard Stat 110, MIT 18.05, and MIT 6.041 do not cover MCMC, importance sampling, or
  variational inference at all — a finding, not a failure: computational Bayesian
  inference sits above the introductory-probability level these three courses target.
  Murphy's *Probabilistic Machine Learning: Advanced Topics* covers the right material
  but supplies worked derivations rather than a freely available problem set with
  solutions, so it is not cited as an adoptable source here.

**Proposed problem set** (8 problems, our reference format):
1. [conceptual] **Prior Variance and Posterior Sensitivity.** Change the prior standard
   deviation from $2$ to $0.5$ and $10$; compare MAP, posterior mean, covariance,
   posterior predictive, and importance-sampling ESS, and identify which quantities are
   most sensitive at this sample size.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **Why a Full-Support Proposal Can Still Fail.** Use an importance
   proposal with covariance $0.05I$ across ten seeds; explain why a plausible estimate in
   one run does not repair the proposal's poor finite-sample coverage.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [short-code] **Metropolis Step Size versus Effective Sample Size.** Sweep the
   Metropolis proposal multiplier over $\{0.05,0.2,0.5,1.1,3,10\}$; plot acceptance rate
   against ESS per log-joint evaluation, and explain why maximizing acceptance is the
   wrong objective.
   *Provenance:* original (existing exercise 3, retitled; no change).
4. [short-code] **Full-Covariance Variational Family via Cholesky.** Replace the
   mean-field variational family with a full-covariance Gaussian parameterized by a
   Cholesky factor with positive diagonal, and compare the fitted correlation with the
   grid reference value.
   *Provenance:* original (existing exercise 4, retitled; no change).
5. [conceptual] **Diagnosing a Bimodal Posterior.** Construct a bimodal 1-D posterior via
   a $\theta\mapsto-\theta$-invariant likelihood, and show how a local Laplace
   approximation, mean-field reverse-KL VI, and a poorly initialized MCMC chain can each
   report only one mode; identify which diagnostics reveal the problem.
   *Provenance:* original (existing exercise 5, retitled; no change).
6. [short-code] **Importance Sampling a Tail Probability.** Use a Gaussian proposal to
   estimate $P(Z>4)$ for standard normal $Z$ by self-normalized importance sampling;
   compare the estimate and its ESS against a plain Monte Carlo baseline, and show the
   variance blow-up once the proposal's tail is lighter than the target's.
   *Provenance:* adapted from Gutmann, *Pen and Paper Exercises in Machine Learning*,
   Exercise 9.1 (overlap med — same tail-probability importance-sampling setup,
   retargeted at this section's own ESS/weight diagnostics rather than Gutmann's
   analytic treatment).
7. [conceptual] **Mean-Field Variational Inference from Scratch.** For a two-parameter
   Gaussian joint (unknown mean and precision) matching the section's own running model
   family, derive the mean-field factorized update equations directly from the ELBO by
   calculus (no reparameterization trick), and confirm the fixed point matches the
   reparameterized-gradient solution the section computes numerically.
   *Provenance:* inspired by Gutmann, *Pen and Paper Exercises in Machine Learning*,
   Exercises 10.1-10.2 (overlap low — Gutmann's exercises use a different joint model; we
   retarget to a Gaussian mean/precision model matching this section's own example).
8. [extended] **Build a New Nonconjugate Posterior and Compare All Four Methods.** Choose
   a small nonconjugate model distinct from the section's logistic-regression example
   (e.g., a two-parameter Weibull survival model), implement grid quadrature as ground
   truth, then implement and compare importance sampling, Metropolis MCMC, the Laplace
   approximation, and mean-field VI on it, reporting the same summary comparison the
   section itself builds.
   *Provenance:* original (project-scale synthesis of the section's own four methods on
   a new model; the comparison-table format is inspired by the section's own
   :numref:`sec_mdl-bayes-decision-map`, but no external source targets this exact
   combination).

---

## chapter_mdl-probability-statistics/mdl-statistics.md — Statistics

**Topic:** Estimators (bias, variance, MSE, consistency, efficiency), the bias-variance
decomposition and the weak law of large numbers, hypothesis testing (permutation tests,
power, multiple testing), confidence intervals, and the bootstrap.
**Current exercises:** 9; disposition: keep 8, rewrite 0, drop 1 — a defect-free set per
the prior review (ex1's bootstrap-failure walkthrough was singled out as "unusually well
explained"), covering the Uniform$(0,\theta)$ estimator/bootstrap-failure pairing,
re-deriving the bias-variance identity, the $n-1$ correction, estimator shrinkage,
$p$-value misinterpretation, CI pathologies at $N=2$, bootstrapping the mean, and
Bonferroni; the one drop is a one-line arithmetic restatement of the sample-size scaling
already given in closed form in the text.

**External sources found:**
- Wasserman, *All of Statistics*, Ch. 8 (Bootstrap), Exercises 4, 5, 7, 8 — a
  stars-and-bars count of bootstrap-resample configurations; the exact bootstrap
  variance-inflation factor $(2n-1)/n$ for the sample mean; a real-data
  standard-error/CI computation; and the classic
  $P(\hat\theta^*=\hat\theta\mid\hat\theta)\to1-e^{-1}\approx0.632$ fact —
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_8_solutions.pdf
- Wasserman, *All of Statistics*, Ch. 9 (Parametric Inference), Exercises 2, 3 — the MLE
  of $\textrm{Uniform}(a,b)$ as order statistics (the non-regular case underlying this
  section's own exercise 1) and a delta-method CI for a quantile functional —
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_9_solutions.pdf
- Wasserman, *All of Statistics*, Ch. 10 (Hypothesis Testing), Exercises 7, 8 — a
  permutation test versus a Wald test on real word-length data (the Mosteller-Wallace
  "twain vs. snodgrass" disputed-authorship comparison), and a $z$-test power/sample-size
  derivation matching this section's own :eqref:`eq_mdl-power-sample-size` —
  https://parsiad.ca/assets/pdf/all_of_statistics_chapter_10_solutions.pdf
- Harvard Stat 110 offers no strategic-practice set on hypothesis testing, confidence
  intervals, or the bootstrap — a finding, not a failure: Stat 110 is a probability course
  and stops before frequentist inference, which is exactly the material this section
  covers.

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **The Uniform-Support MLE and Why Its Bootstrap Fails.** For
   $X_1,\ldots,X_n\sim\textrm{Unif}(0,\theta)$, compare $\hat\theta=\max_iX_i$ against
   $\tilde\theta=\tfrac2n\sum_iX_i$ on bias/variance/MSE, then explain why bootstrapping
   $\hat\theta$ fails (a resample contains the largest observation with probability
   $\to1-e^{-1}$, putting a point mass exactly at $\hat\theta$).
   *Provenance:* original (existing exercise 1, retitled; no change); Wasserman Ch. 9
   Ex. 2 and Ch. 8 Ex. 8 independently derive the same MLE and the same $1-e^{-1}$
   bootstrap-collision constant, confirming this is the standard textbook example.
2. [conceptual] **Re-deriving the Bias-Variance Decomposition.** Expand
   $\mathbb E[(\hat\theta_n-\theta)^2]$ directly via $\mathbb
   E[\hat\theta_n^2]-2\theta\mathbb E[\hat\theta_n]+\theta^2$ and confirm it agrees with
   the add-and-subtract proof in the text.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [short-code] **The Degrees-of-Freedom Correction, Verified.** Rerun the bias-variance
   decomposition check using the unbiased ($\texttt{ddof=1}$) variance estimator instead
   of the plug-in one, and explain which denominator makes the identity exact to
   floating-point precision and why.
   *Provenance:* original (existing exercise 3, retitled; no change).
4. [short-code] **Estimator Spread Shrinks Like $1/\sqrt n$.** Shrink the per-dataset
   size $n$ in the sampling-distribution simulation and confirm the spread of $\hat\theta$
   widens like $\sigma/\sqrt n$; repeat with the biased estimator
   $\hat\theta=\max_iX_i$ for $\textrm{Unif}(0,\theta)$ and watch its center shift away
   from $\theta$.
   *Provenance:* original (existing exercise 4, retitled; no change).
5. [conceptual] **A $p$-value Is Not $P(H_0\mid\textrm{data})$.** Given a reported
   $p=0.5$, explain in terms of $P(\textrm{data}\mid H_0)$ versus $P(H_0\mid\textrm{data})$
   why this is not evidence that $H_0$ is true, and describe a situation where a large
   $p$-value reflects only low power.
   *Provenance:* original (existing exercise 5, retitled; no change).
6. [short-code] **A Short Interval Is Not a Precise One.** Run the confidence-interval
   code with $N=2$ and $\alpha=0.5$ for 100 independently generated datasets; identify the
   extremely short intervals far from the true mean, and explain why this does not
   contradict the $1-\alpha$ coverage guarantee.
   *Provenance:* original (existing exercise 7, retitled; no change).
7. [short-code] **Permutation Testing on Real Word-Length Data.** Reproduce the classic
   Mosteller-Wallace two-sample word-length comparison ("twain" vs. "snodgrass") with a
   permutation test in the style of this section's own model-A/B example, verify the
   permutation $p$-value against a large-sample Wald-test approximation, and discuss why
   the permutation test is preferable at this sample size.
   *Provenance:* adapted from Wasserman, *All of Statistics*, Ch. 10, Exercise 7 (overlap
   high — same dataset and the same Wald-versus-permutation comparison; cite Wasserman on
   adoption).

---

## chapter_mdl-probability-statistics/mdl-concentration-generalization.md — Concentration and Generalization

**Topic:** The Chernoff method, Hoeffding's lemma/inequality, sub-Gaussian and
sub-exponential variables, Bernstein's inequality, high-dimensional norm/angle
concentration, uniform convergence via the union bound and Rademacher complexity, and
interpolation/double descent.
**Current exercises:** 8; disposition: keep 8, rewrite 0, drop 0 — the strongest existing
set in the chapter by the prior review's own account: eight problems spanning
sub-Gaussianity proofs, a weighted-range Hoeffding extension, McDiarmid's inequality with
a bootstrap application, $\ell_1$-ball Rademacher complexity, and two full double-descent
numerical experiments, several carrying real citations (McDiarmid 1989, Dwork et al.
2015). Nothing here needs replacing; the one addition below fills a genuine gap — a
dedicated Bernstein/Bennett-inequality derivation, which the section states but does not
prove and none of the 8 exercises practice.

**External sources found:**
- Wainwright, *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*, Ch. 2 "Basic Tail
  and Concentration Bounds" — 22 named exercises (2.1-2.22) including "Tightness of
  Inequalities," "Sharp Sub-Gaussian Parameter for Bounded Random Variable," "Bennett's
  Inequality," "Bernstein and Expectations," "Upper Bounds for Sub-Gaussian Maxima," and
  "Hanson-Wright Inequality" — the closest thing to a canonical problem set for this
  section's entire theory half; exercise titles verified via
  https://high-dimensional-statistics.github.io/
- Vershynin, *High-Dimensional Probability: An Introduction with Applications in Data
  Science*, Ch. 2 ("Concentration of Sums of Independent Random Variables") — a freely
  hosted PDF whose Exercise 2.6.4 asks the reader to deduce Hoeffding's inequality for
  bounded variables from the general sub-Gaussian result, a second independent
  verification of this section's Hoeffding/sub-Gaussian material, pitched slightly more
  abstractly — https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf
- Harvard Stat 110, MIT 18.05, and MIT 6.041 do not reach exponential concentration,
  Rademacher complexity, or double descent — expected, since these are graduate-level
  topics well outside an introductory-probability syllabus; not a gap in this catalog, just
  outside the suggested sources' scope.

**Proposed problem set** (7 problems, our reference format):
1. [conceptual] **Rademacher Variables Are Sub-Gaussian.** Show $\cosh\lambda\le
   e^{\lambda^2/2}$ by comparing Taylor series term by term, and conclude via Chernoff
   that an average of $n$ fair random signs satisfies $P(|\bar\varepsilon|\ge
   t)\le2e^{-nt^2/2}$.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **Hoeffding with Unequal Ranges.** Redo the Hoeffding proof for
   independent $X_i\in[a_i,b_i]$ to obtain the weighted bound, and check it reduces to the
   standard form when all ranges agree.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [conceptual] **McDiarmid's Inequality and the Bootstrap.** Recover Hoeffding's
   inequality from McDiarmid's bounded-differences result, then argue that a
   bounded-differences statistic's deviation from its mean concentrates at rate
   $e^{-2nt^2/c^2}$, checking that the sample median does *not* have small bounded
   differences in general.
   *Provenance:* original (existing exercise 3, retitled; no change).
4. [conceptual] **Rademacher Complexity of an $\ell_1$ Ball.** Compute the Rademacher
   complexity of the $\ell_1$-norm-bounded linear class using $\ell_1$-$\ell_\infty$
   duality, bounding the expected max of $d$ sub-Gaussian coordinates, and explain why
   dimension now enters only logarithmically.
   *Provenance:* original (existing exercise 4, retitled; no change).
5. [short-code] **Ridge Regression Melts the Double-Descent Peak.** Rerun the
   double-descent sweep with ridge-regularized weights for
   $\lambda\in\{10^{-8},10^{-4},10^{-2},1\}$, watch the interpolation peak shrink as
   $\lambda$ grows, and explain the mechanism via the ridge term's effect on
   $\sigma_{\min}^2$.
   *Provenance:* original (existing exercise 5, retitled; no change).
6. [short-code] **Label Noise and the Double-Descent Peak Height.** Vary the label noise
   in the double-descent experiment and measure the height and location of the
   interpolation peak, confirming the peak's location is a rank condition (fixed at
   $p=n$) while its height scales with the noise variance.
   *Provenance:* original (existing exercise 6, retitled; no change).
7. [conceptual] **Bennett's Inequality from the Chernoff Method.** Derive Bennett's
   inequality for a sum of independent, bounded, mean-zero variables by bounding each
   term's MGF and optimizing the Chernoff exponent, then confirm it reduces to
   Hoeffding's bound when the variance proxy $\sigma^2$ is set to $(b-a)^2/4$.
   *Provenance:* adapted from Wainwright, *High-Dimensional Statistics*, Ch. 2, Exercise
   2.7 "Bennett's Inequality" (overlap high — same inequality and derivation strategy;
   cite Wainwright on adoption).

---

## chapter_mdl-probability-statistics/mdl-naive-bayes.md — Naive Bayes

**Topic:** Bayes-rule classification, the conditional-independence assumption, log-space
computation and linear decision boundaries, Laplace-smoothed counting estimation, the
MNIST worked example, and calibration/confusion-matrix/bootstrap evaluation.
**Current exercises:** 6; disposition: keep 5, rewrite 0, drop 1 — a set the prior review
found defect-free (exercise 6 was singled out for stating the expected numeric answer as
a self-check), spanning naive Bayes' failure on XOR, the consequence of skipping Laplace
smoothing, the posterior-as-softmax derivation, a smoothing-strength sweep, and the
template-overlap/confusion-matrix cosine check; the one drop is a graphical-models
"explain why adding an edge would fix XOR" prompt that is discursive rather than
checkable and largely restates exercise 1's point.

**External sources found:**
- CMU 10-601, Spring 2015 (instructor Nina Balcan), Homework 3 "Implementing Naive
  Bayes," Problem 1 — a full text-classification assignment (*The Economist* vs. *The
  Onion*, $\approx$26,000-word vocabulary), Beta$(2,1)$-MAP-smoothed Bernoulli naive
  Bayes trained and evaluated in log space, with parts (g)-(i) comparing train/test
  error on the full and a shrunk training set, and inspecting the most-diagnostic words
  per class — the single closest real-course analogue to this section's own MNIST
  pipeline — http://www.cs.cmu.edu/~ninamf/courses/601sp15/hw/homework3.pdf
- Stanford CS229 (public course), Problem Set #1, Problem 4 "Naive Bayes" — derive the
  joint log-likelihood, show the MLE-parameter formulas match the standard counting
  rule, and prove the naive-Bayes decision rule is a linear classifier — a clean
  pencil-and-paper companion to this section's own linear-boundary derivation —
  https://see.stanford.edu/materials/aimlcs229/problemset1.pdf
- MIT 18.05/6.041 and Harvard Stat 110 do not cover naive Bayes at all — a finding, not a
  failure; naive Bayes is a machine-learning topic, not a probability-theory one, so the
  two ML-course sources above are far better matches than any pure-probability course.

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **Naive Bayes Fails on XOR.** For the dataset
   $\{(0,0),(0,1),(1,0),(1,1)\}$ labeled by XOR, compute the naive Bayes estimates
   $p(y)$ and $p(x_i\mid y)$, and determine which assumption the classifier's failure
   violates.
   *Provenance:* original (existing exercise 1, retitled; no change).
2. [conceptual] **The Cost of Skipping Laplace Smoothing.** Without smoothing, determine
   the log-score for a class whose training set never showed a given feature value, and
   explain the resulting failure.
   *Provenance:* original (existing exercise 2, retitled; no change).
3. [conceptual] **The Posterior Is a Softmax of the Scores.** Derive the exact posterior
   $p(y\mid\mathbf x)=\exp(s_y(\mathbf x))/\sum_{y'}\exp(s_{y'}(\mathbf x))$ by
   normalizing the affine class scores, and connect this to softmax regression fitting
   the same functional form directly.
   *Provenance:* original (existing exercise 4, retitled; no change).
4. [short-code] **Smoothing Strength versus MNIST Accuracy.** Generalize the pseudocount
   to $\alpha$, estimating $p(x_i{=}1\mid y)$ as $(n_{iy}+\alpha)/(n_y+2\alpha)$; report
   test accuracy for $\alpha\in\{0,1,10\}$ and explain the NaN failure mode at $\alpha=0$
   when a zero-probability pixel is off in a test image.
   *Provenance:* original (existing exercise 5, retitled; no change).
5. [short-code] **Template Overlap and the Confusion Matrix.** Flatten the ten smoothed
   templates into vectors, compute all pairwise cosine similarities, and check how well
   the most-similar pairs match the top confusions in the confusion matrix.
   *Provenance:* original (existing exercise 6, retitled; no change).
6. [short-code] **Naive Bayes on a Bag-of-Words Text Corpus.** Train a Bernoulli-event
   naive Bayes classifier, MAP-smoothed with a $\textrm{Beta}(2,1)$ prior in place of
   add-one, on a small labeled two-class text corpus (following this section's own MNIST
   recipe); report train/test error, and explain why the gap between them widens on a
   smaller training subset.
   *Provenance:* adapted from CMU 10-601, Spring 2015, Homework 3, Problem 1 parts (g)-(i)
   (overlap high — same Beta$(2,1)$-MAP Bernoulli-naive-Bayes-on-text setup and the same
   train/test-gap and small-vs-large-training-set comparison; cite CMU 10-601 on
   adoption).
