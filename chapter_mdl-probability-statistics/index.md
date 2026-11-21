# Probability and Statistical Learning
:label:`chap_mdl-probability-statistics`

Throughout the main text, we rely on probability at almost every turn without
pausing to justify the rules behind it. We train regression models with squared
error and classifiers with cross-entropy (:numref:`sec_linear_regression`,
:numref:`sec_softmax`), report an accuracy measured on a few thousand test
examples, and pick the best model out of dozens of candidates by comparing
validation scores (:numref:`sec_generalization_basics`). Every one of those
habits is an inference drawn from a finite sample of data. This chapter builds
the mathematical foundations that explain when those inferences hold and when
they can mislead us.

A single, concrete example will accompany us throughout the chapter: a coin
that lands heads $9$ times in $13$ flips. The natural estimate of its bias is
$9/13 \approx 0.69$, yet a completely fair coin produces a result at least this
extreme more than one time in eight. A test accuracy is the very same coin,
flipped once per test example, and the very same uncertainty clings to it.

Sorting out that uncertainty calls for four distinct ideas. First, we need a
*distribution* to describe how random outcomes vary. Second, we need an
*estimator* to turn a sample into a number, along with a sense of how much that
number would wobble if we drew a fresh sample. Third, we need a *posterior* to
describe what we believe about unknown parameters after seeing the data we
actually have. Notice how easily the second and third ideas are confused: the
spread of an estimate across hypothetical samples we did not draw is not the
same thing as the spread of a posterior conditioned on the sample in front of
us, even though the two often produce similar-looking error bars. Fourth, we
need a *generalization bound* to tell us when good performance on a finite
sample genuinely guarantees good performance on new data.

We take these four ideas in order. We begin by giving distributions a proper
language: densities for a single random quantity
(:numref:`sec_mdl-random_variables`), then joint distributions for several
quantities at once, together with the change-of-variables rule that tracks how
a transformation reshapes a density (:numref:`sec_mdl-joint-distributions`).
With those tools in hand, we tour the standard named distributions
(:numref:`sec_mdl-distributions`) and discover that nearly all of them belong
to a single unified template, the exponential family
(:numref:`sec_mdl-exponential-family`). That shared structure is the reason
linear, logistic, and softmax regression all produce the same clean gradient:
"prediction minus observation."

The middle of the chapter turns to estimation and inference.
:numref:`sec_mdl-statistics` develops error bars, hypothesis tests, and
confidence intervals, while :numref:`sec_mdl-maximum_likelihood` derives the
book's training losses from the single principle of maximum likelihood. Linking
the two is the Fisher information: the expected curvature of the negative
log-likelihood, whose inverse governs the variance of fitted parameters.
Posteriors then enter naturally through models with hidden variables, where the
likelihood can no longer be maximized directly and must instead be bounded from
below (:numref:`sec_mdl-latent-em-elbo`), followed by
:numref:`sec_mdl-bayesian-computation` for working with posteriors that have no
closed-form solution.

Guarantees come last. :numref:`sec_mdl-concentration-generalization` proves
Hoeffding's inequality and related tail bounds that hold at any sample size,
and :numref:`sec_mdl-generalization-bounds` asks what survives of those
guarantees once we select a model after looking at the data, supplying the
theory behind :numref:`chap_classification_generalization`. To close the
chapter, :numref:`sec_mdl-naive_bayes` puts distributions, estimation, and
posteriors to work together on real images by fitting a simple classifier from
pixel counts alone.

A brief reading note: the opening sections revisit parts of :numref:`sec_prob`
with proofs and continuous densities, so if you already know that material you
can skim them. :numref:`sec_mdl-bayesian-computation` and
:numref:`sec_mdl-naive_bayes` are self-contained and can be skipped on a first
pass, as no later section depends on them. Entropy and the KL divergence appear
here only as working tools; their full theory lives in
:numref:`chap_mdl-information-theory`. Measure-theoretic probability, time
series, and nonparametric Bayesian models are left outside our scope.

```toc
:maxdepth: 2

mdl-random-variables
mdl-joint-distributions
mdl-distributions
mdl-exponential-family
mdl-statistics
mdl-maximum-likelihood
mdl-latent-em-elbo
mdl-bayesian-computation
mdl-concentration-generalization
mdl-generalization-bounds
mdl-naive-bayes
further-reading
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §36]{.kicker}

Probability and Statistical Learning: Foundations of Inference Under Uncertainty<br>**densities & Jacobians · exponential families · estimation & testing · MLE & ELBO · Bayesian computation · concentration & generalization**
:::
:::

::: {.slide title="Why Deep Learning Needs Statistical Foundations"}
[Motivation]{.kicker}

Every decision in deep learning—training with cross-entropy or squared error, reporting an accuracy on $1{,}000$ test examples, or picking the best model from 20 runs—is an inference from a finite random sample.

::: {.cols}
::: {.col}
- **The Running Coin ($9$ Heads in $13$ Flips):** The natural bias estimate is $\hat{\theta} = 9/13 \approx 0.692$, yet a fair coin ($\theta = 0.5$) produces $\ge 9$ heads $1093/8192 \approx 13.3\%$ of the time. A test accuracy is the exact same coin flipped once per test example.
- **Forward vs. Backward Reasoning:** *Probability* (§36.1–36.4) maps a known distribution to outcome frequencies; *statistics* (§36.5–36.8) maps observed finite data back to parameter estimates, error bars, and posteriors.
:::

::: {.col}
- **Sampling Spread vs. Posterior Spread:** The scatter of an estimator $\hat{\theta}_n$ across hypothetical datasets we *did not draw* (frequentist confidence interval) is distinct from the spread of a posterior $p(\theta \mid \mathcal{D})$ conditioned on the data *in hand* (Bayesian credible interval).
- **Post-Selection Guarantees (§36.9–36.10):** Choosing a model *after* inspecting validation scores is multiple testing in disguise and requires uniform convergence bounds.
:::
:::

::: {.d2l-note}
A single mathematical thread connects continuous densities, the exponential family, maximum likelihood losses, variational bounds, and finite-sample generalization guarantees.
:::
:::

::: {.slide title="Four Pillars of Probabilistic Learning"}
[Chapter Architecture]{.kicker}

Chapter 36 progresses from the calculus of continuous distributions to estimation, latent-variable inference, and distribution-free generalization bounds:

| Pillar | Sections | Central Mathematical Objects | Deep Learning Payoff |
| :--- | :--- | :--- | :--- |
| **1. Distributions & Unified Form** | **§36.1–36.4** | PDFs, CDFs, Jacobians, named distributions, exponential family $h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})}$ | Inverse-transform & Cholesky sampling, normalizing flows, GLM gradient $\mathbf{x}(\boldsymbol{\mu} - T(y))^\top$ |
| **2. Estimation, Tests & Likelihood** | **§36.5–36.6** | Bias–variance decomposition, $p$-values, power, confidence intervals, MLE, Fisher information | Test-set sizing, paired model evaluation, bootstrap SEs, cross-entropy & weight decay |
| **3. Latent Models & Posteriors** | **§36.7–36.8** | Marginal likelihood $\log p(\mathbf{x})$, Jensen's inequality, ELBO, EM, Laplace, VI, MCMC | Gaussian mixtures, variational autoencoders (VAEs), posterior predictive uncertainty |
| **4. Concentration & Generalization** | **§36.9–36.11** | Chernoff & Hoeffding bounds, high-$d$ geometry, union bound, Rademacher complexity | Finite-sample risk bounds after model selection, end-to-end Naive Bayes image classifier |
:::

::: {.slide title="Roadmap: Distributions to Posteriors (§36.1–36.8)"}
[Section Guide (Part I)]{.kicker}

| Section | Core Theorems & Derivations | Concrete Running Benchmark |
| :--- | :--- | :--- |
| **§36.1 Random Variables** | $F'(x) = p(x)$, $X = F^{-1}(U)$, Markov & Chebyshev bounds | Dart distance $R$ on unit disk ($p(r)=2r$); Cauchy vs. Gaussian averages |
| **§36.2 Joint Distributions** | Marginals, conditioning, Eve's law, Jacobian $\|\det J_g\|^{-1}$ | Dart coordinates $(X, Y)$ (dependent yet $\mathrm{Cov}=0$); log-normal density |
| **§36.3 Distributions** | Poisson limit, exponential waits, CLT, MVN Schur conditioning | $9/13$ coin flips; server at $4\text{ req/min}$; bivariate $\boldsymbol{\Sigma}$ with rows $(2,1),(1,2)$ |
| **§36.4 Exponential Family** | $\nabla A = \mathbb{E}[T]$, $\nabla^2 A = \mathrm{Cov}(T)$, maximum entropy, conjugacy | Loaded die ($\mathbb{E}=4.5$); Gumbel-max softmax; $\mathrm{Beta}(10, 5)$ coin posterior |
| **§36.5 Statistics** | $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$, uniform $p$-values, power, bootstrap | $912/1000$ vs. $921/1000$ paired test accuracy; shrunk mean $0.65\bar{x}$ |
| **§36.6 Maximum Likelihood** | KL projection, score $\mathbb{E}[s]=0$, Fisher information, Cramér–Rao | Curvature $nI(\theta)$ of coin NLL; MAP as $L_2$/$L_1$ regularization; BIC |
| **§36.7–36.8 EM, ELBO & Bayes** | EM monotonic ascent via ELBO; Laplace, variational inference, MCMC | 2-component GMM on Old Faithful; Bayesian logistic regression posteriors |
:::

::: {.slide title="Roadmap: Guarantees and Synthesis (§36.9–36.11)"}
[Section Guide (Part II)]{.kicker}

::: {.cols}
::: {.col}
- **§36.9 Concentration Inequalities:** Upgrades Chebyshev's polynomial $1/(n\varepsilon^2)$ bound to exponential decay $\exp(-2n\varepsilon^2)$ via Markov's inequality on $e^{\lambda X}$ (Chernoff & Hoeffding bounds), and proves why high-dimensional Gaussian vectors concentrate on a thin spherical shell of radius $\sqrt{d}$.
- **§36.10 Generalization Bounds:** Explains why selecting the best model $\hat{h}$ from a class $\mathcal{H}$ invalidates single-test guarantees, and bounds $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)|$ via union bounds and Rademacher complexity.
:::

::: {.col}
- **§36.11 Naive Bayes Classifier:** Puts distributions, maximum likelihood, and Dirichlet/Beta conjugate smoothing ($\text{add-}1$ Laplace rule) to work together in a closed-form generative image classifier trained from pixel counts alone.
- **One Coin Across Eleven Sections:** The $9$-heads-in-$13$-flips coin anchors binomial tail probabilities (§36.3), log-odds & Beta posteriors (§36.4), Wald vs. exact intervals (§36.5), Fisher information (§36.6), and Hoeffding bounds (§36.9).
:::
:::
:::

::: {.slide title="How to Navigate Chapter 36"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core theoretical backbone:** Read **§36.1–36.6** in sequence to see how continuous densities and the exponential family unify every supervised loss function and statistical error bar in the main text.
- **Fast-track option:** If you are already comfortable with §1.6 (preliminary discrete probability), skim the basic moment algebra in §36.1–36.2 and focus on inverse-transform sampling, Eve's law, Jacobian determinants, and Gaussian conditioning.
:::

::: {.col}
- **Modular advanced topics:** **§36.8 (Bayesian Computation)** and **§36.11 (Naive Bayes)** are self-contained and can be skipped on a first pass without breaking later dependencies.
- **Connections to other math chapters:** Linear-algebraic spectra and positive definiteness come from **Chapter 33**, multivariable Jacobians and convexity from **Chapters 34–35**, and full entropy/KL theory continues in **Chapter 37**.
:::
:::

::: {.d2l-note}
**Up next (§36.1 Random Variables):** Why continuous outcomes have zero point probability, how densities and CDFs relate via $F'(x) = p(x)$, and why heavy tails break sample averages.
:::
:::

