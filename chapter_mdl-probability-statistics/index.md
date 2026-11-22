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

::: {.cols .vc}
::: {.col}
- **The Running Coin ($9$ Heads in $13$ Flips):** $\hat{\theta} = 9/13 \approx 0.692$, yet a fair coin ($\theta = 0.5$) yields $\ge 9$ heads $13.3\%$ of the time—just like test accuracy on a finite holdout set.
- **Forward vs. Backward Reasoning:** *Probability* (§{sec_mdl-random_variables}–§{num:sec_mdl-exponential-family}) maps a known model to outcome frequencies; *statistics* (§{sec_mdl-statistics}–§{num:sec_mdl-bayesian-computation}) maps finite data back to estimates, error bars, and posteriors.
- **Post-Selection Guarantees (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-generalization-bounds}):** Picking the best model across validation runs is multiple testing in disguise and requires uniform convergence bounds.
:::

::: {.col .fig .big}
@fig:ch36-prob-coin-thread
:::
:::
:::

::: {.slide title="Four Pillars of Probabilistic Learning"}
[Chapter Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Distributions & Unified Form (§{sec_mdl-random_variables}–§{num:sec_mdl-exponential-family}):** Densities, Jacobians, named distributions, and the exponential family $h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})}$.
- **2. Estimation, Tests & Likelihood (§{sec_mdl-statistics}–§{num:sec_mdl-maximum_likelihood}):** Bias–variance trade-off, $p$-values, confidence intervals, MLE as KL projection, and Fisher information.
- **3. Latent Models & Posteriors (§{sec_mdl-latent-em-elbo}–§{num:sec_mdl-bayesian-computation}):** Marginal likelihood $\log p(\mathbf{x})$, Jensen's inequality, ELBO, EM, Laplace approximation, VI, and MCMC.
- **4. Concentration & Generalization (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-naive_bayes}):** Chernoff/Hoeffding bounds, high-$d$ shell geometry, Rademacher complexity, and Naive Bayes.
:::

::: {.col .fig .big}
@fig:ch36-prob-four-pillars
:::
:::
:::

::: {.slide title="Roadmap: Distributions to Posteriors (§{sec_mdl-random_variables}–§{num:sec_mdl-bayesian-computation})"}
[Section Guide (Part I)]{.kicker}

::: {.cols .vc}
::: {.col}
- **§{sec_mdl-random_variables}–§{num:sec_mdl-joint-distributions} (Continuous Calculus):** $F'(x) = p(x)$, inverse-transform sampling $X = F^{-1}(U)$, Eve's law, and Jacobian volume stretch $|\det J_g|^{-1}$.
- **§{sec_mdl-distributions}–§{num:sec_mdl-exponential-family} (Catalog & Exponential Family):** Binomial-to-Poisson and CLT limits, Schur Gaussian conditioning, cumulant derivatives $\nabla A = \mathbb{E}[T]$, and conjugate updates.
- **§{sec_mdl-statistics}–§{num:sec_mdl-bayesian-computation} (Inference & Latent Variables):** $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$, paired tests, bootstrap, Fisher curvature $nI(\theta)$, EM via ELBO, and MCMC/VI posteriors.
:::

::: {.col .fig .big}
@fig:ch36-prob-roadmap-p1
:::
:::
:::

::: {.slide title="Roadmap: Guarantees and Synthesis (§{sec_mdl-concentration-generalization}–§{num:sec_mdl-naive_bayes})"}
[Section Guide (Part II)]{.kicker}

::: {.cols .vc}
::: {.col}
- **§{sec_mdl-concentration-generalization} Concentration Inequalities:** Upgrades Chebyshev's $1/(n\varepsilon^2)$ bound to exponential decay $\exp(-2n\varepsilon^2)$ via Chernoff/Hoeffding and proves why high-$d$ Gaussians concentrate on a thin shell of radius $\sqrt{d}$.
- **§{sec_mdl-generalization-bounds} Generalization Bounds:** Bounds worst-case generalization error $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)|$ across model search via union bounds, Rademacher complexity, and benign overfitting.
- **§{sec_mdl-naive_bayes} Naive Bayes Classifier:** Combines categorical/Bernoulli likelihoods and Laplace add-one smoothing into an end-to-end generative image classifier.
:::

::: {.col .fig .big}
@fig:ch36-prob-roadmap-p2
:::
:::
:::

::: {.slide title="How to Navigate Chapter §{num:chap_mdl-probability-statistics}"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Core theoretical backbone:** Read **§{sec_mdl-random_variables}–§{num:sec_mdl-maximum_likelihood}** in sequence to see how continuous densities and the exponential family unify every supervised loss function and statistical error bar in the main text.
- **Fast-track option:** If you are already comfortable with §{sec_prob} (preliminary discrete probability), skim the basic moment algebra in §{sec_mdl-random_variables}–§{num:sec_mdl-joint-distributions} and focus on inverse-transform sampling, Eve's law, Jacobian determinants, and Gaussian conditioning.
:::

::: {.col}
- **Modular advanced topics:** **§{sec_mdl-bayesian-computation} (Bayesian Computation)** and **§{sec_mdl-naive_bayes} (Naive Bayes)** are self-contained and can be skipped on a first pass without breaking later dependencies.
- **Connections to other math chapters:** Linear-algebraic spectra and positive definiteness come from **Chapter §{num:chap_mdl-linear-algebra}**, multivariable Jacobians and convexity from **Chapters §{num:chap_mdl-calculus}–§{num:chap_mdl-optimization}**, and full entropy/KL theory continues in **Chapter §{num:chap_mdl-information-theory}**.
:::
:::

::: {.d2l-note}
**Up next (§{sec_mdl-random_variables} Random Variables):** Why continuous outcomes have zero point probability, how densities and CDFs relate via $F'(x) = p(x)$, and why heavy tails break sample averages.
:::
:::

