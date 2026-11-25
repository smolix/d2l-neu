# Probability and Statistical Learning
:label:`chap_mdl-probability-statistics`

Throughout the main text, we train regression models with squared error and
classifiers with cross-entropy (:numref:`sec_linear_regression`,
:numref:`sec_softmax`), report accuracies on finite test sets, and pick models
by comparing validation scores (:numref:`sec_generalization_basics`). Every
one of those steps draws an inference from a finite sample. Consider a coin
that lands heads $9$ times in $13$ flips: its empirical bias is
$9/13 \approx 0.69$, yet a fair coin produces at least nine heads more than
one time in eight. A test accuracy is that same coin flipped once per test
example, and this chapter develops the mathematics that explains when finite
samples support reliable conclusions and when they mislead us.

Sorting out sample uncertainty rests on four distinct ideas: a *distribution*
describing how outcomes vary; an *estimator* mapping a sample to a number,
whose spread across hypothetical samples differs from a *posterior*
conditioned on the single sample in front of us; and a *generalization bound*
certifying when low error on a finite sample guarantees low error on fresh
data.

We take these four ideas in order. First, we establish continuous densities
(:numref:`sec_mdl-random_variables`), joint distributions and Jacobian
transformations (:numref:`sec_mdl-joint-distributions`), the standard named
distributions (:numref:`sec_mdl-distributions`), and their unification in the
exponential family (:numref:`sec_mdl-exponential-family`), which explains why
linear, logistic, and softmax regression all share the gradient "prediction
minus observation." Next, :numref:`sec_mdl-statistics` develops error bars,
hypothesis tests, confidence intervals, and the bootstrap, while
:numref:`sec_mdl-maximum_likelihood` derives our training losses from maximum
likelihood and links parameter variance to Fisher curvature. Hidden variables
motivate the evidence lower bound and expectation--maximization
(:numref:`sec_mdl-latent-em-elbo`), followed by Laplace, Monte Carlo, Markov
chain Monte Carlo, and variational approximations when posteriors lack closed
forms (:numref:`sec_mdl-bayesian-computation`). Finally,
:numref:`sec_mdl-concentration-generalization` proves finite-sample tail
bounds and high-dimensional shell concentration,
:numref:`sec_mdl-generalization-bounds` extends them to post-selection uniform
convergence (:numref:`chap_classification_generalization`), and
:numref:`sec_mdl-naive_bayes` puts distributions, smoothing, and calibration to
work on image classification.

Readers familiar with :numref:`sec_prob` can skim the opening moment algebra,
and :numref:`sec_mdl-bayesian-computation` and :numref:`sec_mdl-naive_bayes`
can be skipped on a first pass without breaking later dependencies. Full
entropy and divergence theory continues in
:numref:`chap_mdl-information-theory`, while measure theory, time series, and
nonparametric extensions are surveyed in
:numref:`sec_mdl-probability-statistics-further-reading`.

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

