# Probability and Statistical Learning
:label:`chap_mdl-probability-statistics`

This chapter separates four questions. A probability distribution describes
how outcomes vary; an estimator maps observed data to a parameter or other
quantity; a posterior approximation computes with uncertainty under a chosen
probabilistic model; and a generalization result states conditions under which
sample performance controls population performance. Confusing these objects
leads, for example, to treating a density as a point probability or a confidence
interval as a posterior probability.

The foundational path is random variables, distributions, maximum likelihood,
and statistics. It develops densities and transformations, common probability
laws, point estimation, sampling distributions, tests, and confidence
intervals. Bayesian computation and concentration/generalization form an
advanced path: the former approximates posterior integrals, while the latter
derives tail and uniform-convergence bounds under explicit assumptions. The
final section uses naive Bayes to combine a likelihood, a conditional-
independence factorization, parameter estimation, prediction, and uncertainty
analysis in one classifier.

```toc
:maxdepth: 2

mdl-random-variables
mdl-distributions
mdl-maximum-likelihood
mdl-bayesian-computation
mdl-statistics
mdl-concentration-generalization
mdl-naive-bayes
```

## Resources and Further Reading {.unnumbered}

The following references cover random variables and distributions,
maximum-likelihood and MAP estimation, Bayesian inference, estimators, and
hypothesis testing.

**Books**

- [Introduction to Probability --- Blitzstein & Hwang](https://probabilitybook.net/): introduces random variables, expectation, and conditioning; a free PDF accompanies Harvard's Stat 110.
- [All of Statistics --- Wasserman](https://www.stat.cmu.edu/~larry/all-of-statistics/): gives a compact treatment of probability and statistical inference for computer scientists; the companion page includes errata, code, and datasets.
- [Pattern Recognition and Machine Learning --- Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/): covers probabilistic modeling and Bayesian methods; Microsoft Research provides a free PDF.
- [Probabilistic Machine Learning: An Introduction --- Murphy](https://probml.github.io/pml-book/book1.html): develops machine learning through probability and decision theory, with a free draft and runnable notebooks.
- [Mathematics for Machine Learning --- Deisenroth, Faisal & Ong](https://mml-book.github.io/): Chapter 6 provides a self-contained review of probability and distributions.
- [Information Theory, Inference, and Learning Algorithms --- MacKay](https://www.inference.org.uk/itila/book.html): connects probability and inference with information theory and coding.
- [Bayesian Data Analysis --- Gelman et al.](https://sites.stat.columbia.edu/gelman/book/): treats priors, posteriors, computation, and practical Bayesian workflow; the third edition is available as a free PDF.
- [High-Dimensional Probability --- Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html): develops concentration inequalities, sub-Gaussian variables, and high-dimensional geometry.
- [Computer Age Statistical Inference --- Efron & Hastie](https://hastie.su.domains/CASI/): discusses bootstrap methods, large-scale testing, and connections between frequentist and Bayesian inference.

**Courses and video lectures**

- [Statistics 110: Probability --- Harvard (Blitzstein)](https://stat110.hsites.harvard.edu/): provides lecture videos, problem sets, and more than 250 practice problems with solutions.
- [Stat 110 lecture videos --- YouTube](https://stat110.hsites.harvard.edu/youtube): contains the complete lecture series on probability, distributions, and conditioning.

**Tutorials and notes**

- [Probability Theory Review --- Stanford CS229 (Maleki & Do)](https://cs229.stanford.edu/section/cs229-prob.pdf): reviews random variables, expectation, variance, Gaussian distributions, and multivariate probability for machine learning.
