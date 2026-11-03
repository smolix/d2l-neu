# Probability and Statistical Learning
:label:`chap_mdl-probability-statistics`

Every model in this book was judged by a number computed on data, such as
a test accuracy or a validation loss. We read such a number as a fact about the model. It is also a fact about the particular examples
that happened to be drawn. Take the simplest case, a coin that lands heads
9 times in 13 flips. The obvious estimate of its bias is $9/13\approx0.69$, but
a fair coin does this often enough that 13 flips cannot rule it out. A
test accuracy is the same coin, flipped once per test question, and the same
questions apply: how far can the true accuracy be from the measured
one, and is a second model that scores a little higher really better?

Probability and statistics were built for questions of this kind, from the
games of chance that Pascal and Fermat analyzed to the field trials that
Fisher designed. Answering them needs four objects that are easy to confuse:
a distribution, which describes how outcomes vary; an estimator, which turns
a sample into a number; a posterior, which describes what we believe about
the parameters after seeing data; and a generalization bound, which says
when performance on a sample controls performance on new data. The same
objects explain tools of the main text. Cross-entropy and squared error are
negative log-likelihoods, the curvature of a loss sets the error bar of the
fitted weights, and a validation score loses part of its meaning when we
keep the best of many configurations.

:numref:`sec_mdl-random_variables` asks how to assign probabilities to
quantities that vary continuously. :numref:`sec_mdl-distributions` asks
which few distributions model most labels, and
:numref:`sec_mdl-exponential-family` why almost all of them share one form
that makes their losses convex. :numref:`sec_mdl-statistics` asks how much
an estimate varies from sample to sample and how to decide whether a
difference is real. :numref:`sec_mdl-maximum_likelihood` asks why
maximizing likelihood produces the losses of the main text and how precisely
it pins down the parameters. :numref:`sec_mdl-bayesian-computation` asks how
to average over a posterior that has no closed form.
:numref:`sec_mdl-concentration-generalization` asks why an average of many
observations is exponentially unlikely to stray far from its mean, and
:numref:`sec_mdl-generalization-bounds` what remains of that guarantee when
a model is chosen after seeing the data. :numref:`sec_mdl-naive_bayes` puts
these tools to work on one classifier.

We assume the probability of :numref:`sec_prob` and calculus. Measure theory, time series and nonparametric
Bayesian models are left out.

```toc
:maxdepth: 2

mdl-random-variables
mdl-distributions
mdl-exponential-family
mdl-statistics
mdl-maximum-likelihood
mdl-bayesian-computation
mdl-concentration-generalization
mdl-generalization-bounds
mdl-naive-bayes
further-reading
```

## Resources and Further Reading {.unnumbered}

The following references accompany the material this chapter teaches;
topics it does not teach are collected in
:numref:`sec_mdl-probability-statistics-further-reading`.

**Books**

- [Introduction to Probability --- Blitzstein & Hwang](https://probabilitybook.net/): introduces random variables, expectation, conditioning and the named distributions of :numref:`sec_mdl-random_variables` and :numref:`sec_mdl-distributions`; a free PDF accompanies Harvard's Stat 110.
- [Mathematics for Machine Learning --- Deisenroth, Faisal & Ong](https://mml-book.github.io/): Chapter 6 gives a self-contained review of probability and distributions, including the exponential family of :numref:`sec_mdl-exponential-family`.
- [All of Statistics --- Wasserman](https://www.stat.cmu.edu/~larry/all-of-statistics/): a compact treatment of estimation, testing, confidence intervals and the bootstrap for computer scientists, covering :numref:`sec_mdl-statistics` and :numref:`sec_mdl-maximum_likelihood`; the companion page includes errata, code, and datasets.
- [Computer Age Statistical Inference --- Efron & Hastie](https://hastie.su.domains/CASI/): discusses the bootstrap, large-scale testing, and the relation between frequentist and Bayesian inference, for :numref:`sec_mdl-statistics`.
- [Pattern Recognition and Machine Learning --- Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/): covers the exponential family, mixture models and EM (:numref:`sec_mdl-latent-em-elbo`), and the Laplace and variational approximations of :numref:`sec_mdl-bayesian-computation`; Microsoft Research provides a free PDF.
- [Probabilistic Machine Learning: An Introduction --- Murphy](https://probml.github.io/pml-book/book1.html): develops machine learning through probability and decision theory, including MAP estimation (:numref:`sec_mdl-maximum_likelihood`) and generative classifiers (:numref:`sec_mdl-naive_bayes`), with a free draft and runnable notebooks.
- [Information Theory, Inference, and Learning Algorithms --- MacKay](https://www.inference.org.uk/itila/book.html): connects inference with information theory, and its chapters on Laplace's method, Monte Carlo methods and variational methods accompany :numref:`sec_mdl-bayesian-computation`.
- [Bayesian Data Analysis --- Gelman et al.](https://sites.stat.columbia.edu/gelman/book/): treats priors, posteriors, Markov chain Monte Carlo and its diagnostics, and practical Bayesian workflow, for :numref:`sec_mdl-bayesian-computation`; the third edition is available as a free PDF.
- [High-Dimensional Probability --- Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html): develops concentration inequalities, sub-Gaussian variables and high-dimensional geometry, the material of :numref:`sec_mdl-concentration-generalization`.
- [Understanding Machine Learning: From Theory to Algorithms --- Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/): derives uniform convergence, Rademacher complexity and margin bounds, as in :numref:`sec_mdl-generalization-bounds`; the authors provide a free PDF.

**Courses and video lectures**

- [Statistics 110: Probability --- Harvard (Blitzstein)](https://stat110.hsites.harvard.edu/): provides lecture videos, problem sets, and more than 250 practice problems with solutions for :numref:`sec_mdl-random_variables` and :numref:`sec_mdl-distributions`.
- [Stat 110 lecture videos --- YouTube](https://stat110.hsites.harvard.edu/youtube): contains the complete lecture series on probability, distributions, and conditioning.

**Tutorials and notes**

- [Probability Theory Review --- Stanford CS229 (Maleki & Do)](https://cs229.stanford.edu/section/cs229-prob.pdf): reviews random variables, expectation, variance, Gaussian distributions, and multivariate probability for machine learning, a short companion to :numref:`sec_mdl-random_variables`.
