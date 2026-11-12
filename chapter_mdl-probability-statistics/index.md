# Probability and Statistical Learning
:label:`chap_mdl-probability-statistics`

The main text uses probability constantly and seldom stops to argue about
it. It adopts squared error and cross-entropy as losses
(:numref:`sec_linear_regression`, :numref:`sec_softmax`), reports an accuracy
measured on a few thousand test examples, and keeps the best of many
configurations by their validation scores
(:numref:`sec_generalization_basics`). Each step is an inference from a
finite sample. This chapter supplies the arguments.

One example runs through it: a coin that lands heads 9 times in 13 flips.
We would estimate its bias as $9/13\approx0.69$, yet a fair coin does at
least this well more than one time in eight. A test accuracy is the same
coin, flipped once per test question, and the same doubt applies to it.

Resolving such doubts takes four objects: a distribution, which describes
how outcomes vary; an estimator, which turns a sample into a number; a
posterior, which describes what we believe about the parameters after seeing
the data; and a generalization bound, which says when performance on a
sample controls performance on new data. The middle two are easily confused:
the scatter of an estimate over the samples we did not draw is not the
spread of a posterior given the sample we have, although the two often
produce similar error bars.

The chapter takes them in order. Distributions need a language first:
densities for one quantity (:numref:`sec_mdl-random_variables`), then for
several at once, together with the rule for what a transformation does to a
density (:numref:`sec_mdl-joint-distributions`). With it we meet the
distributions that have names (:numref:`sec_mdl-distributions`) and find that
nearly all of them share one form, the exponential family
(:numref:`sec_mdl-exponential-family`). That common form is why linear,
logistic and softmax regression all train with the gradient "prediction minus
observation".

Estimators occupy the middle of the chapter. :numref:`sec_mdl-statistics`
supplies error bars, and :numref:`sec_mdl-maximum_likelihood` derives the
losses of the main text from one principle. The Fisher information connects
the two: it is the expected curvature of the negative log-likelihood, and its
inverse is the variance of the fitted parameters. Posteriors enter through
models with hidden variables, whose likelihood can only be bounded from below
(:numref:`sec_mdl-latent-em-elbo`), and :numref:`sec_mdl-bayesian-computation`
computes with posteriors that have no closed form. Guarantees come last:
Hoeffding's inequality (:numref:`sec_mdl-concentration-generalization`),
valid at every sample size, and what remains of it once the model is chosen
after seeing the data (:numref:`sec_mdl-generalization-bounds`), the
mathematics behind :numref:`chap_classification_generalization`. One
classifier, fitted by counting pixels, then puts the first three objects to
work on real images (:numref:`sec_mdl-naive_bayes`).

The opening sections repeat parts of :numref:`sec_prob`, now with proofs and
for densities, and a reader who knows them can skim. Bayesian computation and
naive Bayes can be skipped at first; no other section needs them. Entropy and
the KL divergence appear only as tools; their theory is in
:numref:`chap_mdl-information-theory`. Measure theory, time series and
nonparametric Bayesian models are left out.

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
