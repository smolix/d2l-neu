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
derives tail and uniform-convergence bounds under explicit assumptions. Naive
Bayes is the capstone. It combines a likelihood, an independence factorization,
parameter estimation, prediction, and an uncertainty audit in one classifier.

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

- [Introduction to Probability --- Blitzstein & Hwang](https://probabilitybook.net/): the gentlest rigorous route into random variables, expectation, and conditioning; the free PDF accompanies Harvard's Stat 110.
- [All of Statistics --- Wasserman](https://www.stat.cmu.edu/~larry/all-of-statistics/): a compact graduate-level tour of probability *and* inference (estimation, testing, bootstrap) written for computer scientists; the companion page hosts errata, code, and datasets.
- [Pattern Recognition and Machine Learning --- Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/): the classic treatment of probabilistic modeling and Bayesian methods, now released by Microsoft Research as a free PDF.
- [Probabilistic Machine Learning: An Introduction --- Murphy](https://probml.github.io/pml-book/book1.html): a modern, deep-learning-aware reframing of ML through probability and decision theory; free draft PDF with runnable notebooks.
- [Mathematics for Machine Learning --- Deisenroth, Faisal & Ong](https://mml-book.github.io/): Chapter 6, "Probability and Distribution," is a clean self-contained refresher pitched exactly at ML readers; full PDF is free.
- [Information Theory, Inference, and Learning Algorithms --- MacKay](https://www.inference.org.uk/itila/book.html): connects probability and inference to information theory and coding; the full text is free to read online.
- [Bayesian Data Analysis --- Gelman et al.](https://sites.stat.columbia.edu/gelman/book/): the standard reference for priors, posteriors, and practical Bayesian workflow once MAP and naive Bayes whet the appetite; third edition free as PDF.
- [High-Dimensional Probability --- Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html): where concentration inequalities, sub-Gaussian variables, and the strange geometry of high dimensions (norm concentration, near-orthogonality) get their systematic treatment; the free PDF is the standard modern reference.
- [Computer Age Statistical Inference --- Efron & Hastie](https://hastie.su.domains/CASI/): the bootstrap, large-scale testing, and the frequentist-Bayesian interplay told by two of the field's architects---the natural sequel to this chapter's statistics; free PDF from the authors.

**Courses and video lectures**

- [Statistics 110: Probability --- Harvard (Blitzstein)](https://stat110.hsites.harvard.edu/): a full course with lecture videos, problem sets, and 250+ practice problems with solutions, mirroring the Blitzstein & Hwang text.
- [Stat 110 lecture videos --- YouTube](https://stat110.hsites.harvard.edu/youtube): the complete lecture series, ideal for building intuition about distributions and conditioning alongside the reading.

**Tutorials and notes**

- [Probability Theory Review --- Stanford CS229 (Maleki & Do)](https://cs229.stanford.edu/section/cs229-prob.pdf): a terse, well-organized refresher of exactly the probability used in ML---random variables, expectation/variance, Gaussians, and multivariate distributions.
