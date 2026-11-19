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

## Resources and Further Reading {.unnumbered}

The references below accompany the topics taught in this chapter. Related
subjects that we leave out are surveyed in
:numref:`sec_mdl-probability-statistics-further-reading`.

**Books**

- [Introduction to Probability --- Blitzstein & Hwang](https://probabilitybook.net/): an inviting introduction to random variables, expectation, conditioning, and the named distributions of :numref:`sec_mdl-random_variables` and :numref:`sec_mdl-distributions`; a free PDF accompanies Harvard's Stat 110.
- [Mathematics for Machine Learning --- Deisenroth, Faisal & Ong](https://mml-book.github.io/): Chapter 6 offers a self-contained review of probability and distributions, including the exponential family of :numref:`sec_mdl-exponential-family`.
- [All of Statistics --- Wasserman](https://www.stat.cmu.edu/~larry/all-of-statistics/): a crisp, computer-scientist-friendly treatment of estimation, testing, confidence intervals, and the bootstrap, covering :numref:`sec_mdl-statistics` and :numref:`sec_mdl-maximum_likelihood`; the companion page includes errata, code, and datasets.
- [Computer Age Statistical Inference --- Efron & Hastie](https://hastie.su.domains/CASI/): explores the bootstrap, large-scale testing, and the relationship between frequentist and Bayesian inference, complementing :numref:`sec_mdl-statistics`.
- [Pattern Recognition and Machine Learning --- Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/): covers the exponential family, mixture models and EM (:numref:`sec_mdl-latent-em-elbo`), and the Laplace and variational approximations of :numref:`sec_mdl-bayesian-computation`; Microsoft Research provides a free PDF.
- [Probabilistic Machine Learning: An Introduction --- Murphy](https://probml.github.io/pml-book/book1.html): develops machine learning through probability and decision theory, including MAP estimation (:numref:`sec_mdl-maximum_likelihood`) and generative classifiers (:numref:`sec_mdl-naive_bayes`), with a free draft and runnable notebooks.
- [Information Theory, Inference, and Learning Algorithms --- MacKay](https://www.inference.org.uk/itila/book.html): weaves inference together with information theory; its chapters on Laplace's method, Monte Carlo methods, and variational inference accompany :numref:`sec_mdl-bayesian-computation`.
- [Bayesian Data Analysis --- Gelman et al.](https://sites.stat.columbia.edu/gelman/book/): covers priors, posteriors, Markov chain Monte Carlo with its diagnostics, and practical Bayesian workflow for :numref:`sec_mdl-bayesian-computation`; the third edition is available as a free PDF.
- [High-Dimensional Probability --- Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html): develops concentration inequalities, sub-Gaussian variables, and high-dimensional geometry, extending :numref:`sec_mdl-concentration-generalization`.
- [Understanding Machine Learning: From Theory to Algorithms --- Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/): derives uniform convergence, Rademacher complexity, and margin bounds as in :numref:`sec_mdl-generalization-bounds`; the authors provide a free PDF.

**Courses and video lectures**

- [Statistics 110: Probability --- Harvard (Blitzstein)](https://stat110.hsites.harvard.edu/): provides lecture videos, problem sets, and more than 250 practice problems with solutions for :numref:`sec_mdl-random_variables` and :numref:`sec_mdl-distributions`.
- [Stat 110 lecture videos --- YouTube](https://stat110.hsites.harvard.edu/youtube): contains the complete lecture series on probability, distributions, and conditioning.

**Tutorials and notes**

- [Probability Theory Review --- Stanford CS229 (Maleki & Do)](https://cs229.stanford.edu/section/cs229-prob.pdf): reviews random variables, expectation, variance, Gaussian distributions, and multivariate probability for machine learning, serving as a concise companion to :numref:`sec_mdl-random_variables`.
