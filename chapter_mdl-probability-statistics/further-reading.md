# Further Reading
:label:`sec_mdl-probability-statistics-further-reading`

The chapter introduction (:numref:`chap_mdl-probability-statistics`) collects
books and courses that accompany what this chapter teaches. Here we turn to
what it leaves out. Following the order of the chapter, each short entry begins
from a limitation of a method we covered and explains how the literature
moves past it. Two of these ideas, conformal prediction and confidence
sequences, are immediately useful in practice, repairing the intervals and
tests of :numref:`sec_mdl-statistics` under the messy conditions of real
experiments. Others, such as the VC dimension, are chiefly of theoretical
interest, and some, like whether gradient noise is truly heavy-tailed, remain
actively debated. Nothing later in the book depends on this section, so treat
it as a map for exploring further.

## Probability Models

We start with three extensions to our treatment of distributions: a subtle class
of distributions that slips between discrete and continuous, the limiting laws
for averages whose variance is infinite, and a classic quirk of the softmax
when options look alike.

### Distributions Without Atoms or Densities

In :numref:`sec_mdl-random_variables`, we noted that a random variable with a
density has no point masses (atoms), and we treated every distribution as
discrete, continuous with a density, or a mixture of the two. Surprisingly,
the converse does not hold: having no atoms is not enough to guarantee a
density. To see why, build a random number in $[0, 1]$ in base $3$ using only
the digits $0$ and $2$, choosing each digit independently with a fair coin.
Its cumulative distribution function is continuous everywhere, so no single
point carries positive probability, yet all of its mass sits on the Cantor
set, whose total length is zero. No density function can integrate to $1$ over
a set of length zero.

The *Lebesgue decomposition* completes this classification: every probability
distribution on the real line is a mixture of a discrete part, a part with a
density, and a *singular continuous* part like the Cantor distribution. Having
a density corresponds precisely to being *absolutely continuous* with respect
to length, in which case the Radon--Nikodym theorem supplies the density.
:citet:`Billingsley.1995` develops probability on this measure-theoretic
footing, where expectation becomes a single integral against the distribution
and the sums and integrals of our chapter appear simply as two special cases.

### Heavy Tails and Stable Laws

As exercise :exref:`ex_mdl-random-variables-heavy-tail-averages` shows, the
average of independent Cauchy draws has the exact same Cauchy distribution,
and averages of variables with power-law tails settle down far more slowly
than the familiar $1/\sqrt{n}$ rate. If they do not approach a Gaussian, what
do they converge to? When the tail $P(|X| > x)$ decays like $x^{-\alpha}$ with
$0 < \alpha < 2$, the variance is infinite, and properly centered sums scaled
by $n^{1/\alpha}$ rather than $\sqrt{n}$ converge to an *$\alpha$-stable* law.
The Cauchy distribution is the symmetric stable law with $\alpha = 1$, while
the Gaussian sits at the boundary $\alpha = 2$. :citet:`Gnedenko.Kolmogorov.1954`
develop the classical theory of these limit laws.

In deep learning, heavy tails have attracted attention both in minibatch
gradient noise, which :citet:`Simsekli.Sagun.Gurbuzbalaban.2019` model as
$\alpha$-stable, and in the eigenvalue spectra of trained weight matrices
:cite:`Martin.Mahoney.2021`. Keep a grain of salt handy, though: every empirical
claim of this kind relies on estimating the tail index $\alpha$, and those
estimates can shift dramatically depending on how far out in the tail one
looks.

### Choice Models and Irrelevant Alternatives

In :numref:`subsec_mdl-softmax-origins`, we derived the softmax by adding
independent Gumbel noise to each option's score. A consequence of that
independence is that the odds ratio between any two choices,
$p_i / p_j = e^{z_i - z_j}$, ignores every other option on the menu. That
property breaks down as soon as two options are near-duplicates. Imagine a
commuter who splits evenly between driving a car and taking a red bus. If the
city adds an identical blue bus, the commuter should still drive half the time
and split the remaining half between the two buses. Under a softmax, however,
all three options receive probability $\tfrac{1}{3}$.

Allowing the noise of similar options to be correlated (as in a nested logit
model, or a probit model with correlated Gaussian noise) keeps the car's
probability near $\tfrac{1}{2}$, at the cost of losing closed-form choice
probabilities. :citet:`McFadden.1974` derived the logit from Gumbel noise and
highlighted this restriction, known as Luce's *independence of irrelevant
alternatives*, as its central modeling limitation.

## Estimation and Testing

The confidence intervals and hypothesis tests of :numref:`sec_mdl-statistics`
leaned on three conveniences: a Gaussian approximation, a known model for the
errors, or a sample size fixed before seeing the data. The three topics below
each show what happens when we give up one of those crutches.

### Better Bootstrap Intervals

The simple percentile bootstrap of :numref:`sec_mdl-statistics` inherits any
bias or skewness in the resampled estimates, so on small samples or skewed
statistics its actual coverage can fall noticeably short of its nominal level.
Two refinements fix most of the gap. The *studentized* bootstrap resamples the
self-normalized pivot $(\hat\theta^* - \hat\theta)/\widehat{\textrm{se}}^*$
rather than $\hat\theta^*$ itself, while the *BCa* (bias-corrected and
accelerated) interval shifts the percentile cutoffs to account for both bias
and the way the standard error varies with $\theta$. For one-sided intervals,
both refinements shrink the coverage error from order $1/\sqrt{n}$ down to
order $1/n$ :cite:`Hall.1992`. You rarely need them when reporting an accuracy
on thousands of test examples, but they pay off immediately on small
evaluation sets or ratios of error rates. :citet:`DiCiccio.Efron.1996` give a
clear review of these methods.

### Conformal Prediction

A trained neural network hands us a point prediction, but it gives no
prediction interval with guaranteed coverage unless we are willing to assume a
parametric error model. *Split conformal prediction* sidesteps modeling
assumptions altogether. Using $n$ held-out calibration points, we score how
badly the model misses each one, for instance by the residual
$|y - f(\mathbf{x})|$. For a new input, we then return every label whose score
is at most the $\lceil(n+1)(1-\alpha)\rceil$-th smallest calibration score.

As long as the calibration and test points are exchangeable, the resulting
prediction set contains the true label with probability at least $1 - \alpha$,
for *any* model and *any* data distribution. The proof uses the same symmetry
behind the permutation test: under exchangeability, the rank of the test score
among all $n + 1$ scores is uniform. One caveat matters in practice: the
guarantee holds on average over inputs, not conditionally for each individual
input. :citet:`Vovk.Gammerman.Shafer.2022` develop the full theory, and
:citet:`Angelopoulos.Bates.2023` offer an accessible tutorial with code for
both classification and regression.

### Confidence Sequences

The $z$-test of :numref:`sec_mdl-statistics` assumes that you chose the sample
size $n$ before looking at a single outcome. If instead you peek at a running
experiment after every batch and stop the moment $p < 0.05$, a false alarm
becomes virtually certain if you watch long enough. Exercise
:exref:`ex_mdl-statistics-peeking` builds a Gaussian mixture martingale test
that stays valid no matter how often you check. *Confidence sequences* push
that idea to full intervals that hold simultaneously at every sample size $n$,
for any bounded or sub-Gaussian data rather than a single parametric model.
You pay for this freedom with wider intervals, though for the sharpest boundaries
the extra width grows remarkably slowly, only like $\sqrt{\log\log n}$.
:citet:`Howard.Ramdas.McAuliffe.ea.2021` construct these sequences from
nonnegative supermartingales, and :citet:`Johari.Koomen.Pekelis.ea.2017`
describe how always-valid $p$-values were deployed in a commercial A/B testing
platform.

## Computing with Posteriors

In :numref:`sec_mdl-bayesian-computation`, we worked with a single Markov
chain, the random-walk Metropolis algorithm, on a two-parameter posterior. Real
Bayesian models have thousands or millions of parameters, which calls for
sharper samplers and scalable approximations.

### Gibbs Sampling and Hamiltonian Monte Carlo

Random-walk Metropolis (:numref:`sec_mdl-bayes-mcmc`) requires tuning a
proposal scale in every direction and inches forward by small diffusive steps.
When a posterior is high-dimensional or tightly correlated, the chain wastes
most of its time wandering back and forth. *Gibbs sampling* avoids proposal
tuning by updating one coordinate at a time from its exact conditional
distribution, so every move is accepted; the catch is that those conditionals
must be easy to sample, and strong correlations between coordinates still slow
it down :cite:`Geman.Geman.1984`.

*Hamiltonian Monte Carlo* takes a more physical approach: it treats
$-\log\tilde{p}$ as a potential energy, injects a random momentum, and uses
gradients to simulate the resulting trajectory for many steps. Because the
dynamics nearly conserve energy, a single proposal can glide far across the
posterior and still be accepted with high probability. :citet:`Neal.2011`
gives a lucid introduction to the method and its tuning, and
:citet:`Hoffman.Gelman.2014` automate the trajectory length with the No-U-Turn
Sampler, which became the default engine behind Stan.

### Posteriors of Neural Networks

Every algorithm in :numref:`sec_mdl-bayesian-computation` evaluates the full
log-joint across the entire dataset at every step, and the Laplace
approximation requires storing and inverting a $d \times d$ Hessian. Neither
is feasible for a neural network with millions of parameters and millions of
training examples. *Stochastic-gradient Langevin dynamics* adapts the
unadjusted Langevin update of exercise
:exref:`ex_mdl-bayesian-computation-mala` by plugging in minibatch gradients
and a shrinking step size, trading away the Metropolis acceptance check for a
per-step cost identical to ordinary SGD :cite:`Welling.Teh.2011`. Laplace
approximations scale up in a different way, by keeping only the most
informative slice of the curvature: a Kronecker-factored block per layer, or
the Hessian of the last layer alone. :citet:`Daxberger.Kristiadi.Immer.ea.2021`
show that these lightweight approximations often match far costlier Bayesian
methods at a tiny fraction of the compute.

## Concentration and Learning Theory

The first two topics below generalize the tools of
:numref:`sec_mdl-concentration-generalization` and
:numref:`sec_mdl-generalization-bounds` beyond simple averages and Rademacher
complexity. The last two take up the puzzle on which our theory sections
closed: how to prove guarantees for the specific solution an algorithm
actually finds, rather than for the worst function lurking in its hypothesis
class.

### Bounded Differences and Lipschitz Functions

The Chernoff method in :numref:`sec_mdl-concentration-generalization` relies
on having a sum of independent terms. Many quantities we care about, such as
the largest gap between true risk and empirical risk across a whole model
class, depend on independent inputs without being simple sums. *McDiarmid's
inequality* generalizes Hoeffding's bound to such functions: replace "each
term lies in an interval of length $c_i$" with "changing input $i$ alone moves
the function by at most $c_i$," and the very same tail bound
$e^{-2t^2/\sum_i c_i^2}$ still holds, proved by replacing the product of
moment generating functions with a martingale.

Gaussian vectors enjoy an even cleaner property: for a standard Gaussian
vector in any dimension $d$, every $L$-Lipschitz function $f$ obeys
$P(|f - \mathbb{E}f| \ge t) \le 2e^{-t^2/(2L^2)}$. Because the Euclidean norm
is $1$-Lipschitz, its fluctuations around the thin shell stay order $1$ no
matter how large $d$ grows. :citet:`Boucheron.Lugosi.Massart.2013` give a
comprehensive treatment of both results and the machinery behind them.

### VC Dimension and Covering Numbers

In :numref:`sec_mdl-generalization-bounds`, we controlled infinite hypothesis
classes through their Rademacher complexity. The classical alternative counts
how many distinct labelings a class can produce on a finite set of points.
Suppose a binary classifier family cannot realize all $2^{d+1}$ labelings of
any $d+1$ points (for example, linear halfspaces in $\mathbb{R}^m$ have
$d = m + 1$). Sauer's lemma then guarantees that the class produces at most
$(en/d)^d$ labelings on any $n \ge d$ points, and a union bound over those
labelings yields a generalization gap of order $\sqrt{d \log n / n}$, the
celebrated argument of :citet:`Vapnik.Chervonenkis.1971`. For real-valued
function classes, one instead covers the class at each resolution $\epsilon$
by finitely many representatives; a technique called *chaining* sums across
scales and shaves off the extra $\log n$ factor :cite:`Vershynin.2018`.

### PAC-Bayes Bounds

Every bound in :numref:`sec_mdl-generalization-bounds` guards against the
worst function in a hypothesis class fixed before seeing the data. A
*PAC-Bayes* bound shifts perspective: it evaluates a posterior distribution
$Q$ over functions chosen *after* observing the data, measured against a prior
$P$ fixed beforehand. The class complexity term is then replaced by
$D_{\textrm{KL}}(Q\|P)/n$ under a square root :cite:`McAllester.1999`. Even
if the full hypothesis class is enormous, a posterior that stays close to the
prior pays only a small complexity penalty. Building on this idea,
:citet:`Dziugaite.Roy.2017` optimized a Gaussian posterior around the weights
found by SGD and produced nonvacuous generalization bounds on a binary MNIST
task for stochastic networks with far more parameters than training examples.

### Algorithmic Stability

The random-label experiment (:numref:`subsec_random_labels`) makes clear that
the class of functions a neural network *can* represent is vastly larger than
the subset SGD actually visits on real data. *Algorithmic stability* bounds
the learning algorithm rather than the hypothesis class. If swapping out any
single training example shifts the loss of the learned model on any test point
by at most $\beta$, then the expected generalization gap is also of order
$\beta$, and McDiarmid's inequality lifts that expectation into a
high-probability bound :cite:`Bousquet.Elisseeff.2002`. For smooth, Lipschitz
convex losses, SGD is stable with $\beta$ proportional to the sum of its step
sizes divided by $n$, giving tighter guarantees when training takes fewer or
smaller steps :cite:`Hardt.Recht.Singer.2016`.

## Classifiers from Counts

We close with two continuations of :numref:`sec_mdl-naive_bayes`: repairing
miscalibrated probabilities when a single temperature is not enough, and using
a generative classifier to learn from unlabeled data, something a purely
discriminative fit cannot do directly.

### Calibration Beyond One Temperature

Temperature scaling (exercise
:exref:`ex_mdl-naive-bayes-temperature-scaling`) works when a classifier's
scores are accurate up to a single multiplicative scale. When the distortion
is more complex, richer maps help. *Platt scaling* fits both a slope and an
offset inside a logistic curve mapping binary scores to probabilities
:cite:`Platt.1999`, while isotonic regression fits any nondecreasing function;
:citet:`Zadrozny.Elkan.2002` applied isotonic regression to calibrate naive
Bayes and other classifiers. The trade-off is familiar: a more flexible
calibrator requires more held-out data to avoid overfitting, and for modern
deep networks a single temperature often works just as well
:cite:`Guo.Pleiss.Sun.Weinberger.2017`.

### Semi-Supervised Naive Bayes

In many real problems, labeled examples are expensive while unlabeled inputs
are plentiful. Because naive Bayes models how each class generates its
features, an unlabeled example fits naturally into the likelihood by treating
its missing class as a latent variable. The EM algorithm of
:numref:`sec_mdl-latent-em-elbo` then alternates between computing class
posteriors for the unlabeled examples and updating the word or pixel counts
using those posteriors as fractional labels.
:citet:`Nigam.McCallum.Thrun.ea.2000` demonstrated substantial accuracy gains
in text classification when only a handful of documents were labeled, while
also warning that unlabeled data can hurt when the generative model fits the
data poorly.

[Discussions](https://d2l.discourse.group/t/36864)
