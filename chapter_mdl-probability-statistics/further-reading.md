# Further Reading
:label:`sec_mdl-probability-statistics-further-reading`

The sections of this chapter describe how data vary, turn data into
estimates with error bars, average over a posterior, and ask when a sample
can be trusted for a model chosen after seeing it. This section collects, in
the order of the chapter, topics they had no room for: probability without
densities, heavy-tailed limits, choice models, sharper intervals and tests,
samplers that scale, learning theory beyond Rademacher complexity, and two
refinements of counting classifiers. Each topic states the problem it
addresses and what it changes relative to the chapter. Resources for the
material the chapter does teach are listed in its introduction
(:numref:`chap_mdl-probability-statistics`). Nothing else in the book
depends on this section.

## Probability Models

### Distributions Without Atoms or Densities

:numref:`sec_mdl-random_variables` notes that a variable with a density has
no atoms, and treats every variable as discrete, as having a density, or as a
mixture of the two. The converse fails. Write a number in base 3 using only
the digits 0 and 2, each chosen by a fair coin. Its c.d.f. is continuous, so
it has no atoms, yet all of its probability lies on the Cantor set, which has
length zero, so no density describes it. The Lebesgue decomposition says
that every distribution on the line is a mixture of a discrete part, a part
with a density, and such a singular part. "Has a density" means *absolutely
continuous* with respect to length, and the Radon–Nikodym theorem then
supplies the density. :citet:`Billingsley.1995` builds probability on this
footing, in which the chapter's results hold without the regularity
assumptions it states along the way.

### Heavy Tails and Stable Laws

The average of Cauchy draws is again Cauchy, and averages of draws with a
power-law tail settle more slowly than $1/\sqrt n$
(exercise :exref:`ex_mdl-random-variables-heavy-tail-averages`). What do they
converge to? If $P(|X|>x)$ falls like $x^{-\alpha}$ with $0<\alpha<2$, the
variance is infinite, and suitably centered sums, scaled by $n^{1/\alpha}$
instead of $\sqrt n$, converge to an *$\alpha$-stable* law rather than to a
Gaussian. The Cauchy is the stable law with $\alpha=1$ and the Gaussian the
one with $\alpha=2$. :citet:`Gnedenko.Kolmogorov.1954` give the classical
theory of these limits. In deep learning, heavy tails have been reported in
minibatch gradient noise, modeled as $\alpha$-stable by
:citet:`Simsekli.Sagun.Gurbuzbalaban.2019`, and in the eigenvalue spectra of
trained weight matrices :cite:`Martin.Mahoney.2021`. Each such claim rests on
an estimate of the tail index, which depends strongly on how much of the
tail it uses.

### Choice Models and Irrelevant Alternatives

:numref:`subsec_mdl-softmax-origins` derives the softmax from Gumbel noise
added to the scores of the options. The ratio $p_i/p_j=e^{z_i-z_j}$ then
does not depend on any other option, which fails for near-duplicates. A
commuter who splits evenly between a car and a red bus should still take
the car half the time when an identical blue bus is added, but the softmax
gives each of the three options $\tfrac13$. Letting the noise of similar
options be correlated (nested logit, or a probit with correlated Gaussian
noise) keeps the car near $\tfrac12$, at the price of choice probabilities
that have no closed form. :citet:`McFadden.1974` derived the logit from Gumbel
noise and stated this property, the *independence of irrelevant
alternatives*, as its main restriction.

## Estimation and Testing

### Better Bootstrap Intervals

The percentile interval of :numref:`sec_mdl-statistics` inherits the bias
and skew of the replicate distribution, and for small samples or skewed
statistics it covers the parameter less often than it claims. The
*studentized* bootstrap resamples the pivot
$(\hat\theta^*-\hat\theta)/\widehat{\textrm{se}}^*$ instead of $\hat\theta^*$,
and the *BCa* interval shifts the percentiles to correct for the bias and for
how the standard error changes with $\theta$. For a one-sided interval both
reduce the coverage error from order $1/\sqrt n$ to order $1/n$
:cite:`Hall.1992`. :citet:`DiCiccio.Efron.1996` review these intervals and
when each is worth its extra computation.

### Conformal Prediction

A trained model gives a prediction, but no interval with a guaranteed
coverage unless its errors follow a known model. *Split conformal
prediction* scores $n$ held-out calibration points by how badly the model
fits them, for example by $|y-f(\mathbf x)|$, and returns for a new input
all labels whose score is at most the $\lceil(n+1)(1-\alpha)\rceil$-th
smallest calibration score. If the calibration and test points are
exchangeable, the set contains the true label with probability at least
$1-\alpha$, for any model and any data distribution. The argument is the one
behind the permutation test: under exchangeability the rank of the test
score among all $n+1$ scores is uniform. The guarantee holds on average over
inputs, not for each input separately. :citet:`Vovk.Gammerman.Shafer.2022`
develop the theory, and :citet:`Angelopoulos.Bates.2023` give a tutorial
with code for classifiers and regression models.

### Confidence Sequences

The $z$-test of :numref:`sec_mdl-statistics` is valid at a sample size fixed
in advance. Checking a running comparison after every batch and stopping at
the first $p<0.05$ makes a false alarm certain if the checks go on long
enough, and exercise :exref:`ex_mdl-statistics-peeking` builds a test that
stays valid however often it is inspected, from a Gaussian mixture
martingale. *Confidence sequences* extend this to intervals that hold
simultaneously at every sample size, for any bounded or sub-Gaussian data
rather than one known model, so the experimenter may stop at any time. They
are wider than a fixed-$n$ interval, but for the best boundaries only by a
factor that grows like $\sqrt{\log\log n}$.
:citet:`Howard.Ramdas.McAuliffe.ea.2021` construct them from nonnegative
supermartingales, and :citet:`Johari.Koomen.Pekelis.ea.2017` describe the
always-valid $p$-values that a commercial A/B testing platform adopted.

## Computing with Posteriors

### Gibbs Sampling and Hamiltonian Monte Carlo

Random-walk Metropolis (:numref:`sec_mdl-bayes-mcmc`) needs a proposal scale
for every direction and moves a short distance per step, so strongly
correlated or high-dimensional posteriors take many steps per independent
draw. *Gibbs sampling* updates one coordinate at a time from its exact
conditional distribution, so every proposal is accepted and no scale is
tuned; it needs conditionals that can be sampled, and it is slow when
coordinates are strongly correlated :cite:`Geman.Geman.1984`. *Hamiltonian
Monte Carlo* treats $-\log\tilde p$ as a potential energy, adds a random
momentum, and simulates the resulting motion for many steps with the
gradient. A proposal can then travel across the posterior and still be
accepted, because the simulation nearly conserves energy. :citet:`Neal.2011`
explains the method and its tuning, and :citet:`Hoffman.Gelman.2014` choose
the trajectory length automatically with the No-U-Turn Sampler, the default
sampler of Stan and several other probabilistic programming systems.

### Posteriors of Neural Networks

Every method of :numref:`sec_mdl-bayesian-computation` evaluates the full
log-joint at each step, and the Laplace approximation stores a $d\times d$
Hessian; a network has millions of examples and parameters.
*Stochastic-gradient Langevin dynamics* takes the unadjusted Langevin step
of exercise :exref:`ex_mdl-bayesian-computation-mala` with a minibatch
gradient and a decreasing step size, giving up the Metropolis correction for
a cost per step equal to that of SGD :cite:`Welling.Teh.2011`. Laplace
approximations scale by keeping only part of the curvature, a
Kronecker-factored block per layer or the Hessian of the last layer alone;
:citet:`Daxberger.Kristiadi.Immer.ea.2021` show that these give calibrated
predictive uncertainty for trained networks at little extra cost.

## Concentration and Learning Theory

### Bounded Differences and Lipschitz Functions

The Chernoff method of :numref:`sec_mdl-concentration-generalization` needs
a sum of independent terms. Many quantities, such as the largest gap between
risk and empirical risk over a class, are functions of independent inputs
but not sums. *McDiarmid's inequality* replaces "each term lies in an
interval of length $c_i$" by "changing input $i$ moves the function by at
most $c_i$" and keeps the tail $e^{-2t^2/\sum_ic_i^2}$; its proof replaces
the product of moment generating functions by a martingale. For a standard
Gaussian vector, every $L$-Lipschitz function $f$ satisfies
$P(|f-\mathbb{E}f|\ge t)\le 2e^{-t^2/(2L^2)}$ in any dimension. The norm is
the case $L=1$, which is why its fluctuation in the thin shell does not grow
with $d$. :citet:`Boucheron.Lugosi.Massart.2013` develop both results and
the methods behind them.

### VC Dimension and Covering Numbers

:numref:`sec_mdl-generalization-bounds` bounds infinite classes by their
Rademacher complexity. The older route counts labelings. If a binary class
cannot realize all $2^{d+1}$ labelings of any $d+1$ points, Sauer's lemma
says that it realizes at most $(en/d)^d$ labelings of $n\ge d$ points, and
the union bound over labelings gives a complexity of order
$\sqrt{d\log n/n}$; :citet:`Vapnik.Chervonenkis.1971` introduced this
argument. For real-valued classes one covers the class at each scale
$\epsilon$ by finitely many functions instead, and chaining sums the scales
and removes the $\log n$ :cite:`Vershynin.2018`.

### PAC-Bayes Bounds

Every bound of :numref:`sec_mdl-generalization-bounds` covers the worst
function in a class fixed before the data. A *PAC-Bayes* bound covers a
distribution $Q$ over functions chosen after the data, against a prior $P$
fixed before, and the class complexity is replaced by
$D_{\textrm{KL}}(Q\|P)/n$ under a square root :cite:`McAllester.1999`. A
posterior that stays close to the prior is cheap even if the class is huge.
:citet:`Dziugaite.Roy.2017` optimized a Gaussian posterior around the weights
found by SGD and obtained the first nonvacuous bounds for stochastic networks
with far more parameters than training examples, on a binary MNIST task.

### Algorithmic Stability

The random-label experiment shows that the class a network can represent is
far larger than what SGD reaches on real data. Stability bounds the
*algorithm* instead. If replacing one training example changes the loss of
the output at any point by at most $\beta$, the expected generalization gap
is of order $\beta$, and McDiarmid's inequality turns this into a
high-probability bound :cite:`Bousquet.Elisseeff.2002`. For smooth convex
losses, SGD is stable with $\beta$ proportional to the sum of its step sizes
divided by $n$, so the bound is smaller for fewer or smaller steps
:cite:`Hardt.Recht.Singer.2016`.

## Classifiers from Counts

### Calibration Beyond One Temperature

Temperature scaling (exercise
:exref:`ex_mdl-naive-bayes-temperature-scaling`) assumes that the scores are
right up to a scale. *Platt scaling* fits a slope and an offset of a
logistic map from a binary score to a probability :cite:`Platt.1999`, and
isotonic regression fits any monotone map; :citet:`Zadrozny.Elkan.2002` used
it to calibrate naive Bayes and other classifiers. A more flexible map needs
more held-out data, and for modern networks a single temperature is often as
good :cite:`Guo.Pleiss.Sun.Weinberger.2017`.

### Semi-Supervised Naive Bayes

Labels are scarce and unlabeled inputs are not. Because naive Bayes models
how each class produces its inputs, an unlabeled example enters the
likelihood with its class as a latent variable. The EM algorithm of
:numref:`sec_mdl-latent-em-elbo` then alternates between class posteriors
for the unlabeled examples and counting with those posteriors as fractional
labels. :citet:`Nigam.McCallum.Thrun.ea.2000` showed large gains for text
classification with few labeled documents, and that unlabeled data can hurt
when the generative model fits the data poorly.
