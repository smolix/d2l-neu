# The Exponential Family
:label:`sec_mdl-exponential-family`

Linear regression, logistic regression, and softmax regression each minimize a
different loss function. Yet when we differentiate those losses with respect to
the model weights, all three yield the exact same training rule: the input
multiplied by the difference between the prediction and the observation. When
three seemingly separate derivations converge on one formula, there is usually
a single structure underneath. The catalogue of :numref:`sec_mdl-distributions`,
where each distribution's mean and variance seemed to require its own clever
algebraic trick, points to the same suspicion.

Consider the coin that runs through this chapter, which landed heads 9 times
in 13 flips. The natural estimate of its bias is $9/13\approx0.69$. If we fit
a logistic-regression model with only a bias term to those same 13 flips, it
arrives at $9/13$ by an indirect route: its final layer computes the log-odds
$\log(9/4)\approx0.81$, and a sigmoid maps that number back to $9/13$. Why take
this detour through the log-odds instead of working directly with the
probability?

One idea explains both puzzles. Nearly every distribution in our catalogue can
be written in a single canonical form, where the parameters enter only through
the exponential of a linear combination of a few data statistics. Distributions
with this structure form the *exponential family*, and for a coin, the
parameter sitting in the exponent is precisely the log-odds. Remarkably, four
central properties follow from this form alone: the mean and variance are
simply the first two derivatives of a single function; maximum likelihood
becomes a convex optimization problem whose gradient is always prediction minus
observation; the softmax emerges automatically when we model class labels; and
a prior of the same form updates by adding counts.

For deep learning, the second property is the central result, and the first is
what proves it (:numref:`subsec_mdl-moment-matching` and
:numref:`subsec_mdl-glm`). Together, they tell us which loss function to pair
with a new kind of label, such as a count, and when a maximum-likelihood fit
has no finite solution at all, as happens when a coin lands heads on every
single flip. Two parts of this section can be skipped on a first reading: the
maximum-entropy derivation (:numref:`subsec_mdl-max-entropy-derivation`), which
explains where the exponential form comes from, and the alternative derivations
of the softmax from noisy choices and statistical physics.

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-exponential-family-the-exponential-family-1}
#@tab mxnet
from mxnet import nd
```

## One Form for Many Distributions

Let us start with the simplest distribution in our catalogue, a single coin
flip, and rewrite it until the log-odds appears on its own. That small bit of
algebra will show us what the general form has to look like.

### The Bernoulli in Exponential Form

A coin with bias $p$ lands heads ($x=1$) with probability $p$ and tails
($x=0$) with probability $1-p$. In the previous section we wrote both cases in
one expression, $p^x(1-p)^{1-x}$. To see how the outcome $x$ and the parameter
$p$ interact, take the logarithm and group the terms that contain $x$:

$$
\log P(X=x) = x\log p + (1-x)\log(1-p) = x\log\frac{p}{1-p} + \log(1-p).
$$

Notice how cleanly the two roles separate: the outcome enters only through $x$
multiplied by the log-odds $\log\frac{p}{1-p}$, while the second term depends
on $p$ alone. Let us give the log-odds a name, $\eta=\log\frac{p}{1-p}$, and
rewrite the second term in terms of $\eta$. Solving for $p$ yields
$p=e^\eta/(1+e^\eta)$, so $1-p=1/(1+e^\eta)$ and
$\log(1-p)=-\log(1+e^\eta)$. Exponentiating both sides gives

$$
P(X=x) = \exp\bigl(x\eta - \log(1+e^\eta)\bigr), \qquad x\in\{0,1\}.
$$

The coin's probability is now the exponential of a linear function of $x$,
minus a term that does not depend on $x$ at all. That second term is simply
the normalizer: the two unnormalized weights are $e^{0}=1$ and $e^\eta$, and
they sum to $1+e^\eta$.

Three ingredients from this expression will reappear throughout the section.
The *statistic* $T(x)=x$ is the part of the outcome that matters. The
*natural parameter* $\eta$ multiplies it; for the Bernoulli distribution, this
is the log-odds, also known as the *logit*. Finally, the *log-partition
function* $A(\eta)=\log(1+e^\eta)$ is the logarithm of the normalizer, a
function you may recognize as the *softplus*. Reading our earlier substitution
in reverse,

$$
p = \frac{e^\eta}{1+e^\eta} = \operatorname{sigmoid}(\eta),
\qquad \operatorname{sigmoid}(\eta)=\frac{1}{1+e^{-\eta}}.
$$

The sigmoid of logistic regression is therefore nothing more than the map from
the coin's natural parameter back to its mean. Nobody had to invent it.

For our running coin example, the empirical bias $9/13$ corresponds to the
natural parameter

$$
\hat\eta = \log\frac{9/13}{4/13} = \log\frac94 \approx 0.811,
$$

and the sigmoid takes it right back:
$\operatorname{sigmoid}(\log\frac94)=\frac{9/4}{1+9/4}=\frac{9}{13}$.
Shortly, we will see why $\hat\eta$ is the maximum-likelihood value of the
log-odds, and what the curvature of the loss around $\hat\eta$ tells us about
how uncertain that estimate is.

### Definition and Members

Our rewriting of the Bernoulli had three moving parts: a statistic of the
data, a parameter that multiplies it, and a normalizer. Generalizing to several
statistics at once, while allowing a fixed, parameter-free weight on each
outcome, gives a single template that captures most of our catalogue.

A family of distributions is an *exponential family* if its probability mass
or density function can be written as

$$
p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,\exp\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).
$$
:eqlabel:`eq_mdl-exp_family`

In this form, the data and the parameters interact only through the inner
product $\boldsymbol\eta^\top T(\mathbf x)$. The *base measure*
$h(\mathbf x)\ge0$ assigns a fixed weight to each outcome independently of the
parameters, and the region where $h>0$ defines the support of the distribution.
The *natural parameters* $\boldsymbol\eta\in\mathbb R^d$ are the adjustable
parameters, and the *sufficient statistics* $T(\mathbf x)\in\mathbb R^d$ are
the features of the data they multiply. The *log-partition function* is the
logarithm of the normalizing constant,

$$
A(\boldsymbol\eta) = \log\int h(\mathbf x)\,\exp\bigl(\boldsymbol\eta^\top T(\mathbf x)\bigr)\,d\mathbf x,
$$
:eqlabel:`eq_mdl-log_partition`

with an integral for continuous densities and a sum for discrete $\mathbf x$.
In plain terms, $A$ is whatever constant makes :eqref:`eq_mdl-exp_family`
integrate or sum to one, and the valid natural parameters are those for which
$A(\boldsymbol\eta)$ is finite. The Bernoulli distribution is the special case
$h(x)=1$, $T(x)=x$, and $A(\eta)=\log(1+e^\eta)$.

Why are the features $T(\mathbf x)$ called *sufficient*? Because they hold
everything the data can tell us about $\boldsymbol\eta$. For $n$ independent
observations, the joint likelihood factors as

$$
\prod_{i=1}^n p(\mathbf x_i\mid\boldsymbol\eta)
= \Bigl(\prod_{i=1}^n h(\mathbf x_i)\Bigr)
\exp\Bigl(\boldsymbol\eta^\top\sum_{i=1}^n T(\mathbf x_i) - nA(\boldsymbol\eta)\Bigr).
$$

The parameters touch the data only through the sum $\sum_i T(\mathbf x_i)$.
Any two datasets that share the same sum produce likelihoods differing only by
a constant factor independent of $\boldsymbol\eta$, so they yield the same
parameter estimates. For a coin, this sum is simply the total number of heads.
Any permutation of the 13 flips has the same probability under every bias, so
once we know that 9 flips came up heads, the order in which they appeared adds
nothing further about the coin.

Matching pieces against :eqref:`eq_mdl-exp_family` quickly identifies the
other members of the family. For a Poisson distribution with rate $\lambda$,
the probability of $k$ events is
$\frac{\lambda^k e^{-\lambda}}{k!}=\frac{1}{k!}\exp(k\log\lambda-\lambda)$,
so we read off $h(k)=1/k!$, $T(k)=k$, $\eta=\log\lambda$, and
$A(\eta)=e^\eta=\lambda$. A Gaussian with unknown mean and variance needs two
statistics. Expanding the square in its exponent and collecting the terms in
$x$ and $x^2$ gives

$$
\frac{1}{\sqrt{2\pi}}\exp\Bigl(\underbrace{\tfrac{\mu}{\sigma^2}}_{\eta_1}\,x
+ \underbrace{\bigl(-\tfrac{1}{2\sigma^2}\bigr)}_{\eta_2}\,x^2
- \underbrace{\bigl(\tfrac{\mu^2}{2\sigma^2}+\log\sigma\bigr)}_{A(\boldsymbol\eta)}\Bigr),
$$

so $T(x)=(x,x^2)^\top$ and $h(x)=1/\sqrt{2\pi}$. Notice that the second
natural parameter must be strictly negative: when $\eta_2<0$, the factor
$e^{\eta_2 x^2}$ supplies the Gaussian's rapid tail decay, whereas for
$\eta_2\ge0$ the normalizing integral for $A$ diverges.
:numref:`tab_mdl-exp-family-members` collects these and several other common
members.

:Members of the exponential family. In each row the probability or density is $h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$; the binomial requires a fixed number of trials $n$ and the Laplace a fixed location $\mu$.
:label:`tab_mdl-exp-family-members`

| distribution | $h(\mathbf x)$ | $T(\mathbf x)$ | $\boldsymbol\eta$ | $A(\boldsymbol\eta)$ |
|:--|:--|:--|:--|:--|
| Bernoulli($p$) | $1$ | $x$ | $\log\frac{p}{1-p}$ | $\log(1+e^\eta)$ |
| Binomial($n,p$) | $\binom nx$ | $x$ | $\log\frac{p}{1-p}$ | $n\log(1+e^\eta)$ |
| Poisson($\lambda$) | $1/k!$ | $k$ | $\log\lambda$ | $e^\eta$ |
| Categorical, one-hot $\mathbf y$ | $1$ | $\mathbf y$ | logits $\mathbf z$ | $\log\sum_k e^{z_k}$ |
| Exponential($\lambda$) | $1$ for $x\ge0$ | $x$ | $-\lambda$ | $-\log(-\eta)$ |
| $\mathcal N(\mu,\sigma^2)$ | $1/\sqrt{2\pi}$ | $(x,x^2)$ | $\bigl(\frac{\mu}{\sigma^2},-\frac{1}{2\sigma^2}\bigr)$ | $-\frac{\eta_1^2}{4\eta_2}-\frac12\log(-2\eta_2)$ |
| Laplace($\mu,b$) | $1$ | $\lvert x-\mu\rvert$ | $-1/b$ | $\log(-2/\eta)$ |

The categorical row expresses class probabilities through the logits, giving
rise to the softmax, whose origins we explore in
:numref:`subsec_mdl-softmax-origins`. Two further members live in higher
dimensions: the multivariate Gaussian, with sufficient statistics $\mathbf x$
and $\mathbf x\mathbf x^\top$ and natural parameters
$\boldsymbol\Sigma^{-1}\boldsymbol\mu$ and $-\frac12\boldsymbol\Sigma^{-1}$,
and the multinomial with a fixed number of draws. Keep in mind that the split
among $h$, $\boldsymbol\eta$, and $T$ is not unique. The one redundancy that
will matter later sits in the categorical row: because the $K$ entries of a
one-hot vector always sum to one, one of the $K$ logits is redundant
(:numref:`subsec_mdl-moment-matching`).

### Distributions Outside the Family

Not every distribution fits inside :eqref:`eq_mdl-exp_family`, and the two
classic exceptions fail for illuminatingly different reasons.

First, consider the uniform distribution $U(0,\theta)$, whose density is
$1/\theta$ on $[0,\theta]$ and zero outside. Here the support itself moves with
the parameter $\theta$. By contrast, every exponential family has a fixed
support, the set where $h>0$, because the exponential factor never vanishes. A
moving support changes estimation fundamentally. Suppose we observe three
draws: $0.1$, $0.2$, and $0.9$. The likelihood is $\theta^{-3}$ whenever
$\theta\ge0.9$ and $0$ whenever $\theta<0.9$ (a smaller $\theta$ could never
have generated $0.9$), so the maximum-likelihood estimate is the sample
maximum, $\hat\theta=\max_i x_i=0.9$.

Inside an exponential family, as we will prove shortly, maximum likelihood
always equates the model's mean of the sufficient statistic to its sample
average. Applying that recipe here sets the mean $\theta/2$ equal to
$\bar x=0.4$, yielding $\theta=0.8$, a value under which the observed draw
$0.9$ is impossible. Because the uniform likelihood depends on the data through
their maximum rather than a sum, the machinery of this section does not apply.
As we will see in :numref:`sec_mdl-maximum_likelihood` (Exercise
:exref:`ex_mdl-maximum-likelihood-moving-support`), the two estimators also
converge at different rates as the sample size grows.

Heavy-tailed distributions fall outside the family for a second reason. The
*Student-$t$* distribution arises by taking a Gaussian whose precision
$1/\sigma^2$ is itself random; averaging over that precision replaces the
rapidly decaying $e^{-x^2}$ tails with polynomial ones (Exercise
:exref:`ex_mdl-distributions-student-t` derives its density). With unit scale
and $\nu$ degrees of freedom, the likelihood of $n$ observations is a product
of factors $\bigl(1+(x_i-\mu)^2/\nu\bigr)^{-(\nu+1)/2}$, which cannot be
compressed into any fixed number of sums of the data. In fact, a classical
theorem of Pitman, Koopman, and Darmois states that among regular families with
fixed support, only exponential families possess a sufficient statistic of
fixed dimension :cite:`Darmois.1935,Koopman.1936,Pitman.1936`.

As a practical consequence, the convexity of the negative log-likelihood, which
we will prove holds for every exponential family, can break down outside it.
For a Student-$t$ distribution, when the data fall into two well-separated
clusters, the negative log-likelihood for the location $\mu$ develops a
separate local minimum near each cluster.

## Maximum Entropy
:label:`subsec_mdl-max-entropy-derivation`

So far, the exponential form has appeared simply by rewriting familiar
distributions. A deeper question asks where this form comes from in the first
place. Suppose all we know about a random quantity is the average value of a
few of its statistics. Which distribution should we choose? Remarkably, the
distribution that assumes nothing beyond those averages is always a member of
the exponential family.

### Entropy and a Loaded Die

To make "assumes nothing beyond the averages" precise, we need a way to
measure how spread out, or uncommitted, a distribution is. For a distribution
$p=(p_1,\dots,p_K)$ on a finite set, the *entropy* is

$$
H[p] = -\sum_{k=1}^K p_k\log p_k,
$$

measured in nats, with the convention $0\log0=0$. Each term weights the
surprise $-\log p_k$ of outcome $k$ by its probability, so entropy is simply
the average surprise. A fair six-sided die assigns every face a surprise of
$\log6$, giving an entropy of $\log6\approx1.79$ nats. A die that always lands
on six holds no surprises at all and has entropy $0$. Among all distributions
on $K$ outcomes, the uniform distribution achieves the largest possible
entropy, $\log K$, a fact we will prove in a moment.

E. T. Jaynes framed the problem with a concrete puzzle :cite:`Jaynes.1957`.
Suppose someone tells you that a six-sided die averages $4.5$ rather than
$3.5$, and nothing else. Many distributions on $\{1,\dots,6\}$ have mean
$4.5$. Putting probability $1/2$ on $3$ and $1/2$ on $6$ is one of them, yet
it boldly asserts that $1$, $2$, $4$, and $5$ can never occur, a claim with no
support in what we were told. Its entropy is only $\log2\approx0.69$.

Jaynes's *principle of maximum entropy* selects, among all distributions
matching the known averages, the one with the largest entropy, the least
committal choice consistent with our knowledge. Let us state the answer before
proving it. The unique maximizer is an *exponential tilt* of the uniform
distribution,

$$
p_k = \frac{e^{\eta k}}{\sum_{j=1}^6 e^{\eta j}}, \qquad k=1,\dots,6,
$$

with $\eta$ tuned so that the mean equals $4.5$. The distribution concentrated
on threes and sixes surrendered $\log6-\log2\approx1.10$ of the fair die's
$1.79$ nats to hit this mean. Before running the code below, pause and guess
how much entropy the exponential tilt has to give up. Because the mean of the
tilt increases monotonically with $\eta$, simple bisection finds the exact
value of $\eta$.

```{.python .input #mdl-exponential-family-entropy-and-a-loaded-die}
k = np.arange(1, 7)
def tilt(eta):                                   # p_k proportional to exp(eta k)
    w = np.exp(eta * k)
    return w / w.sum()

lo, hi = -5.0, 5.0
for _ in range(50):                              # bisection: the mean increases with eta
    eta = (lo + hi) / 2
    lo, hi = (eta, hi) if (tilt(eta) * k).sum() < 4.5 else (lo, eta)
p = tilt(eta)
print('eta =', round(eta, 3), ' p =', p.round(3))
print('H =', round(-(p * np.log(p)).sum(), 3), ' log 6 =', round(np.log(6), 3))
```

The resulting distribution is geometric across the faces: each face is
$e^{0.371}\approx1.45$ times as likely as the face below it. Its entropy is
$1.61$ nats, giving up only $0.18$ nats compared with the fair die, and
setting $\eta=0$ recovers the fair die. :numref:`fig_mdl-prob-maxent-die`
compares the exponential tilts for three different required means.

![Maximum-entropy distributions of a die whose average roll is fixed at $3.5$, $4.5$, and $5.5$ (black triangles). Each is an exponential tilt $p_k\propto e^{\eta k}$ of the uniform distribution; as the required mean rises, the tilt $\eta$ grows and the entropy $H$ falls.](../img/mdl-prob-maxent-die.svg)
:label:`fig_mdl-prob-maxent-die`

Notice that this tilt already has the exponential-family form
:eqref:`eq_mdl-exp_family`, with $h\equiv1$ on the six faces, statistic
$T(k)=k$, and natural parameter $\eta$. Fixing the mean of a statistic and
maximizing entropy led us straight to the exponential of a multiple of that
statistic. That is no coincidence of the die.

### The Maximum-Entropy Proposition

To state the general theorem, we need one small extension of entropy, because
some outcome spaces carry natural baseline weights. For a distribution $q$ and
a base measure $h$, the *entropy relative to* $h$ is

$$
H_h[q] = -\sum_{\mathbf x} q(\mathbf x)\log\frac{q(\mathbf x)}{h(\mathbf x)}
$$

for discrete outcomes, with an integral in place of the sum for continuous
densities. Setting $h\equiv1$ recovers the ordinary entropy defined above.

The proof compares any candidate distribution $q$ directly against a member
$p$ of the exponential family. Given two distributions $p$ and $q$ on the same
space, with $p(\mathbf x)>0$ wherever $q(\mathbf x)>0$, the
*Kullback--Leibler divergence* of $q$ from $p$ is
$D_{\textrm{KL}}(q\|p)=\sum q\log(q/p)$ summed over outcomes with $q>0$, or
the corresponding integral for densities :cite:`Cover.Thomas.1999`.
:numref:`sec_mdl-information_theory` studies entropy and this divergence in
depth. All we need here is one elementary fact, proved by a single tangent
line: because the logarithm is concave, $\log u\le u-1$ for every $u>0$, with
equality only at $u=1$.

**Lemma (Gibbs' inequality).** *The Kullback--Leibler divergence is never
negative,*

$$
D_{\textrm{KL}}(q\|p) = \sum_{\mathbf x:\,q(\mathbf x)>0} q(\mathbf x)\log\frac{q(\mathbf x)}{p(\mathbf x)} \ge 0,
$$
:eqlabel:`eq_mdl-exp_family_gibbs`

*with equality if and only if $q=p$.*

**Proof.** Apply the tangent-line bound $\log u \le u-1$ with
$u=p(\mathbf x)/q(\mathbf x)$ and sum over all outcomes with $q(\mathbf x)>0$:

$$
-D_{\textrm{KL}}(q\|p) = \sum_{q>0} q\log\frac pq \le \sum_{q>0} q\Bigl(\frac pq-1\Bigr) = \sum_{q>0} p - 1 \le 0.
$$

Equality in the first inequality requires $p=q$ wherever $q>0$. That forces
$\sum_{q>0}p=1$, leaving $p$ with no remaining probability mass outside the
support of $q$, so $p=q$ everywhere. $\blacksquare$

**Proposition (maximum entropy).** *Let
$p(\mathbf x)=h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$
be a member of an exponential family, and let $q$ be any distribution (with
a density, in the continuous case) that vanishes wherever $h$ vanishes, has a
finite mean statistic $\mathbb{E}_q[T(\mathbf x)]$, and matches $p$ on that
mean: $\mathbb{E}_q[T(\mathbf x)]=\mathbb{E}_p[T(\mathbf x)]$. Then
$H_h[q]\le H_h[p]$, with equality if and only if $q=p$.*

**Proof.** Split the logarithm in $H_h[q]$ by inserting $p$:
$\log\frac qh=\log\frac qp+\log\frac ph$. The first term yields
$-D_{\textrm{KL}}(q\|p)\le0$ by Gibbs' inequality, which applies because $p>0$
wherever $h>0$. The second term is linear in the sufficient statistic,
$\log\frac ph=\boldsymbol\eta^\top T-A(\boldsymbol\eta)$, so $q$ enters only
through its mean of $T$:

$$
-\sum_{\mathbf x} q\log\frac ph = -\boldsymbol\eta^\top\mathbb{E}_q[T] + A(\boldsymbol\eta)
= -\boldsymbol\eta^\top\mathbb{E}_p[T] + A(\boldsymbol\eta) = H_h[p].
$$

The middle equality uses the mean constraint, and the last repeats the same
calculation with $p$ in place of $q$. Combining the two terms gives
$H_h[q]=H_h[p]-D_{\textrm{KL}}(q\|p)\le H_h[p]$, with equality only when
$q=p$. $\blacksquare$

The key step is the second one: because $\log(p/h)$ is linear in $T$, the
cross term depends on $q$ solely through $\mathbb{E}_q[T]$, which the
constraint holds fixed. In plain words, *whenever a member of the exponential
family satisfies the mean constraints, it is the unique distribution of largest
entropy among all distributions that satisfy them*. Remarkably, no Lagrange
multipliers are needed, and the proof certifies a global maximum rather than
merely a stationary point.

For the loaded die, the tilt with $\eta\approx0.371$ belongs to the family
with $h\equiv1$ and $T(k)=k$ and has mean $4.5$, so it is the unique
maximizer. Three familiar distributions follow immediately as corollaries:

* *No constraint* ($T\equiv0$): every $q$ is admissible, and the maximizer is
  $p\propto h$, provided $h$ has a finite sum. On a finite set with
  $h\equiv1$, this is the uniform distribution with entropy $\log K$; the fair
  die is the case $K=6$.
* *Mean and variance on $\mathbb R$*: with $T(x)=(x,x^2)$ and constant $h$,
  the maximizer is the Gaussian. Because a constant $h$ shifts $H_h$ by a
  fixed constant, the Gaussian has the largest entropy among all densities
  with a prescribed mean and variance.
* *Mean on the counts* $\{0,1,2,\dots\}$ with $T(k)=k$: taking $h(k)=1/k!$
  makes the maximizer of $H_h$ with mean $m$ the Poisson distribution with
  rate $\lambda=m$, whereas taking $h\equiv1$ yields the geometric
  distribution $p_k=(1-r)r^k$ with $r=m/(1+m)$. The base measure decides which
  distribution emerges, as Exercise
  :exref:`ex_mdl-distributions-max-entropy-integers` explores.

Be careful not to drop the hypothesis that some member of the family meets the
constraints. Suppose we fix only the mean $m$ on all of $\mathbb R$, with
$h\equiv1$. The candidate density $e^{\eta x}$ is not integrable on
$\mathbb R$ for any $\eta$, so no member exists. Nor does any maximizer exist
at all: the Gaussian $\mathcal N(m,s^2)$ has mean $m$ and entropy
$\frac12\log(2\pi e s^2)$, which grows without bound as $s\to\infty$.

## The Log-Partition Function

In the catalogue of :numref:`sec_mdl-distributions`, every mean and variance
seemed to demand its own trick: $X^2=X$ for the Bernoulli, a sum of
independent flips for the binomial, an index shift in the series for the
Poisson, and integration by parts for the Gaussian. Inside the exponential
family, all of those calculations collapse into one operation: differentiating
the log-partition function $A$. The first derivative gives the mean, the
second gives the variance, and the very same function governs maximum
likelihood.

### The Moment Property

Recall that $A(\boldsymbol\eta)$ is the logarithm of an integral or sum of
exponentials $\exp(\boldsymbol\eta^\top T(\mathbf x))$. Differentiating such a
logarithm pulls down a factor of $T(\mathbf x)$, weighted by the probability
of $\mathbf x$, which is simply the expectation of $T$.

**Proposition (mean of the sufficient statistics).** *For an
exponential-family distribution :eqref:`eq_mdl-exp_family`,*

$$
\nabla_{\boldsymbol\eta} A(\boldsymbol\eta) = \mathbb{E}[T(\mathbf x)].
$$
:eqlabel:`eq_mdl-exp_family_moment`

**Proof.** Differentiate :eqref:`eq_mdl-log_partition` under the integral
sign. This interchange is justified in the interior of the set of natural
parameters where $A(\boldsymbol\eta)<\infty$ :cite:`Folland.1999`; that set is
convex (by Hölder's inequality), and we take $\boldsymbol\eta$ in its interior
from here on. Writing
$Z(\boldsymbol\eta)=e^{A(\boldsymbol\eta)}=\int h(\mathbf x)e^{\boldsymbol\eta^\top T(\mathbf x)}d\mathbf x$,
the chain rule gives $\nabla A = (\nabla Z)/Z$, where

$$
\nabla_{\boldsymbol\eta} Z
= \int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x .
$$

Dividing by $Z=e^{A}$ brings the factor $e^{-A}$ inside the integral,
completing the density :eqref:`eq_mdl-exp_family`:

$$
\nabla_{\boldsymbol\eta} A
= \frac{1}{Z}\int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x
= \int T(\mathbf x)\,\underbrace{h(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)}}_{p(\mathbf x\mid\boldsymbol\eta)}\,d\mathbf x
= \mathbb{E}[T(\mathbf x)] . \quad\blacksquare
$$

Differentiating $\nabla A=(\nabla Z)/Z$ a second time, under the same
interchange, produces the covariance matrix of the sufficient statistics:

$$
\nabla^2 A
= \frac{\nabla^2 Z}{Z} - \frac{(\nabla Z)(\nabla Z)^\top}{Z^2}
= \mathbb{E}[T\,T^\top] - \mathbb{E}[T]\,\mathbb{E}[T]^\top
= \textrm{Cov}(T).
$$

In short, the slope of $A$ is the mean of $T$, and its curvature is the
covariance of $T$. Because every covariance matrix is positive semidefinite
($\mathbf v^\top\textrm{Cov}(T)\mathbf v=\textrm{Var}(\mathbf v^\top T)\ge0$
for all $\mathbf v$; :numref:`subsec_mdl-psd`), the log-partition function $A$
is always convex.

Let us check this on two familiar examples. For the coin,
$A(\eta)=\log(1+e^\eta)$ gives $A'(\eta)=\operatorname{sigmoid}(\eta)=p$, the
Bernoulli mean, and
$A''(\eta)=\operatorname{sigmoid}(\eta)\bigl(1-\operatorname{sigmoid}(\eta)\bigr)=p(1-p)$,
its variance. At $\hat\eta=\log\frac94$, these evaluate to $9/13\approx0.692$
and $\frac{9}{13}\cdot\frac{4}{13}=\frac{36}{169}\approx0.213$. For the
Poisson distribution, $A(\eta)=e^\eta$ is its own derivative, so at
$\lambda=4$ ($\eta=\log4$) both $A'$ and $A''$ equal $4$. Since moments are
just derivatives, automatic differentiation computes them directly for any $A$
we can write in code. Let us verify both cases.

```{.python .input #distributions-exp-family}
#@tab mxnet
def derivatives(A, eta):
    """First and second derivative of the scalar function A at eta."""
    x = nd.array([eta], dtype='float64')
    x.attach_grad()
    with autograd.record():           # the np-module ops lack second derivatives
        dA = autograd.grad(A(x), [x], create_graph=True)[0]
    dA.backward()
    return dA.asscalar(), x.grad.asscalar()

cases = [('Bernoulli', lambda x: nd.log(1 + nd.exp(x)), math.log(9 / 4), 9 / 13, 36 / 169),
         ('Poisson', nd.exp, math.log(4), 4, 4)]
for name, A, eta, mean, var in cases:
    dA, d2A = derivatives(A, eta)
    print(f"{name:9s} A' = {dA:.4f} (mean {mean:.4f})  A'' = {d2A:.4f} (variance {var:.4f})")
```

```{.python .input #distributions-exp-family}
#@tab pytorch
def derivatives(A, eta):
    """First and second derivative of the scalar function A at eta."""
    eta = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    (dA,) = torch.autograd.grad(A(eta), eta, create_graph=True)
    (d2A,) = torch.autograd.grad(dA, eta)
    return dA.item(), d2A.item()

cases = [('Bernoulli', torch.nn.functional.softplus, math.log(9 / 4), 9 / 13, 36 / 169),
         ('Poisson', torch.exp, math.log(4), 4, 4)]
for name, A, eta, mean, var in cases:
    dA, d2A = derivatives(A, eta)
    print(f"{name:9s} A' = {dA:.4f} (mean {mean:.4f})  A'' = {d2A:.4f} (variance {var:.4f})")
```

```{.python .input #distributions-exp-family}
#@tab tensorflow
def derivatives(A, eta):
    """First and second derivative of the scalar function A at eta."""
    eta = tf.Variable(eta, dtype=tf.float64)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            a = A(eta)
        dA = inner.gradient(a, eta)
    d2A = outer.gradient(dA, eta)
    return float(dA), float(d2A)

cases = [('Bernoulli', tf.math.softplus, math.log(9 / 4), 9 / 13, 36 / 169),
         ('Poisson', tf.exp, math.log(4), 4, 4)]
for name, A, eta, mean, var in cases:
    dA, d2A = derivatives(A, eta)
    print(f"{name:9s} A' = {dA:.4f} (mean {mean:.4f})  A'' = {d2A:.4f} (variance {var:.4f})")
```

```{.python .input #distributions-exp-family}
#@tab jax
def derivatives(A, eta):
    """First and second derivative of the scalar function A at eta."""
    return float(jax.grad(A)(eta)), float(jax.grad(jax.grad(A))(eta))

cases = [('Bernoulli', jax.nn.softplus, math.log(9 / 4), 9 / 13, 36 / 169),
         ('Poisson', jnp.exp, math.log(4), 4, 4)]
for name, A, eta, mean, var in cases:
    dA, d2A = derivatives(A, eta)
    print(f"{name:9s} A' = {dA:.4f} (mean {mean:.4f})  A'' = {d2A:.4f} (variance {var:.4f})")
```

Both derivatives match the analytical mean and variance to all four printed
digits. Notice how the Poisson's famous equality of mean and variance, which
required a power-series calculation in the catalogue, now reduces to a single
line: $A=A'=A''$ for $A(\eta)=e^\eta$.

### Maximum Likelihood as Moment Matching
:label:`subsec_mdl-moment-matching`

The moment property turns maximum likelihood into a remarkably clean condition
on expectations: the fitted model must reproduce the sample average of the
sufficient statistics. Given independent observations
$\mathbf x_1,\dots,\mathbf x_n$, taking the negative logarithm of the product
likelihood from :eqref:`eq_mdl-exp_family` gives

$$
-\sum_{i=1}^{n} \log p(\mathbf x_i\mid\boldsymbol\eta)
= n\,A(\boldsymbol\eta) - \boldsymbol\eta^\top \sum_{i=1}^{n} T(\mathbf x_i)
- \sum_{i=1}^{n} \log h(\mathbf x_i).
$$
:eqlabel:`eq_mdl-exp_family_nll`

Here the data enter solely through the sum $\sum_{i=1}^n T(\mathbf x_i)$.
Because $A$ is convex and the second term is linear in $\boldsymbol\eta$, the
negative log-likelihood is convex in $\boldsymbol\eta$. Setting its gradient
to zero and applying the moment property :eqref:`eq_mdl-exp_family_moment`
yields

$$
\nabla A(\hat{\boldsymbol\eta}) = \bar T, \qquad \bar T=\frac1n\sum_{i=1}^n T(\mathbf x_i).
$$

Maximum likelihood therefore sets the model's expected sufficient statistic
equal to its observed sample average, a principle called *moment matching*.
And because the objective is convex, any solution to this equation is
automatically a global minimizer.

Let us see what this gives for our coin with 9 heads in 13 flips. Its negative
log-likelihood is $13A(\eta)-9\eta=13\log(1+e^\eta)-9\eta$. The derivative
$13\operatorname{sigmoid}(\eta)-9$ vanishes precisely where
$\operatorname{sigmoid}(\hat\eta)=9/13$, which gives
$\hat\eta=\log\frac94\approx0.811$. At this minimum, the loss equals

$$
13\log\frac{13}{4} - 9\log\frac94 \approx 8.02 \text{ nats},
$$

which is 13 times the entropy of a coin with bias $9/13$,
$-\frac9{13}\log\frac9{13}-\frac4{13}\log\frac4{13}\approx0.617$. At the
optimum, the average loss per flip equals the entropy of the fitted coin.

This identity holds for every exponential family with $h\equiv1$: the
average negative log-likelihood at $\hat{\boldsymbol\eta}$ is
$A(\hat{\boldsymbol\eta})-\hat{\boldsymbol\eta}^\top\bar T$, while the
entropy of the fitted distribution is
$A(\hat{\boldsymbol\eta})-\hat{\boldsymbol\eta}^\top\nabla A(\hat{\boldsymbol\eta})$.
Because $\nabla A(\hat{\boldsymbol\eta})=\bar T$, the two numbers coincide.

The second derivative of the loss at the minimum is
$13A''(\hat\eta)=13\cdot\frac{36}{169}=\frac{36}{13}\approx2.77$. Large
curvature means the loss climbs steeply away from $\hat\eta$, pinning down the
log-odds tightly. Thirteen flips leave plenty of wiggle room: moving by
$1/\sqrt{2.77}\approx0.60$ away from $\hat\eta\approx0.81$ raises the loss by
only about half a nat. As we will formalize in
:numref:`sec_mdl-maximum_likelihood`, this expected curvature is the *Fisher
information*
$I(\theta) = \textrm{Var}_\theta(\nabla_\theta \log p(X \mid \theta)) = -\mathbb{E}_\theta[\nabla_\theta^2 \log p(X \mid \theta)]$,
which turns directly into an asymptotic error bar.
:numref:`fig_mdl-prob-log-partition` illustrates both the tangent of $A$ at
$\hat\eta$ and the shape of the negative log-likelihood.

![Left: the Bernoulli log-partition function $A(\eta)=\log(1+e^\eta)$ with its asymptotes and its tangent at $\hat\eta=\log(9/4)$, whose slope is the mean $9/13$. Right: the negative log-likelihood $13A(\eta)-x\eta$ for $x=9$ heads in 13 flips, with its minimum at $\hat\eta$, and for $x=13$ heads, which decreases for every $\eta$ and has no finite minimizer.](../img/mdl-prob-log-partition.svg)
:label:`fig_mdl-prob-log-partition`

There is an important caveat: the moment-matching equation
$\nabla A(\hat{\boldsymbol\eta})=\bar T$ has a solution only if $\bar T$ is
a mean that some member of the family can actually attain. For a Bernoulli
coin, the attainable means are $\operatorname{sigmoid}(\eta)$ for real $\eta$,
namely the open interval $(0,1)$. If all 13 flips come up heads, then
$\bar T=1$ lies on the boundary and is not attainable. The negative
log-likelihood becomes $13A(\eta)-13\eta=13\log(1+e^{-\eta})$, which drops
monotonically for every $\eta$: it equals $9.01$ at $\eta=0$, $0.087$ at
$\eta=5$, and $0.0006$ at $\eta=10$. It approaches $0$ without ever reaching
it, so no finite minimizer exists; the fitted probability approaches $1$ as
$\hat\eta\to\infty$.

Logistic regression hits the exact same wall whenever a hyperplane separates
the two classes in the training set: the loss keeps decreasing as the weights
scale up, driving them toward infinity. Adding a prior on the parameters
(which we introduce at the end of this section) or using weight decay restores
a finite optimum, as we will see in :numref:`sec_mdl-maximum_likelihood`
(Exercise :exref:`ex_mdl-maximum-likelihood-small-samples-boundary`).

When a minimizer does exist, it is unique whenever the negative log-likelihood
is strictly convex. That holds if $\textrm{Cov}(T)$ is positive definite,
meaning no nontrivial linear combination of the sufficient statistics is
constant across the support. Such a parameterization is called *minimal*. The
categorical distribution parameterized by all $K$ logits is not minimal: the
$K$ one-hot coordinates always sum to one, so adding the same constant to
every logit leaves all probabilities unchanged. As a result, its negative
log-likelihood is flat along the direction $\mathbf 1$, and its minimizers
form a line rather than a point. Deep learning libraries keep all $K$ logits
anyway and exploit this shift invariance to subtract the largest logit before
exponentiating for numerical stability (:numref:`sec_softmax_concise`).

### Generalized Linear Models
:label:`subsec_mdl-glm`

Supervised learning models predict a conditional distribution over a label $y$
given an input $\mathbf x$. A *generalized linear model* chooses that
distribution from an exponential family :eqref:`eq_mdl-exp_family` and lets
the input enter through a linear map $\mathbf W^\top\mathbf x$ (folding any
bias term into a constant feature). The cleanest choice sets this linear map
directly equal to the natural parameter,
$\boldsymbol\eta = \mathbf W^\top\mathbf x$, a pairing known as the
*canonical link*. Writing
$\boldsymbol\mu(\mathbf x) = \nabla A(\boldsymbol\eta) = \mathbb{E}[T(y)\mid\mathbf x]$
for the model's predicted mean of the sufficient statistic, the negative
log-likelihood of a single example is
$A(\boldsymbol\eta) - \boldsymbol\eta^\top T(y) - \log h(y)$. By the moment
property :eqref:`eq_mdl-exp_family_moment`, its gradient with respect to
$\boldsymbol\eta$ is simply $\boldsymbol\mu(\mathbf x) - T(y)$. Applying the
chain rule through $\boldsymbol\eta = \mathbf W^\top\mathbf x$ immediately
gives

$$
\nabla_{\mathbf W}\bigl(-\log p(y\mid\mathbf x)\bigr)
= \mathbf x\,\bigl(\boldsymbol\mu(\mathbf x) - T(y)\bigr)^\top .
$$
:eqlabel:`eq_mdl-glm-gradient`

Whatever the exponential family, the gradient is always the input multiplied
by the prediction error: predicted mean minus observed statistic. With the
moment property in hand, the mystery from the opening of this section resolves
in two lines. Moreover, the loss is convex in $\mathbf W$, since it composes
the convex function $A$ with a linear function of $\mathbf W$ and adds a
linear term. Note that this tidy gradient belongs specifically to the
canonical link: the probit model in :numref:`subsec_mdl-softmax-origins` feeds
the same linear score into a Gaussian CDF instead, and its gradient picks up
an extra factor that varies with the prediction. Four familiar models arise
simply by swapping the exponential family:

| label $y$ | family | $T(y)$ | mean $\boldsymbol\mu(\mathbf x)$ | model |
|:--|:--|:--|:--|:--|
| real number | Gaussian, variance 1 | $y$ | $\eta$ | linear regression |
| yes or no | Bernoulli | $y \in \{0, 1\}$ | $\operatorname{sigmoid}(\eta)$ | logistic regression |
| one of $K$ classes | categorical | one-hot $\mathbf y$ | $\mathrm{softmax}(\boldsymbol\eta)$ | softmax regression |
| count | Poisson | $y$ | $e^{\eta}$ | Poisson regression |

To see the rule on a less familiar label, consider a Poisson model for count
data. If the model predicts a mean of $e^\eta=3$ for an input and the observed
count is $y=5$, then the gradient of the loss with respect to $\eta$ is
$3-5=-2$. A gradient descent step therefore increases $\eta$ and pushes the
predicted mean upward toward $5$.

In the first row, fixing the Gaussian variance at $1$ puts the quadratic term
$e^{-y^2/2}$ into the base measure $h(y)$, so the negative log-likelihood
reduces to the squared loss of :numref:`sec_linear_regression` up to an
additive constant. A deep neural network keeps this final layer intact (a
linear map into natural parameters followed by the family's negative
log-likelihood), while replacing the raw input $\mathbf x$ with learned
representations. Although the overall loss is no longer convex in the network
weights, its gradient with respect to the natural parameters at the output
layer remains prediction minus observation :cite:`Bishop.2006`.

## Origins of the Softmax
:label:`subsec_mdl-softmax-origins`

Throughout the main text, classifiers convert a vector of $K$ logits
$\mathbf z$ into class probabilities using the *softmax*,

$$
\mathrm{softmax}(\mathbf z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \qquad k=1,\dots,K.
$$
:eqlabel:`eq_mdl-exp_family_softmax`

We just saw this map emerge from the categorical row of our exponential-family
tables. It has obvious desirable properties: each probability increases with
its own logit, and all probabilities are positive and sum to one. Yet many
other maps share those properties; normalizing $\log(1+e^{z_k})$ instead of
$e^{z_k}$, for instance, also produces positive probabilities that preserve
the ranking of the logits. What sets the softmax apart is that logit
differences are log-odds: $z_i-z_j=\log(p_i/p_j)$. Remarkably, three
independent arguments from three different fields all converge on this exact
formula: a model of noisy human choices, the Boltzmann distribution of
statistical physics, and the exponential family. The third is the shortest and
is the only one the rest of this section relies on.

### Noisy Choices and Gumbel Noise

The first argument models how discrete labels are generated in the first
place. Imagine a decision maker, such as a shopper choosing a product or an
annotator labeling an image, who assigns each option $k$ an underlying score
$z_k$ but makes occasional errors. A natural way to model noisy selection is to add
independent random noise $\epsilon_k$ to every score and pick the option with
the highest perturbed score:

$$
y = \operatorname*{argmax}_{k}\,(z_k+\epsilon_k).
$$

Under this model, the probability of class $k$ is the chance that option $k$
wins the noisy contest, so its formula is determined by the distribution of
the noise. Gaussian noise is the first thing most people try. With two options
and $\epsilon_k\sim\mathcal N(0,s^2)$, option 1 wins whenever
$\epsilon_2-\epsilon_1<z_1-z_2$. Because the difference
$\epsilon_2-\epsilon_1$ is itself Gaussian with variance $2s^2$, the winning
probability is

$$
P(y=1) = \Phi\Bigl(\frac{z_1-z_2}{\sqrt2\,s}\Bigr),
$$

where $\Phi$ denotes the standard Gaussian CDF (and $\phi$ its density). This
is the *probit* model: the choice probability depends only on the score gap,
measured in units of the noise scale.

Unfortunately, the probit model loses its closed form as soon as there are
more than two options. The probability that option $k$ beats all $K-1$ rivals
becomes an integral of a product of $K-1$ Gaussian CDFs, which has no
elementary formula and would have to be integrated numerically, along with its
gradient, for every training example at every step.

Switching from Gaussian to Gumbel noise eliminates that integral completely.
The standard *Gumbel* distribution has the CDF $G(x)=\exp(-e^{-x})$. It arises
naturally in extreme-value theory as the limiting distribution of the
suitably shifted maximum of many independent draws from an exponentially
decaying tail, such as a Gaussian or exponential distribution
:cite:`Gumbel.1954`, and :citet:`McFadden.1974` introduced it to model
economic choice.

**Proposition (Gumbel noise gives the softmax).** *If
$\epsilon_1,\dots,\epsilon_K$ are independent standard Gumbel random
variables, then the noisy argmax follows the softmax distribution,*

$$
P\Bigl(\operatorname*{argmax}_j\,(z_j+\epsilon_j)=k\Bigr) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} .
$$

**Proof.** Let $S=\sum_j e^{z_j}$. Each perturbed score $z_j+\epsilon_j$ has
CDF $G(t-z_j)=\exp(-e^{z_j}e^{-t})$. Conditioned on option $k$ achieving a
score of $t=z_k+\epsilon_k$, independence implies that all other $K-1$ options
fall below $t$ with probability
$\prod_{j\ne k}\exp(-e^{z_j}e^{-t})=\exp\bigl(-(S-e^{z_k})e^{-t}\bigr)$.
Multiplying by the density $e^{z_k}e^{-t}\exp(-e^{z_k}e^{-t})$ of
$t=z_k+\epsilon_k$, integrating over $t$, and substituting $v=e^{-t}$ yields

$$
P(y=k) = \int_{-\infty}^{\infty} e^{z_k}e^{-t}\exp\bigl(-Se^{-t}\bigr)\,dt
= e^{z_k}\int_0^{\infty} e^{-Sv}\,dv = \frac{e^{z_k}}{S} . \quad\blacksquare
$$

The calculation succeeds because a product of Gumbel CDFs is still a Gumbel
CDF. Across all $K$ options, the maximum score satisfies
$P\bigl(\max_j(z_j+\epsilon_j)\le t\bigr)=\exp(-Se^{-t})=G(t-\log S)$, meaning
the winning score is simply a standard Gumbel variable shifted by the
log-sum-exp $\log\sum_j e^{z_j}$. Viewed this way, log-sum-exp acts as a
smooth maximum of the logits, and its gradient, the softmax, gives the
probability that each option achieves that maximum. Exercise
:exref:`ex_mdl-distributions-exponential-races` offers a second proof of the
same identity using competing exponential waiting times.

Read in reverse, the proposition doubles as a sampling algorithm: to draw a
category from $\mathrm{softmax}(\mathbf z)$, add independent standard Gumbel
noise to each logit and take the argmax. This is the widely used *Gumbel-max
trick*.

For the remainder of :numref:`subsec_mdl-softmax-origins`, we follow the main
text and use $T$ to denote a temperature rather than a sufficient statistic.
Because
$\operatorname{argmax}_k(z_k+T\epsilon_k)=\operatorname{argmax}_k(z_k/T+\epsilon_k)$
for any $T>0$, scaling the Gumbel noise by $T$ samples from
$\mathrm{softmax}(\mathbf z/T)$: noisier decisions correspond to higher
temperatures. Let us test the Gumbel-max trick directly by drawing $10^5$
choices over four logits at $T=1$ and $T=2$ and comparing the empirical win
rates against the exact softmax probabilities.

```{.python .input #mdl-exponential-family-noisy-choices-and-gumbel-noise}
z = np.array([2.0, 1.0, 0.1, -1.0])                      # four logits
softmax = lambda v: np.exp(v) / np.exp(v).sum()
rng = np.random.default_rng()
n = 100_000
for T in (1.0, 2.0):                                      # noise scale = temperature
    winners = np.argmax(z + T * rng.gumbel(size=(n, 4)), axis=1)
    freq = np.bincount(winners, minlength=4) / n
    print(f'T = {T:.0f}: argmax frequencies {freq.round(3)}')
    print(f'      softmax(z / T)        {softmax(z / T).round(3)}'
          f'  max diff {np.abs(freq - softmax(z / T)).max():.3f}')
```

The empirical frequencies agree with $\mathrm{softmax}(\mathbf z/T)$ to within
a few thousandths, right at the Monte Carlo noise floor for $10^5$ draws.
Doubling the noise scale flattens the win rates in the exact same way as
halving the logits.

With two options, the difference of two independent standard Gumbel random
variables follows the logistic distribution, whose CDF is
$\operatorname{sigmoid}(x)=1/(1+e^{-x})$, so this *logit* model yields
$P(y=1)=\operatorname{sigmoid}(z_1-z_2)$. Because the sigmoid closely tracks a
rescaled Gaussian CDF ($|\operatorname{sigmoid}(1.7x)-\Phi(x)|<0.01$ for all
$x$), fitted probit and logit models produce nearly identical probabilities on
ordinary examples. Where they diverge is in the tails: on a confidently
mislabeled outlier, the logit loss grows only linearly in the score difference,
whereas the probit loss grows quadratically, allowing outliers to pull a
probit fit much harder (Exercise :exref:`ex_mdl-distributions-probit-logit`).

### Temperature and the Boltzmann Distribution

The second argument comes from statistical physics, where the exact same
exponential law governs energy states. A physical system in thermal
equilibrium at temperature $T$, such as a molecule in a gas, occupies a state
$k$ of energy $E_k$ with probability

$$
p_k = \frac{\exp\bigl(-E_k/(k_{\mathrm B}T)\bigr)}{\sum_j \exp\bigl(-E_j/(k_{\mathrm B}T)\bigr)},
$$

where $k_{\mathrm B}$ is Boltzmann's constant. Lower-energy states are more
probable, and the temperature $T$ controls how sharply probability
concentrates on them. Boltzmann discovered this distribution for gas
molecules, and :citet:`Gibbs.1902` built statistical mechanics upon it. If we
measure temperature in energy units (setting $k_{\mathrm B}=1$) and identify
energies with negative logits ($E_k=-z_k$), the Boltzmann law becomes
$\mathrm{softmax}(\mathbf z/T)$. In this language, a logit is a negative
energy, and the most likely class is simply the lowest-energy state.

Our maximum-entropy proposition derives the same formula without invoking
physics at all. A physicist fixes the mean energy and assumes nothing else; in
classification, the analogous constraint fixes the average logit
$\sum_k p_k z_k$. Take the $K$ classes as outcomes with $h\equiv1$ and treat
the logit $z_k$ as the statistic. Among all distributions on the $K$ classes
achieving a prescribed average logit (strictly between the smallest and
largest entries of $\mathbf z$), the maximum-entropy distribution is
$p_k\propto e^{\eta z_k}$. Whenever the target average exceeds the unweighted
mean of the logits, $\eta>0$, and setting $T=1/\eta$ recovers
$\mathrm{softmax}(\mathbf z/T)$. As $T$ cools from $\infty$ down to $0$, the
average logit rises smoothly from the mean of the logits to their maximum. For
the logits $\mathbf z=(2,1,0.1,-1)$ in our simulation, the standard softmax
($T=1$) achieves an average logit of $1.49$, sitting between the unweighted mean
$0.525$ and the maximum $2$.

### The Categorical as an Exponential Family

The third argument is the one we already uncovered earlier in this section.
Encoding a class label as a one-hot vector $\mathbf y$, the categorical
distribution is $P(\mathbf y)=\exp\bigl(\mathbf z^\top\mathbf y-A(\mathbf z)\bigr)$
with log-partition function $A(\mathbf z)=\log\sum_k e^{z_k}$, an exponential
family whose natural parameters are the logits. By the moment property, its
mean is $\nabla A(\mathbf z)=\mathrm{softmax}(\mathbf z)=\mathbb{E}[\mathbf y]$.
Its negative log-likelihood is automatically convex in $\mathbf z$, and by
:eqref:`eq_mdl-glm-gradient`, its gradient is simply
$\mathrm{softmax}(\mathbf z)-\mathbf y$.

All three arguments arrive at the same function, yet each illuminates a
different facet of it: noisy choices give us an exact sampler, the Boltzmann
distribution explains temperature scaling, and the exponential family hands us
a convex loss and its effortless gradient.

## Conjugate Priors

Maximum likelihood compresses 9 heads in 13 flips into a single number,
$9/13$. A point estimate alone cannot tell us how confident to be in it:
9 heads in 13 flips and 900 heads in 1300 flips produce the exact same ratio.
Representing our knowledge as a probability distribution over the coin's bias
captures that uncertainty directly, and ideally it should absorb each new flip
through a simple update rather than starting from scratch.

From here on, we treat the bias itself as an unknown random quantity and
denote it by $\theta$, matching the notation of the upcoming sections. A
distribution over $\theta$ before seeing the data is a *prior*. Bayes' rule
multiplies the prior by the likelihood and normalizes the result,
$p(\theta\mid x)\propto p(\theta)\,P(x\mid\theta)$, producing the *posterior*.
A prior is *conjugate* to a likelihood when the posterior belongs to the very
same family of distributions as the prior, so that observing data merely
updates its parameters.

### The Beta Prior on a Coin

A prior on a coin's bias must be a density supported on $[0,1]$. Viewed as a
function of $\theta$, the likelihood of $x$ heads in $n$ flips is
$\theta^x(1-\theta)^{n-x}$, so a prior with the same algebraic form combines
with it simply by adding exponents. Its normalizing constant uses the Gamma
function $\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}\,dt$ from
:numref:`sec_mdl-distributions`, which generalizes the factorial via
$\Gamma(z+1)=z\,\Gamma(z)$ and $\Gamma(n+1)=n!$. The *Beta* distribution with
parameters $a,b>0$ has the density

$$
\mathrm{Beta}(\theta\mid a,b) = \frac{\theta^{a-1}(1-\theta)^{b-1}}{B(a,b)}, \qquad 0\le\theta\le1,
$$
:eqlabel:`eq_mdl-beta_pdf`

where the normalizer $B(a,b)=\int_0^1\theta^{a-1}(1-\theta)^{b-1}d\theta$
equals $\Gamma(a)\Gamma(b)/\Gamma(a+b)$. The mean follows effortlessly from
this normalizer, since multiplying the density by $\theta$ simply increments
$a$ by one:

$$
\mathbb{E}[\theta]
= \frac{1}{B(a,b)}\int_0^1 \theta^{a}(1-\theta)^{b-1}\,d\theta
= \frac{B(a+1,b)}{B(a,b)}
= \frac{a}{a+b},
$$

where we used $\Gamma(a+1)=a\,\Gamma(a)$ and
$\Gamma(a+b+1)=(a+b)\Gamma(a+b)$ to cancel the Gamma factors. Applying the
same trick twice yields the variance; writing $m=a/(a+b)$ for the mean,

$$
\textrm{Var}(\theta) = \frac{ab}{(a+b)^2(a+b+1)} = \frac{m(1-m)}{a+b+1}.
$$

Larger values of $a+b$ concentrate the density more tightly around its mean,
while $a=b=1$ gives the flat uniform density on $[0,1]$.

Now suppose we observe $x$ heads in $n$ flips. Multiplying the prior by the
likelihood adds the exponents directly,

$$
\underbrace{\theta^{a-1}(1-\theta)^{b-1}}_{\text{prior}}\cdot
\underbrace{\theta^{x}(1-\theta)^{n-x}}_{\text{likelihood}}
= \theta^{(a+x)-1}(1-\theta)^{(b+n-x)-1},
$$

which, once normalized, is again a Beta density with parameters shifted by the
observed counts:

$$
\theta \mid x \;\sim\; \mathrm{Beta}\bigl(a+x,\ b+n-x\bigr).
$$
:eqlabel:`eq_mdl-beta_posterior`

This update has a clean *pseudo-count* interpretation: $a$ and $b$ act as
prior counts of heads and tails, and seeing $x$ heads and $n-x$ tails simply
adds them to the tally. The posterior mean,

$$
\mathbb{E}[\theta\mid x] = \frac{a+x}{a+b+n},
$$

is the fraction of heads after folding in those pseudo-counts. It sits between
the prior mean $a/(a+b)$ and the maximum-likelihood estimate $x/n$, converging
to $x/n$ as $n$ grows large.

Be careful about one subtle distinction: how many flips the prior is worth
depends on which summary you compute. When both posterior parameters exceed
$1$, the posterior mode is $(a+x-1)/(a+b+n-2)$, so the prior contributes
$a-1$ heads and $b-1$ tails to the mode. To keep the convention clear once
and for all: *$a$ and $b$ are the pseudo-counts of the posterior mean, whereas
$a-1$ and $b-1$ are those of the posterior mode*. Under the uniform prior
$\mathrm{Beta}(1,1)$, the mean includes one imagined head and one imagined
tail, while the mode includes none.

Returning to our chapter's coin with the uniform prior, 9 heads in 13 flips
update $\mathrm{Beta}(1,1)$ to $\mathrm{Beta}(10,5)$. Its posterior mean is
$10/15=2/3$, a classic formula known as *Laplace's rule of succession*,
$(x+1)/(n+2)$ :cite:`Laplace.1814`. Its mode $9/13$ coincides with the
maximum-likelihood estimate, and its standard deviation is
$\sqrt{(2/3)(1/3)/16}\approx0.118$.

If we instead see 90 heads in 130 flips, the posterior sharpens to
$\mathrm{Beta}(91,41)$, with mean $0.689$ and standard deviation $0.040$.
Collecting ten times as many flips shrinks the standard deviation by a factor
of $2.9\approx\sqrt{133/16}$: posterior uncertainty narrows at the $1/\sqrt n$
rate, the same scaling we will see for standard errors in
:numref:`sec_mdl-statistics`.

What probability should we assign to the next flip landing heads? Given
$\theta$, that probability is $\theta$; averaging over our posterior beliefs
yields
$P(\text{heads}\mid x)=\int_0^1\theta\,p(\theta\mid x)\,d\theta=\mathbb{E}[\theta\mid x]$.
For our coin, that predictive probability is $2/3$: Laplace's rule directly
gives the probability of the next head.

Conjugacy saved us from computing the normalizing integral in Bayes' rule.
What if our prior is not conjugate? In one dimension, we can simply evaluate
the product of prior and likelihood on a fine grid of $\theta$ values and
normalize the sum. Consider a Gaussian prior on the natural parameter,
$\eta\sim\mathcal N(0,1)$, the same kind of prior that
:numref:`sec_mdl-bayesian-computation` places on logistic-regression weights.
It expresses a belief that the coin is probably near fair, and it is not
conjugate to the Bernoulli likelihood: by a change of variables, its density
in $\theta$ is $\phi(\eta)/\bigl(\theta(1-\theta)\bigr)$, where $\phi$ is the
standard Gaussian density, which is not a Beta density. Let us compute the
posterior under both priors on the same grid and compare their summaries.

```{.python .input #distributions-conjugate}
x, n = 9, 13                                     # 9 heads in 13 flips
theta = np.linspace(0, 1, 100_001)[1:-1]         # interior grid points
eta = np.log(theta) - np.log1p(-theta)           # natural parameter
priors = {'Beta(1, 1)': np.ones_like(theta),     # densities in theta
          'eta ~ N(0, 1)': np.exp(-eta**2 / 2) / (theta * (1 - theta))}
for name, prior in priors.items():
    post = prior * theta**x * (1 - theta)**(n - x)
    post /= post.sum()                           # grid weights sum to one
    mean = post @ theta
    sd = math.sqrt(post @ (theta - mean)**2)
    print(f'{name:13s}: mean {mean:.3f}, sd {sd:.3f}, '
          f'mode {theta[post.argmax()]:.3f}')
```

Under the uniform prior, the grid recovers the exact moments of
$\mathrm{Beta}(10,5)$: mean $0.667$, standard deviation $0.118$, and mode
$9/13\approx0.692$. Placing a Gaussian prior on the log-odds pulls the
posterior closer to a fair coin, lowering the mean to $0.644$, the mode to
$0.665$, and the standard deviation slightly to $0.111$. A grid handles the
non-conjugate prior just as easily as the conjugate one, but only because
$\theta$ is scalar; :numref:`sec_mdl-bayesian-computation` develops methods
that scale to high-dimensional parameter vectors.
:numref:`fig_mdl-prob-beta-posterior` visualizes how the Beta posterior
sharpens as flips accumulate.

![Bayesian updating of a coin's bias $\theta$. Left: the flat $\mathrm{Beta}(1,1)$ prior. Middle: the $\mathrm{Beta}(10,5)$ posterior after 9 heads in 13 flips of a coin with true bias $\theta^\ast=0.7$ (dashed), with mean $2/3$ and standard deviation $0.12$. Right: the $\mathrm{Beta}(91,41)$ posterior after 130 flips, with standard deviation $0.04$. Each update adds observed heads and tails to the pseudo-counts, and the posterior width shrinks like $1/\sqrt{n}$.](../img/mdl-prob-beta-posterior.svg)
:label:`fig_mdl-prob-beta-posterior`

### Conjugacy for Every Exponential Family

At first glance, the Beta update looks like a lucky algebraic coincidence. In
fact, it is a universal property of exponential families: every exponential
family admits a conjugate prior, and updating it always amounts to adding the
sufficient statistics.

**Proposition (conjugate prior).** *Consider a likelihood
:eqref:`eq_mdl-exp_family` and the prior on its natural parameters*

$$
p(\boldsymbol\eta) \propto \exp\bigl(\boldsymbol\nu^\top\boldsymbol\eta - \kappa\,A(\boldsymbol\eta)\bigr),
$$
:eqlabel:`eq_mdl-exp_family_conjugate`

*where $\boldsymbol\nu\in\mathbb R^d$ and $\kappa\in\mathbb R$ are chosen so
that the density is normalizable. After $n$ independent observations
$\mathbf x_1,\dots,\mathbf x_n$, the posterior retains the exact same form,
with $\boldsymbol\nu$ updated to $\boldsymbol\nu+\sum_i T(\mathbf x_i)$ and
$\kappa$ updated to $\kappa+n$.*

**Proof.** Dropping factors that do not depend on $\boldsymbol\eta$, the
likelihood contributes
$\exp\bigl(\boldsymbol\eta^\top\sum_i T(\mathbf x_i)-nA(\boldsymbol\eta)\bigr)$.
Multiplying it by the prior combines the exponents in a single line:

$$
e^{\boldsymbol\nu^\top\boldsymbol\eta-\kappa A(\boldsymbol\eta)}\cdot
e^{\boldsymbol\eta^\top\sum_iT(\mathbf x_i)-nA(\boldsymbol\eta)}
= e^{(\boldsymbol\nu+\sum_iT(\mathbf x_i))^\top\boldsymbol\eta-(\kappa+n)A(\boldsymbol\eta)} .
$$

Normalizing this product gives the posterior, which therefore belongs to the
same family with the updated parameters $\boldsymbol\nu+\sum_i T(\mathbf x_i)$
and $\kappa+n$. $\blacksquare$

The prior's parameters $\boldsymbol\nu$ and $\kappa$, called
*hyperparameters*, behave like *pseudo-data*: $\boldsymbol\nu$ acts as a prior
sum of sufficient statistics and $\kappa$ as a prior sample size
:cite:`Bishop.2006`. For the Bernoulli coin, this prior is written in terms of
the logit $\eta=\log\frac{\theta}{1-\theta}$; changing variables from $\eta$
to $\theta$ transforms it into $\mathrm{Beta}(\nu,\kappa-\nu)$, and the
general update $\nu\to\nu+x$, $\kappa\to\kappa+n$ reproduces the Beta update
$a\to a+x$, $b\to b+n-x$. Exercise
:exref:`ex_mdl-distributions-conjugate-recipe` works through this change of
variables, including its Jacobian.

### Gamma–Poisson and Dirichlet–Multinomial
:label:`subsec_mdl-other-conjugate-pairs`

Two more widely used conjugate pairs follow the same pattern, one for event
rates and one for categorical probabilities over words or classes. Viewed as a
function of the rate $\lambda$, the likelihood of $n$ Poisson counts
$x_1,\dots,x_n$ is proportional to $\lambda^{\sum_i x_i}e^{-n\lambda}$. The
matching conjugate prior is the *Gamma* distribution with shape $a>0$ and rate
$b>0$, whose density is
$\mathrm{Gamma}(\lambda\mid a,b)=\frac{b^a}{\Gamma(a)}\lambda^{a-1}e^{-b\lambda}$
for $\lambda>0$ (with mean $a/b$). Multiplying prior and likelihood yields the
posterior $\mathrm{Gamma}\bigl(a+\sum_i x_i,\ b+n\bigr)$, so the prior
encodes $a$ imagined events spread across $b$ imagined time periods. Exercise
:exref:`ex_mdl-distributions-conjugate-recipe` derives this pair directly from
the general proposition.

To generalize the Beta prior from two outcomes to $K$ categories, we use the
*Dirichlet* distribution, a density over probability vectors $\mathbf p$ with
$p_k\ge0$ and $\sum_k p_k=1$. With concentration parameters $a_k>0$, its
density is

$$
\mathrm{Dir}(\mathbf p\mid\mathbf a)=\frac{\Gamma(\sum_ka_k)}{\prod_k\Gamma(a_k)}\prod_{k=1}^K p_k^{a_k-1},
$$

which has mean $\mathbb{E}[p_k]=a_k/\sum_j a_j$ and reduces to the Beta
distribution when $K=2$. Observing category counts $x_1,\dots,x_K$ gives a
likelihood proportional to $\prod_k p_k^{x_k}$, so the posterior is
$\mathrm{Dir}(\mathbf a+\mathbf x)$, adding one pseudo-count per category.

To see why this matters in practice, suppose a three-word vocabulary produces
the counts $(3,0,1)$. The maximum-likelihood estimate $(0.75,0,0.25)$ assigns
probability zero to the unseen second word, which would make any future
document containing that word seem impossible. Placing a uniform prior
$\mathrm{Dir}(1,1,1)$ on $\mathbf p$ yields the posterior
$\mathrm{Dir}(4,1,2)$, whose mean is
$(4/7,1/7,2/7)\approx(0.571,0.143,0.286)$. Adding one to every count before
normalizing, known as *add-one smoothing*, is simply the posterior mean under
this Dirichlet prior; the Naive Bayes classifier in
:numref:`sec_mdl-naive_bayes` uses the two-class case.

Closed-form posteriors are the exception rather than the rule: outside
conjugate pairs, Bayesian inference requires numerical approximation
(:numref:`sec_mdl-bayesian-computation`). Before tackling that, however, we
need an answer to a more basic question about point estimates. From 13 flips
we obtained $9/13$, a single number without an error bar: how far might $9/13$
be from the true bias, and does a coin with 11 heads in 13 flips really differ
from one with 9? :numref:`sec_mdl-statistics` takes up these questions.

## Summary

Nearly every distribution in our catalogue shares the exponential-family form
$h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$,
where data and parameters interact solely through the inner product
$\boldsymbol\eta^\top T(\mathbf x)$, and maximizing entropy under fixed
average statistics produces this exact form. Differentiating the
log-partition function $A$ generates moments: $\nabla A$ is the mean of $T$
and $\nabla^2 A$ is its covariance, which immediately proves that $A$ is
convex. Consequently, maximum likelihood is a convex optimization problem
that matches the model's expected sufficient statistics to their sample
averages, admitting a finite solution whenever that sample average is
attainable. Pairing an exponential family with its canonical link yields a
generalized linear model whose gradient is always input times prediction minus
observation, unifying linear, logistic, softmax, and Poisson regression while
explaining why non-canonical links such as the probit behave differently. The
softmax emerges as the mean map of the categorical family (as well as from
Gumbel noise and the Boltzmann distribution). Finally, every exponential
family admits a conjugate prior updated simply by adding sufficient
statistics; for a coin, $\mathrm{Beta}(a,b)$ contributes $a$ and $b$
pseudo-counts to the posterior mean and $a-1$ and $b-1$ to the mode.

## Exercises

1. **Exponential races.** Let $X_1,\dots,X_m$ be independent with
   $X_i\sim\mathrm{Exp}(\lambda_i)$, for example the waiting times for the next
   event in $m$ independent event streams.
    1. Show that $\min_i X_i\sim\mathrm{Exp}\bigl(\sum_i\lambda_i\bigr)$.
    1. Show that the $j$-th stream fires first with probability
       $P\bigl(X_j=\min_i X_i\bigr)=\lambda_j/\sum_i\lambda_i$.
    1. Let $V_1,\dots,V_K$ be independent $\mathrm{Exp}(1)$ variables and let
       $\mathbf z\in\mathbb R^K$ be a vector of logits. Use (b) to show that
       $\operatorname{argmax}_k\,(z_k-\log V_k)$ is distributed according to
       the softmax probabilities :eqref:`eq_mdl-exp_family_softmax`. Because
       $-\log V_k$ has the standard Gumbel distribution, this is a second
       proof that Gumbel noise gives the softmax.
   :exlabel:`ex_mdl-distributions-exponential-races`

1. **Maximum entropy on the integers.** Consider distributions
   $(p_0,p_1,p_2,\dots)$ on $\{0,1,2,\dots\}$ with a fixed mean $m>0$.
    1. Maximize the Shannon entropy $H[p]=-\sum_k p_k\log p_k$ subject to
       $\sum_k p_k=1$ and $\sum_k k\,p_k=m$, using Lagrange multipliers
       (:numref:`subsec_mdl-lagrange-multipliers`). Show that the stationary
       point is a geometric distribution $p_k=(1-r)\,r^k$, and express $r$
       through $m$.
    1. Repeat the derivation for the entropy relative to $h(k)=1/k!$,
       $H_h[p]=-\sum_k p_k\log\bigl(p_k/h(k)\bigr)$, and show that the
       stationary point is $\mathrm{Poisson}(m)$.
    1. At $m=4$, compute the Shannon entropy of both distributions
       numerically. Which one is larger, and why must it be? Compare
       $P(X\ge15)$ under the two distributions. Which assumption about large
       counts does the choice of base measure encode?
    1. A stationary point of a Lagrangian need not be a maximum. Use the
       maximum-entropy proposition to prove that the two distributions of
       (a) and (b) are the global maximizers, by exhibiting each as a member
       of an exponential family whose mean equals $m$.
   :exlabel:`ex_mdl-distributions-max-entropy-integers`

1. **Moments from the log-partition.** The moment property
   :eqref:`eq_mdl-exp_family_moment` and its second-order form
   $\nabla^2A=\textrm{Cov}(T)$ turn moment calculations into
   differentiation.
    1. For the Poisson, $A(\eta)=e^\eta$. Derive the mean and the variance
       from $A$. Which property of the function $A$ makes them equal?
    1. For the Gaussian, the natural parameters are $\eta_1=\mu/\sigma^2$
       and $\eta_2=-1/(2\sigma^2)$. Show that
       $A=-\eta_1^2/(4\eta_2)-\frac12\log(-2\eta_2)$, and differentiate it to
       recover $\mathbb{E}[x]$, $\mathbb{E}[x^2]$, $\textrm{Var}(x)$ and
       $\textrm{Cov}(x,x^2)$.
    1. The Laplace distribution with a fixed location $\mu$ and scale $b$ has
       the sufficient statistic $T(x)=|x-\mu|$. Write it in the form
       :eqref:`eq_mdl-exp_family`, and use $A$ to compute
       $\mathbb{E}\,|x-\mu|$ and $\textrm{Var}|x-\mu|$. Check the result
       against the Laplace variance $2b^2$.
    1. For $n$ Gaussian observations, write the two moment-matching
       equations $\nabla A(\hat{\boldsymbol\eta})=\bar T$ and solve them for
       $\hat\mu$ and $\hat\sigma^2$. What happens for $n=1$? Relate your
       answer to the condition that $\bar T$ be an attainable mean.
   :exlabel:`ex_mdl-distributions-log-partition-moments`

1. [code] **Probit and logit.** For two classes with the score difference
   $m=z_1-z_2$, the logit model gives $P(y=1)=\operatorname{sigmoid}(m)$ and
   the probit model with noise scale $s$ gives
   $P(y=1)=\Phi\bigl(m/(\sqrt2\,s)\bigr)$.
    1. Show that the difference of two independent standard Gumbel variables
       has the CDF $\operatorname{sigmoid}(x)$. The standard Gumbel
       distribution has variance $\pi^2/6$. Find the scale $c$ for which the
       distribution with CDF $\operatorname{sigmoid}(cx)$ has variance 1,
       like the standard Gaussian.
    1. Find numerically the scale $c$ that minimizes
       $\max_x|\operatorname{sigmoid}(cx)-\Phi(x)|$, and the value of this
       maximum. Why is this scale smaller than the one of (a)?
    1. An example of class 2 lies on the wrong side of the decision
       boundary, with $m>0$. Show that its negative log-likelihood
       $-\log\operatorname{sigmoid}(-m)$ under the logit model lies between
       $m$ and $m+e^{-m}$, and that its derivative with respect to $m$ is
       less than 1. Use the bounds
       $\frac{x}{1+x^2}\,\phi(x)\le1-\Phi(x)\le\frac{\phi(x)}{x}$ for $x>0$ to
       show that its negative log-likelihood
       $-\log\Phi\bigl(-m/(\sqrt2\,s)\bigr)$ under the probit model grows like
       $m^2/(4s^2)$, and that its derivative grows linearly in $m$.
    1. Draw 100 inputs $x\sim\mathcal N(1,1)$ of class 1 and 100 inputs
       $x\sim\mathcal N(-1,1)$ of class 2, and let $m=wx+b$. Fit $(w,b)$ by
       maximum likelihood under both models. Then add three examples at
       $x=6$ labeled as class 2, and fit again. For each model, compute the
       ratio of the slope $w$ with these mislabeled examples to the slope
       without them, and explain the difference between the models with (c).
       Compute $\log\Phi$ with a numerically stable routine such as
       `scipy.special.log_ndtr`.
   :exlabel:`ex_mdl-distributions-probit-logit`

1. **Minimal categorical family.** ● Code a categorical outcome as a one-hot
   vector $\mathbf x\in\{0,1\}^K$ with class probabilities
   $\mathbf p=\mathrm{softmax}(\mathbf z)$, written as an exponential family
   with $T(\mathbf x)=\mathbf x$, natural parameters $\mathbf z$ and
   $A(\mathbf z)=\log\sum_ke^{z_k}$.
    1. Compute $\nabla^2 A(\mathbf z)$ and show that it equals the
       covariance $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$
       of the one-hot vector $\mathbf x$. Every one-hot vector satisfies
       $\mathbf x\mathbf x^\top=\operatorname{diag}(\mathbf x)$. Show that
       $\nabla^2A(\mathbf z)\,\mathbf 1=\mathbf 0$, and that the gradient of
       the negative log-likelihood of a dataset is orthogonal to $\mathbf 1$
       at every $\mathbf z$. Gradient descent with a small step size runs on
       the logits from a starting point $\mathbf z_0$, and every class
       occurs in the data. Which function of $\mathbf z$ does gradient
       descent never change? Which of the minimizers does it therefore
       approach?
    1. Fix $z_K=0$. This gives a minimal representation with
       $\boldsymbol\eta=(z_1,\dots,z_{K-1})$ and
       $T(\mathbf x)=(x_1,\dots,x_{K-1})$. Show that $\nabla^2A(\boldsymbol\eta)$
       is positive definite whenever every $p_k>0$. Given class counts
       $n_1,\dots,n_K$, solve the moment-matching condition
       $\nabla A(\hat{\boldsymbol\eta})=\bar T$ for the maximum-likelihood
       $\hat{\boldsymbol\eta}$. What happens when some class has $n_k=0$?
   :exlabel:`ex_mdl-distributions-minimal-categorical`

1. [code] **Poisson regression.** ● A generalized linear model for counts
   makes the natural parameter of a Poisson an affine function of the
   features: $y_i\mid\mathbf x_i\sim\mathrm{Poisson}(e^{\eta_i})$ with
   $\eta_i=\mathbf w^\top\mathbf x_i+b$.
    1. Use the negative log-likelihood :eqref:`eq_mdl-exp_family_nll`
       (equivalently, the empirical log-likelihood
       $\ell(\boldsymbol\eta) = \sum_{i=1}^n \bigl(\boldsymbol\eta^\top T(y_i) - A(\boldsymbol\eta)\bigr)$
       up to terms independent of the parameters) to
       write the loss as a function of $(\mathbf w,b)$, and compute its
       gradient and Hessian. Show that the loss is convex. Which equations
       does a zero gradient impose on the fitted means
       $\hat m_i=e^{\hat\eta_i}$?
    1. Generate $n=1000$ examples with $\mathbf x_i\sim\mathcal N(\mathbf 0,\mathbf I_2)$,
       $\mathbf w=(-0.3,1.0)$ and $b=0.5$. Starting from zero, fit
       $\mathbf u=(\mathbf w,b)$ with Newton's method, which repeats
       the step $\mathbf u\leftarrow\mathbf u-\bigl(\nabla^2L(\mathbf u)\bigr)^{-1}\nabla L(\mathbf u)$
       for the loss $L$, and check the equations of (a) at the solution.
    1. Multiply each rate by an independent $g_i\sim\mathrm{Gamma}(1,1)$,
       which has mean $1$, so that the counts become overdispersed while their
       conditional means stay $e^{\eta_i}$. Fit the Poisson model to $200$
       such datasets and to $200$ datasets without the multiplier. Compare
       the mean and the standard deviation of the estimates across datasets.
       Explain the behavior of the mean with the equations of (a).
   :exlabel:`ex_mdl-distributions-poisson-regression`

1. **Conjugate priors from natural parameters.** ● For a natural
   parameter $\boldsymbol\eta$, the conjugate prior
   :eqref:`eq_mdl-exp_family_conjugate` reads
   $p(\boldsymbol\eta)\propto\exp\bigl(\boldsymbol\nu_0^\top\boldsymbol\eta-n_0\,A(\boldsymbol\eta)\bigr)$,
   and $n$ observations $x_{1:n}$ produce the posterior density
   $p(\boldsymbol\eta \mid x_{1:n}) \propto \exp\bigl(\boldsymbol\eta^\top\bigl(\boldsymbol\nu_0 + \sum_{i=1}^n T(x_i)\bigr) - (n_0 + n)A(\boldsymbol\eta)\bigr)$
   (written in the scalar case with $\nu = \nu_0$ and $\kappa = n_0$, updated by
   $\nu\to\nu+\sum_i T(x_i)$ and $\kappa\to\kappa+n$).
    1. For the Poisson, $\eta=\log\lambda$ and $A(\eta)=e^\eta$. Change
       variables from $\eta$ to $\lambda$, including the Jacobian, and show
       that the prior on $\lambda$ is a Gamma distribution. Identify its shape
       and rate, and check that the update reproduces the Gamma posterior
       of :numref:`subsec_mdl-other-conjugate-pairs`.
    1. For the Bernoulli, $\eta=\log\frac{\theta}{1-\theta}$ and
       $A(\eta)=\log(1+e^\eta)$. Show that the prior on $\theta$ is
       $\mathrm{Beta}(\nu,\kappa-\nu)$, and find the values of $(\nu,\kappa)$
       for which it is normalizable.
    1. Take the flat prior on the logit, $\nu=\kappa=0$. Which density on
       $\theta$ does it induce, and is that density normalizable? After $x$
       heads in $n$ flips, when is the posterior normalizable, and what is
       its mean? Compare it with the mean of the posterior
       :eqref:`eq_mdl-beta_posterior` under the uniform prior
       $\mathrm{Beta}(1,1)$ on $\theta$. What does the comparison say about
       the phrase "a flat prior"?
   :exlabel:`ex_mdl-distributions-conjugate-recipe`

[Discussions](https://d2l.discourse.group/t/36862)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §36.4]{.kicker}

The Exponential Family: Sufficient Statistics, Cumulants, Maximum Entropy, and Conjugacy<br>**canonical form $h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ · maximum entropy · $\nabla A = \mathbb{E}[T]$ & $\nabla^2 A = \mathrm{Cov}(T)$ · GLMs · softmax origins · conjugate priors**
:::
:::

::: {.slide title="Four Regressions, Four Losses, One Gradient"}
[Motivation]{.kicker}

Linear, logistic, softmax, and Poisson regression minimize four distinct loss functions, yet differentiating each loss with respect to its linear weights $\mathbf{W}$ yields the exact same training rule:

$$\nabla_{\mathbf{W}} \mathcal{L} = \mathbf{x}\bigl(\text{prediction} - \text{observation}\bigr)^\top$$

::: {.cols}
::: {.col}
- **Why predict log-odds instead of probability?** For our coin with $9$ heads in $13$ flips ($\hat{p} = 9/13 \approx 0.692$), a classifier's linear layer outputs the log-odds $\hat{\eta} = \log(9/4) \approx 0.811$ and applies $\mathrm{sigmoid}(\hat{\eta}) = 9/13$.
- **Why did every distribution in §36.3 need a separate moment trick?** In canonical form, every mean and variance is simply the first and second derivative of one function $A(\boldsymbol{\eta})$.
:::

::: {.col}
- **Four consequences of one canonical template:**
  1. **Cumulant generation:** $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(\mathbf{x})]$ and $\nabla^2 A(\boldsymbol{\eta}) = \mathrm{Cov}(T(\mathbf{x})) \succeq \mathbf{0}$.
  2. **Convex MLE & moment matching:** $\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T}$.
  3. **Maximum entropy:** Least-committal distribution with given mean statistics $\mathbb{E}[T]$.
  4. **Universal conjugacy:** Priors update by adding sufficient statistics.
:::
:::
:::

::: {.slide title="Rewriting the Bernoulli Into Exponential Form"}
[How Log-Odds and Softplus Emerge]{.kicker}

Take the logarithm of the Bernoulli PMF $P(X = x) = p^x(1 - p)^{1 - x}$ for $x \in \{0, 1\}$ and group terms multiplying $x$:

$$\log P(X = x) = x\log p + (1 - x)\log(1 - p) = x\underbrace{\log\frac{p}{1 - p}}_{\eta} + \underbrace{\log(1 - p)}_{-A(\eta)}$$

- **Natural parameter (log-odds / logit):** $\eta = \log\frac{p}{1 - p} \in \mathbb{R} \iff p = \frac{e^\eta}{1 + e^\eta} = \mathrm{sigmoid}(\eta) = \frac{1}{1 + e^{-\eta}}$.
- **Log-partition function (softplus):** Since $1 - p = \frac{1}{1 + e^\eta}$, the normalizing constant is $A(\eta) = -\log(1 - p) = \log(1 + e^\eta)$, giving
  $$P(X = x) = \exp\bigl(\eta x - \log(1 + e^\eta)\bigr), \qquad x \in \{0, 1\}$$
- **Nobody had to invent the sigmoid:** The logistic sigmoid $\mathrm{sigmoid}(\eta)$ is simply the inverse map from the Bernoulli's natural parameter $\eta$ back to its mean $p = \mathbb{E}[X]$ (for $9/13$ heads, $\hat{\eta} = \log(9/4) \approx 0.811$).
:::

::: {.slide title="The Canonical Exponential Family"}
[General Definition & Sufficient Statistics]{.kicker}

A family of distributions belongs to the **exponential family** if its PMF or density takes the canonical form:

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\,\exp\bigl(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})\bigr), \qquad A(\boldsymbol{\eta}) = \log \int h(\mathbf{x})\,\exp\bigl(\boldsymbol{\eta}^\top T(\mathbf{x})\bigr)\,d\mathbf{x}$$

- **Four structural components:**
  1. **Base measure $h(\mathbf{x}) \ge 0$:** Parameter-free baseline weight defining the fixed support $\{\mathbf{x} : h(\mathbf{x}) > 0\}$.
  2. **Sufficient statistics $T(\mathbf{x}) \in \mathbb{R}^d$:** Data features that interact with the parameters.
  3. **Natural parameters $\boldsymbol{\eta} \in \mathbb{R}^d$:** Coordinates multiplying $T(\mathbf{x})$ inside the exponent.
  4. **Log-partition (cumulant) function $A(\boldsymbol{\eta})$:** Logarithm of the normalizer ensuring $\int p(\mathbf{x} \mid \boldsymbol{\eta})\,d\mathbf{x} = 1$.
- **Why $T(\mathbf{x})$ is called *sufficient*:** For $n$ i.i.d. observations, $\prod_{i=1}^n p(\mathbf{x}_i \mid \boldsymbol{\eta}) = \left(\prod_i h(\mathbf{x}_i)\right)\exp\!\left(\boldsymbol{\eta}^\top \sum_{i=1}^n T(\mathbf{x}_i) - nA(\boldsymbol{\eta})\right)$, so the parameters touch the dataset **only through the sum $\sum_{i=1}^n T(\mathbf{x}_i)$**.
:::

::: {.slide title="Catalog of Exponential Family Members"}
[Unifying Discrete and Continuous Distributions]{.kicker}

Every row below writes $p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$:

| Distribution | Base Measure $h(\mathbf{x})$ | Sufficient Statistic $T(\mathbf{x})$ | Natural Parameter $\boldsymbol{\eta}$ | Log-Partition $A(\boldsymbol{\eta})$ |
| :--- | :--- | :--- | :--- | :--- |
| **$\mathrm{Bernoulli}(p)$** | $1$ | $x \in \{0, 1\}$ | $\log\frac{p}{1-p}$ | $\log(1 + e^\eta)$ |
| **$\mathrm{Binomial}(n, p)$** ($n$ fixed) | $\binom{n}{x}$ | $x \in \{0, \dots, n\}$ | $\log\frac{p}{1-p}$ | $n\log(1 + e^\eta)$ |
| **$\mathrm{Poisson}(\lambda)$** | $1/k!$ | $k \in \{0, 1, \dots\}$ | $\log \lambda$ | $e^\eta$ |
| **$\mathrm{Categorical}(\mathbf{p})$** | $1$ | One-hot $\mathbf{y} \in \{0, 1\}^K$ | Logits $\mathbf{z} \in \mathbb{R}^K$ | $\log \sum_{k=1}^K e^{z_k}$ |
| **$\mathrm{Exp}(\lambda)$** | $\mathbf{1}_{x \ge 0}$ | $x$ | $-\lambda < 0$ | $-\log(-\eta)$ |
| **$\mathcal{N}(\mu, \sigma^2)$** | $1/\sqrt{2\pi}$ | $(x,\ x^2)^\top$ | $\left(\frac{\mu}{\sigma^2},\ -\frac{1}{2\sigma^2}\right)^\top$ | $-\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2)$ |
| **$\mathrm{Laplace}(\mu, b)$** ($\mu$ fixed) | $1$ | $\|x - \mu\|$ | $-1/b < 0$ | $\log(-2/\eta)$ |
:::

::: {.slide title="What Lies Outside the Exponential Family?"}
[Moving Support & Heavy-Tailed Non-Convexity]{.kicker}

Two classic distributions fall outside the exponential family for reasons that drastically alter parameter estimation:

::: {.cols}
::: {.col}
**1. Moving Support: $\mathrm{Unif}(0, \theta)$**

- Density $p(x \mid \theta) = \frac{1}{\theta}\mathbf{1}_{0 \le x \le \theta}$ has support $[0, \theta]$ that depends on $\theta$ (whereas $h(\mathbf{x}) > 0$ must be parameter-free).
- For $n$ draws, $\prod_{i=1}^n p(x_i \mid \theta) = \theta^{-n}\mathbf{1}_{\max_i x_i \le \theta}$, so the MLE is the sample maximum $\hat{\theta}_{\mathrm{MLE}} = \max_i x_i$.
- **Moment-matching failure:** On data $\{0.1, 0.2, 0.9\}$, $\hat{\theta}_{\mathrm{MLE}} = 0.9$, whereas matching the mean $\theta/2 = \bar{x} = 0.4$ gives $\hat{\theta} = 0.8 < 0.9$ (making the observed $0.9$ impossible!).
:::

::: {.col}
**2. Algebraic Heavy Tails: Student's $t_\nu(\mu)$**

- Product likelihood $\prod_{i=1}^n \left(1 + \frac{(x_i - \mu)^2}{\nu}\right)^{-(\nu+1)/2}$ cannot be compressed into any fixed-dimension sum $\sum_i T(x_i)$ (*Pitman–Koopman–Darmois theorem*).
- **Loss of convexity:** Because Student-$t$ is not an exponential family, its negative log-likelihood for location $\mu$ is **non-convex** and develops multiple local minima when data cluster into separated groups.
:::
:::
:::

::: {.slide title="Jaynes's Loaded Die and Maximum Entropy"}
[Where the Exponential Form Comes From]{.kicker}

::: {.cols .vc}
::: {.col}
- **Shannon entropy** measures average surprise (in nats):
  $$H[p] = -\sum_{k=1}^K p_k \log p_k \quad (\text{fair die: } H = \log 6 \approx 1.792)$$
- **Jaynes's puzzle:** A 6-sided die has average roll $\mathbb{E}[k] = 4.5$ instead of $3.5$. Putting mass $1/2$ on $\{3, 6\}$ hits mean $4.5$, but assumes faces $\{1, 2, 4, 5\}$ are impossible ($H = \log 2 \approx 0.693$).
- **Principle of Maximum Entropy:** The least-committal distribution matching $\mathbb{E}[k] = 4.5$ is the **exponential tilt**:
  $$p_k = \frac{e^{\eta k}}{\sum_{j=1}^6 e^{\eta j}}, \quad \eta \approx 0.371 \quad (H \approx 1.614)$$
:::

::: {.col .fig}
@fig:mdl-prob-maxent-die
:::
:::
:::

::: {.slide title="Finding the Maximum-Entropy Tilt by Bisection"}
[Monotonicity of the Mean Statistic]{.kicker}

Because the expected roll $\sum_{k=1}^6 k\,p_k(\eta)$ increases strictly with $\eta$ (its derivative is the variance $\mathrm{Var}_\eta(k) > 0$), bisection locates $\eta \approx 0.371$ in 50 steps:

@mdl-exponential-family-entropy-and-a-loaded-die

::: {.d2l-note}
**Minimal entropy sacrifice:** The exponential tilt surrenders only $\log 6 - 1.614 \approx 0.18\text{ nats}$ of entropy to shift the mean from $3.5$ to $4.5$, compared with $1.10\text{ nats}$ lost by the two-point distribution on $\{3, 6\}$.
:::
:::

::: {.slide title="Gibbs' Inequality: KL Divergence Is Non-Negative"}
[One Tangent Line Proves Global Optimality]{.kicker}

To prove that exponential-family distributions maximize entropy without Lagrange multipliers, we compare any candidate $q$ against $p$ via the **Kullback–Leibler (KL) divergence**:

$$D_{\mathrm{KL}}(q \| p) = \sum_{\mathbf{x}:\,q(\mathbf{x}) > 0} q(\mathbf{x})\log\frac{q(\mathbf{x})}{p(\mathbf{x})} \ge 0 \qquad (\text{with equality iff } q = p)$$

- **Complete proof from $\log u \le u - 1$ (concavity of $\log$ with equality only at $u = 1$):**
  Set $u = p(\mathbf{x})/q(\mathbf{x})$ and sum over all outcomes with $q(\mathbf{x}) > 0$:
  $$-D_{\mathrm{KL}}(q \| p) = \sum_{q > 0} q(\mathbf{x})\log\frac{p(\mathbf{x})}{q(\mathbf{x})} \le \sum_{q > 0} q(\mathbf{x})\left(\frac{p(\mathbf{x})}{q(\mathbf{x})} - 1\right) = \sum_{q > 0} p(\mathbf{x}) - 1 \le 0$$
- Equality holds if and only if $p(\mathbf{x})/q(\mathbf{x}) = 1$ wherever $q(\mathbf{x}) > 0$, which forces $\sum_{q>0} p(\mathbf{x}) = 1$ and $p = q$ everywhere. $\blacksquare$
:::

::: {.slide title="The Maximum-Entropy Theorem and Its Corollaries"}
[Why Every Moment-Constrained MaxEnt Distribution Is Exponential Family]{.kicker}

Define entropy relative to base measure $h(\mathbf{x})$ as $H_h[q] = -\sum_{\mathbf{x}} q(\mathbf{x})\log\frac{q(\mathbf{x})}{h(\mathbf{x})}$.

**Proposition (Maximum Entropy):** Let $p(\mathbf{x}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$, and let $q$ be *any* distribution matching its mean statistics, $\mathbb{E}_q[T(\mathbf{x})] = \mathbb{E}_p[T(\mathbf{x})]$. Then:

$$H_h[q] = H_h[p] - D_{\mathrm{KL}}(q \| p) \le H_h[p] \qquad (\text{with equality iff } q = p)$$

- **Two-line proof:** Split $\log\frac{q}{h} = \log\frac{q}{p} + \log\frac{p}{h}$. The first sum is $-D_{\mathrm{KL}}(q \| p) \le 0$. Because $\log\frac{p}{h} = \boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})$ is linear in $T$, the second sum depends on $q$ only through its constrained mean: $-\boldsymbol{\eta}^\top\mathbb{E}_q[T] + A(\boldsymbol{\eta}) = -\boldsymbol{\eta}^\top\mathbb{E}_p[T] + A(\boldsymbol{\eta}) = H_h[p]$. $\blacksquare$
- **Famous corollaries:**
  - **No constraint ($T \equiv 0$, $h \equiv 1$ on $K$ outcomes):** Uniform distribution ($H = \log K$).
  - **Fixed mean $\mu$ and variance $\sigma^2$ on $\mathbb{R}$ ($T(x) = (x, x^2)$, $h \equiv 1$):** Gaussian $\mathcal{N}(\mu, \sigma^2)$.
  - **Fixed mean $m$ on $\{0, 1, 2, \dots\}$ ($T(k) = k$):** $\mathrm{Poisson}(m)$ when $h(k) = 1/k!$, or $\mathrm{Geometric}$ when $h(k) = 1$.
:::

::: {.slide title="The Moment Property: Derivatives of A(η)"}
[Cumulant Generation & Convexity]{.kicker}

Write $Z(\boldsymbol{\eta}) = e^{A(\boldsymbol{\eta})} = \int h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x})}\,d\mathbf{x}$. Differentiating $A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$ under the integral sign generates all moments of $T(\mathbf{x})$:

$$\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \frac{\nabla Z(\boldsymbol{\eta})}{Z(\boldsymbol{\eta})} = \int T(\mathbf{x})\,\underbrace{h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})}}_{p(\mathbf{x} \mid \boldsymbol{\eta})}\,d\mathbf{x} = \mathbb{E}\bigl[T(\mathbf{x})\bigr]$$

- **Second derivative (Hessian) is the covariance matrix of $T(\mathbf{x})$:**
  $$\nabla_{\boldsymbol{\eta}}^2 A(\boldsymbol{\eta}) = \frac{\nabla^2 Z}{Z} - \frac{(\nabla Z)(\nabla Z)^\top}{Z^2} = \mathbb{E}\bigl[T(\mathbf{x})T(\mathbf{x})^\top\bigr] - \mathbb{E}[T(\mathbf{x})]\,\mathbb{E}[T(\mathbf{x})]^\top = \mathrm{Cov}\bigl(T(\mathbf{x})\bigr) \succeq \mathbf{0}$$
- **Universal convexity:** Because every covariance matrix is positive semidefinite ($\mathbf{v}^\top \mathrm{Cov}(T)\mathbf{v} = \mathrm{Var}(\mathbf{v}^\top T) \ge 0$), the log-partition function $A(\boldsymbol{\eta})$ is **always convex**!
:::

::: {.slide title="Computing Moments via Automatic Differentiation"}
[Analytical & Autograd Verification]{.kicker}

- **Bernoulli ($A(\eta) = \log(1 + e^\eta)$):** $A'(\eta) = \mathrm{sigmoid}(\eta) = p$ and $A''(\eta) = \mathrm{sigmoid}(\eta)(1 - \mathrm{sigmoid}(\eta)) = p(1 - p)$.
- **Poisson ($A(\eta) = e^\eta$ with $\eta = \log\lambda$):** Since $e^\eta$ is its own derivative, $A'(\eta) = A''(\eta) = e^\eta = \lambda$, explaining in one line why the Poisson mean equals its variance!

@distributions-exp-family
:::

::: {.slide title="Maximum Likelihood as Convex Moment Matching"}
[Global Optimization of the Log-Likelihood]{.kicker}

::: {.cols .vc}
::: {.col}
- For $n$ i.i.d. observations $\mathbf{x}_1, \dots, \mathbf{x}_n$, the negative log-likelihood (NLL) is convex in $\boldsymbol{\eta}$:
  $$\mathcal{L}(\boldsymbol{\eta}) = n\,A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top \sum_{i=1}^n T(\mathbf{x}_i) - \sum_{i=1}^n \log h(\mathbf{x}_i)$$
- **Moment-matching optimality condition ($\nabla_{\boldsymbol{\eta}} \mathcal{L} = \mathbf{0}$):**
  $$\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T} = \frac{1}{n}\sum_{i=1}^n T(\mathbf{x}_i)$$
- **Coin ($9/13$ heads):** $\mathrm{sigmoid}(\hat{\eta}) = \frac{9}{13} \implies \hat{\eta} = \log\frac{9}{4} \approx 0.811$. At $\hat{\eta}$, average NLL equals the fitted coin's entropy ($0.617\text{ nats/flip}$).
:::

::: {.col .fig}
@fig:mdl-prob-log-partition
:::
:::
:::

::: {.slide title="Boundary Divergence and Redundant Logits"}
[When Moment Matching Has No Finite or Unique Solution]{.kicker}

::: {.cols}
::: {.col}
**1. Unattainable Boundary Means ($13/13$ Heads)**

- The equation $\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T}$ has a finite solution only when $\bar{T}$ lies in the interior of attainable expectations.
- For a coin, $\mathrm{sigmoid}(\eta) \in (0, 1)$ is strictly open. If all $13$ flips land heads, $\bar{T} = 1 \notin (0, 1)$, and
  $$\mathcal{L}(\eta) = 13\log(1 + e^{-\eta}) \xrightarrow{\eta \to \infty} 0$$
  decreases monotonically without a finite minimizer ($\hat{\eta} \to +\infty$).
- **Linearly separable logistic regression:** Hits this exact runaway regime ($\|\mathbf{w}\| \to \infty$) unless regularized by weight decay or a prior!
:::

::: {.col}
**2. Non-Minimal Parameterizations ($K$ Softmax Logits)**

- Strict convexity ($\nabla^2 A(\boldsymbol{\eta}) \succ \mathbf{0}$) holds iff no linear combination of $T(\mathbf{x})$ is constant (**minimal** representation).
- One-hot categorical vectors satisfy $\mathbf{1}^\top \mathbf{y} = \sum_{k=1}^K y_k = 1$, so shifting $\mathbf{z} \to \mathbf{z} + c\mathbf{1}$ leaves all softmax probabilities unchanged ($\nabla^2 A(\mathbf{z})\mathbf{1} = \mathbf{0}$).
- **Practical benefit:** Frameworks keep all $K$ logits and subtract $c = \max_k z_k$ before exponentiating (`logsumexp` trick) to prevent floating-point overflow.
:::
:::
:::

::: {.slide title="Generalized Linear Models and Canonical Links"}
[Why Every Standard Loss Has Gradient Input × (Prediction − Target)]{.kicker}

A **generalized linear model (GLM)** with the **canonical link** sets the natural parameter directly to a linear map of the features, $\boldsymbol{\eta} = \mathbf{W}^\top \mathbf{x}$. Writing $\boldsymbol{\mu}(\mathbf{x}) = \nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(y) \mid \mathbf{x}]$, the chain rule gives:

$$\nabla_{\mathbf{W}}\bigl(-\log p(y \mid \mathbf{x})\bigr) = \mathbf{x}\,\bigl(\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) - T(y)\bigr)^\top = \mathbf{x}\,\bigl(\boldsymbol{\mu}(\mathbf{x}) - T(y)\bigr)^\top$$

| Target Label $y$ | Exponential Family | Statistic $T(y)$ | Canonical Mean Map $\boldsymbol{\mu}(\mathbf{x}) = \nabla A(\boldsymbol{\eta})$ | Generalized Linear Model |
| :--- | :--- | :--- | :--- | :--- |
| Real scalar $y \in \mathbb{R}$ | Gaussian ($\sigma^2 = 1$) | $y$ | Identity $\eta$ | Linear regression ($L_2$ loss) |
| Binary $y \in \{0, 1\}$ | Bernoulli | $y$ | $\mathrm{sigmoid}(\eta) = (1 + e^{-\eta})^{-1}$ | Logistic regression (BCE) |
| Class ID $y \in \{1, \dots, K\}$ | Categorical | One-hot $\mathbf{y}$ | $\mathrm{softmax}(\boldsymbol{\eta})$ | Softmax regression (CE) |
| Count $y \in \{0, 1, 2, \dots\}$ | Poisson | $y$ | Exponential $e^\eta$ | Poisson count regression |

::: {.d2l-note}
**Deep neural networks:** A deep network replaces raw input $\mathbf{x}$ with a learned representation $\mathbf{h}_\theta(\mathbf{x})$ feeding into a canonical GLM output head, preserving the exact output-layer gradient $\boldsymbol{\mu}(\mathbf{x}) - T(y)$.
:::
:::

::: {.slide title="Origins of the Softmax: Noisy Utility & Gumbel-Max"}
[Discrete Choice Theory & Exact Categorical Sampling]{.kicker}

Suppose a decision maker picks the class with the highest noise-perturbed score $y = \operatorname{argmax}_{k \in \{1,\dots,K\}}(z_k + \epsilon_k)$:

- **Why Gaussian noise ($\epsilon_k \sim \mathcal{N}(0, s^2)$) stalls for $K > 2$:** The **probit** model $P(y=1) = \Phi\!\left(\frac{z_1 - z_2}{\sqrt{2}s}\right)$ requires a $(K-1)$-dimensional numerical integral with no closed form as soon as $K \ge 3$.
- **Proposition (Gumbel Noise Yields Softmax):** If $\epsilon_1, \dots, \epsilon_K$ are i.i.d. standard **Gumbel** variables with CDF $G(t) = \exp(-e^{-t})$, then $z_j + \epsilon_j$ has CDF $\exp(-e^{z_j}e^{-t})$. Integrating over the winning score $t = z_k + \epsilon_k$ with $S = \sum_{j=1}^K e^{z_j}$ and $v = e^{-t}$ closes analytically:
  $$P\Bigl(\operatorname*{argmax}_j (z_j + \epsilon_j) = k\Bigr) = \int_{-\infty}^{\infty} e^{z_k}e^{-t}\exp(-Se^{-t})\,dt = e^{z_k}\int_0^{\infty} e^{-Sv}\,dv = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
- **Log-sum-exp as a smooth maximum:** The winning score itself satisfies $\max_j(z_j + \epsilon_j) \sim \mathrm{Gumbel}(\log \sum_j e^{z_j})$, and its gradient with respect to $\mathbf{z}$ is the softmax vector!
:::

::: {.slide title="The Gumbel-Max Trick and Temperature Scaling"}
[Three Converging Views of Softmax]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Gumbel-Max sampler:** Since $\operatorname{argmax}_k(z_k + T\epsilon_k) = \operatorname{argmax}_k(z_k/T + \epsilon_k)$, adding Gumbel noise of scale $T > 0$ and taking the argmax draws exact samples from $\mathrm{softmax}(\mathbf{z}/T)$.
- **2. Statistical physics (Boltzmann law):** At temperature $T$ with state energy $E_k = -z_k$, thermal equilibrium assigns $p_k \propto e^{-E_k/T} = e^{z_k/T}$ (also the MaxEnt distribution at fixed expected logit $\sum_k p_k z_k$).
- **3. Probit vs. logit tails:** While $|\mathrm{sigmoid}(1.7x) - \Phi(x)| < 0.01$, on a mislabeled outlier ($m \gg 0$) logit loss grows linearly ($\sim m$) while probit loss grows quadratically ($\sim m^2/4s^2$).
:::

::: {.col}
Verifying the Gumbel-max sampler on $\mathbf{z} = (2, 1, 0.1, -1)$ across $10^5$ trials at $T = 1$ and $T = 2$:

@mdl-exponential-family-noisy-choices-and-gumbel-noise
:::
:::
:::

::: {.slide title="Bayesian Updating and the Beta Conjugate Prior"}
[From Point Estimates to Posterior Distributions]{.kicker}

The point estimate $\hat{\theta} = 9/13 \approx 0.692$ cannot distinguish $9/13$ flips from $900/1300$ flips. Treating the bias $\theta \in [0, 1]$ as a random variable with a **Beta prior** $\mathrm{Beta}(\theta \mid a, b)$ ($a, b > 0$) gives:

$$\mathrm{Beta}(\theta \mid a, b) = \frac{\theta^{a-1}(1 - \theta)^{b-1}}{B(a, b)}, \qquad B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a + b)}, \qquad \mathbb{E}[\theta] = \frac{a}{a + b}, \qquad \mathrm{Var}(\theta) = \frac{\mathbb{E}[\theta](1 - \mathbb{E}[\theta])}{a + b + 1}$$

- **Conjugate multiplication by the binomial likelihood $\theta^x(1 - \theta)^{n - x}$:**
  $$p(\theta \mid x) \propto \underbrace{\theta^{a-1}(1 - \theta)^{b-1}}_{\text{prior}} \cdot \underbrace{\theta^x(1 - \theta)^{n - x}}_{\text{likelihood}} = \theta^{(a + x) - 1}(1 - \theta)^{(b + n - x) - 1} \implies \theta \mid x \sim \mathrm{Beta}(a + x,\ b + n - x)$$
- **Mean vs. mode pseudo-counts:**
  - **Posterior mean (and next-flip predictive probability):** $\mathbb{E}[\theta \mid x] = \frac{a + x}{a + b + n}$ ($a, b$ act as prior pseudo-counts; for $\mathrm{Beta}(1, 1)$, this gives **Laplace's rule of succession** $\frac{x + 1}{n + 2} = \frac{10}{15} = \frac{2}{3}$).
  - **Posterior mode (MAP):** $\hat{\theta}_{\mathrm{MAP}} = \frac{a + x - 1}{a + b + n - 2}$ ($a - 1, b - 1$ act as mode pseudo-counts; for $\mathrm{Beta}(1, 1)$, mode $= \frac{9}{13}$).
:::

::: {.slide title="Posterior Contraction and Non-Conjugate Grid Priors"}
[Analytical Beta Updating vs. 1D Numerical Integration]{.kicker}

::: {.cols .vc}
::: {.col}
- **Contraction at rate $1/\sqrt{n}$:** From $\mathrm{Beta}(1, 1)$, $9/13$ heads yields $\mathrm{Beta}(10, 5)$ ($\mathrm{sd} \approx 0.118$); $90/130$ heads yields $\mathrm{Beta}(91, 41)$ ($\mathrm{sd} \approx 0.040$, smaller by $\sqrt{133/16} \approx 2.9\times$).
- **Non-conjugate prior ($\eta \sim \mathcal{N}(0, 1)$ on log-odds):** By change of variables, $p(\theta) = \frac{\phi(\eta)}{\theta(1-\theta)}$ is not a Beta density, but in 1D a fine grid normalizes $p(\theta)P(x\mid\theta)$ directly:

@!distributions-conjugate
:::

::: {.col .fig}
@fig:mdl-prob-beta-posterior
:::
:::
:::

::: {.slide title="Universal Conjugacy and Dirichlet Smoothing"}
[General Exponential-Family Priors]{.kicker}

**Proposition (Universal Conjugate Prior):** Every exponential-family likelihood $p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ admits a conjugate prior parameterized by pseudo-statistic sum $\boldsymbol{\nu} \in \mathbb{R}^d$ and pseudo-sample size $\kappa > 0$:

$$p(\boldsymbol{\eta} \mid \boldsymbol{\nu}, \kappa) \propto \exp\bigl(\boldsymbol{\nu}^\top \boldsymbol{\eta} - \kappa\,A(\boldsymbol{\eta})\bigr) \;\;\xrightarrow{n \text{ observations}}\;\; p(\boldsymbol{\eta} \mid \mathbf{x}_{1:n}) \propto \exp\!\left(\Bigl(\boldsymbol{\nu} + \sum_{i=1}^n T(\mathbf{x}_i)\Bigr)^\top \boldsymbol{\eta} - (\kappa + n)A(\boldsymbol{\eta})\right)$$

| Conjugate Pair | Prior Density | Observed Data ($n$ draws) | Posterior Update | Practical ML Application |
| :--- | :--- | :--- | :--- | :--- |
| **Beta–Binomial** | $\mathrm{Beta}(\theta \mid a, b)$ | $x$ heads, $n - x$ tails | $\mathrm{Beta}(a + x,\ b + n - x)$ | Binary accuracy & A/B click rates |
| **Gamma–Poisson** | $\mathrm{Gamma}(\lambda \mid a, b)$ | Total count $\sum_{i=1}^n x_i$ | $\mathrm{Gamma}(a + \sum_i x_i,\ b + n)$ | Arrival rate estimation ($a$ events in $b$ windows) |
| **Dirichlet–Multinomial** | $\mathrm{Dir}(\mathbf{p} \mid \mathbf{a}) \propto \prod_{k=1}^K p_k^{a_k - 1}$ | Class count vector $\mathbf{x}$ | $\mathrm{Dir}(\mathbf{a} + \mathbf{x})$ | **Add-one smoothing:** counts $(3,0,1)$ with $\mathrm{Dir}(\mathbf{1}) \to$ mean $(4/7, 1/7, 2/7) \ne 0$ |
:::

::: {.slide title="Concept Check: Cumulants, GLMs & Conjugacy"}
[Check Your Understanding]{.kicker}

Test your mastery of the exponential family on three problems:

1. **Moments from the log-partition:** The exponential distribution $\mathrm{Exp}(\lambda)$ has $T(x) = x$, natural parameter $\eta = -\lambda < 0$, and $A(\eta) = -\log(-\eta)$. Compute $A'(\eta)$ and $A''(\eta)$ to recover $\mathbb{E}[X]$ and $\mathrm{Var}(X)$.
2. **Poisson GLM gradient step:** A canonical-link Poisson regression model ($\hat{\lambda} = e^\eta$ with $\eta = \mathbf{w}^\top\mathbf{x}$) predicts $\hat{\lambda} = 3$ for input $\mathbf{x}$, and observes count $y = 5$. What is the gradient of the NLL with respect to $\eta$ and $\mathbf{w}$?
3. **Dirichlet add-one smoothing:** A 4-word vocabulary has observed counts $\mathbf{x} = (5, 0, 1, 0)$ ($n = 6$). What is the MLE probability vector vs. the posterior mean under a uniform $\mathrm{Dir}(1, 1, 1, 1)$ prior?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $A'(\eta) = -\frac{1}{\eta} = \mathbf{\frac{1}{\lambda} = \mathbb{E}[X]}$, and $A''(\eta) = \frac{1}{\eta^2} = \mathbf{\frac{1}{\lambda^2} = \mathrm{Var}(X)}$.
2. By $\nabla_\eta \mathcal{L} = \mu - T(y)$, the gradient with respect to $\eta$ is **$3 - 5 = -2$**, and with respect to $\mathbf{w}$ is **$-2\mathbf{x}$** (so gradient descent increases $\eta$ and pushes $\hat{\lambda}$ toward $5$).
3. $\hat{\mathbf{p}}_{\mathrm{MLE}} = (5/6, 0, 1/6, 0)$ assigns zero probability to unseen words. The $\mathrm{Dir}(1,1,1,1)$ posterior is $\mathrm{Dir}(6, 1, 2, 1)$ with posterior mean **$(0.6, 0.1, 0.2, 0.1)$**.
:::
:::

::: {.slide title="Summary: The Exponential Family"}
[Key Takeaways]{.kicker}

| Core Property | Mathematical Statement | Coin / Benchmark Result | Deep Learning Significance |
| :--- | :--- | :--- | :--- |
| **Canonical Form** | $p(\mathbf{x}\mid\boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ | Bernoulli: $\eta = \log\frac{p}{1-p}$, $A(\eta) = \log(1+e^\eta)$ | Unifies Bernoulli, Categorical, Gaussian, Poisson |
| **Sufficient Statistics** | $\prod_{i=1}^n p(\mathbf{x}_i\mid\boldsymbol{\eta}) \propto \exp(\boldsymbol{\eta}^\top \sum_i T(\mathbf{x}_i) - nA(\boldsymbol{\eta}))$ | Total heads $\sum_i x_i = 9$ holds all information | Compresses arbitrary datasets into fixed-size sums |
| **Maximum Entropy** | $H_h[q] = H_h[p] - D_{\mathrm{KL}}(q \| p) \le H_h[p]$ when $\mathbb{E}_q[T]=\mathbb{E}_p[T]$ | Loaded die ($\mathbb{E}=4.5$): tilt $p_k \propto e^{0.371k}$ | Explains why Gaussian, Poisson & Boltzmann arise |
| **Cumulant Derivatives** | $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(\mathbf{x})]$; $\nabla^2 A(\boldsymbol{\eta}) = \mathrm{Cov}(T(\mathbf{x})) \succeq \mathbf{0}$ | Poisson $A(\eta) = e^\eta \implies \text{mean} = \text{var} = \lambda$ | Proves $A(\boldsymbol{\eta})$ and the NLL are globally convex |
| **Moment Matching & GLMs** | $\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T}$; $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{x}(\boldsymbol{\mu}(\mathbf{x}) - T(y))^\top$ | $\mathrm{sigmoid}(\hat{\eta}) = 9/13$; diverges if $\bar{T} \in \{0, 1\}$ | Universal output-layer gradient across losses |
| **Softmax & Conjugacy** | $\operatorname{argmax}_k(z_k + T\epsilon_k) \sim \mathrm{softmax}(\mathbf{z}/T)$; $\boldsymbol{\nu} \to \boldsymbol{\nu}+\sum_i T_i$ | $\mathrm{Beta}(1,1) \to \mathrm{Beta}(10,5)$, mean $2/3$ | Gumbel-max sampling, temperature & Laplace smoothing |
:::

