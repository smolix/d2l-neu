# Concentration Inequalities
:label:`sec_mdl-concentration-generalization`

Whenever we report a model's test accuracy, we are reporting a measurement made
on a finite sample. In :numref:`sec_mdl-statistics`, we attached an error bar to
that measurement, $\pm1.96\,\hat\sigma/\sqrt n$, by invoking the central limit
theorem. That Gaussian approximation is convenient and often accurate, but it is
exact only in the limit $n\to\infty$, and it relies on plugging in an estimated
standard deviation $\hat\sigma$. Earlier, in :numref:`sec_mdl-random_variables`,
we saw a bound that holds with no approximations at every finite $n$, Chebyshev's
inequality, yet it turned out to be painfully weak. This leaves us with a natural
question: how far can a test-set error stray from the truth if we want a
guarantee that holds at the sample size we actually have, and remains useful
even when we demand that it almost never fail?

To make the gap concrete, consider a classifier that makes 830 mistakes on
10,000 test images, giving an observed error rate of 8.3%. Chebyshev's
inequality bounds the probability that the average of $n$ independent draws
misses its mean by $t$ or more by $\sigma^2/(nt^2)$. Because a 0/1 outcome has
variance at most $\sigma^2\leq\tfrac14$, Chebyshev guarantees that the true
error lies within $\pm2.2\%$ of 8.3% with 95% confidence. That is already wider
than the Gaussian interval, and worse still, it collapses when we demand high
reliability: ask for a guarantee that fails only once in a million, and
Chebyshev answers $\pm500\%$, which tells us nothing at all. Yet the true
probability that an average of 0/1 outcomes strays far from its mean decays
*exponentially* in $n$, not like $1/n$. A bound that captures this exponential
decay answers $\pm1.4\%$ at 95% confidence and $\pm2.7\%$ at one in a million.

The key trick behind these sharper bounds is surprisingly simple: apply Markov's
inequality not to the average itself, but to its exponential, which turns a sum
of independent terms into a product. Sergei Bernstein developed bounds of this
kind in the 1920s, Herman Chernoff used the method in 1952 to measure how
rapidly statistical tests become reliable, and Wassily Hoeffding gave in 1963
the clean form used throughout machine learning :cite:`Hoeffding.1963`.

Why would a practitioner ever ask for a guarantee that fails only once in a
million? In practice, we rarely evaluate just one model: a learner compares many
candidates and keeps the best. Guaranteeing that *none* of them looks good by
chance requires a tiny failure probability for each one, and only an
exponentially shrinking tail makes that affordable at a realistic sample size, a
theme we will develop fully in :numref:`sec_mdl-generalization-bounds`. In this
section, we first use exponential tail bounds to size a test set for a desired
precision. We then apply the exact same machinery to the coordinates of a random
vector in $\mathbb{R}^d$, revealing two striking geometric facts: a standard
Gaussian vector in $\mathbb{R}^{784}$ almost always has length within a few
percent of 28, and two independent random directions are nearly orthogonal. Both
facts underpin neural network initialization and high-dimensional embeddings.

Three results from this section are worth committing to memory: the Chernoff
method :eqref:`eq_mdl-chernoff`, Hoeffding's confidence interval
:eqref:`eq_mdl-hoeffding-interval`, and the concentration of the Euclidean norm
:eqref:`eq_mdl-norm-concentration`.

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-concentration-generalization-concentration-and-generalization}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-concentration-generalization-concentration-inequalities-1}
rng = np.random.default_rng()
```

## Exponential Tail Bounds

The Gaussian confidence interval is narrow because Gaussian tails decay like
$e^{-z^2/2}$, a rate inherited from the central limit theorem and therefore
guaranteed only in the limit of large $n$. We want that same rapid decay
combined with a guarantee that holds at every finite sample size. Before
building such a bound, let us see just how far Chebyshev's inequality falls
short of the truth.

### Chebyshev's Rate

Chebyshev's bound is simply Markov's inequality :eqref:`eq_mdl-markov` applied
to a squared deviation. Consider the sample mean $\bar X$ of $n$ independent
draws, each with mean $\mu$ and finite variance $\sigma^2$. Because the
expected squared deviation is $\mathbb{E}[(\bar X-\mu)^2]=\textrm{Var}(\bar
X)=\sigma^2/n$, Markov's inequality yields

$$
P\bigl(|\bar X-\mu|\geq t\bigr)\leq\frac{\sigma^2}{nt^2}.
$$

Here the miss probability shrinks only like $1/n$. To push the failure
probability from $1$ down to $10^{-6}$ at a fixed deviation $t$, we must grow
the sample size a million-fold.

How small is the actual miss probability? The simplest setting where we can
compute it exactly is a fair coin. Let $\hat p$ denote the fraction of heads in
$n$ independent flips, and ask for the probability that $\hat p$ misses
$\tfrac12$ by $t=0.1$ or more. Each flip has variance $\tfrac14$, so
Chebyshev's bound is $0.25/(0.01\,n)=25/n$. The exact probability is a sum of
binomial probabilities, which we evaluate in log space so that $n=5{,}000$ does
not underflow to zero.

```{.python .input #mdl-concentration-generalization-chebyshev-s-rate-1}
def exact_tail(n, t):
    """P(|p_hat - 1/2| >= t) for n fair flips, a binomial sum in log space."""
    k = np.arange(n + 1)
    log_pmf = n * np.log(0.5) + np.concatenate(
        ([0.0], np.cumsum(np.log((n - k[1:] + 1) / k[1:]))))
    in_tail = log_pmf[np.abs(k / n - 0.5) >= t - 1e-12]  # fp-safe boundary
    m = in_tail.max()
    return np.exp(m) * np.exp(in_tail - m).sum()  # stable log-sum-exp
```

At $n=1{,}000$ Chebyshev allows a miss in one experiment out of forty. Before
reading the table, guess how often such a miss really happens.

```{.python .input #mdl-concentration-generalization-chebyshev-s-rate-2}
t = 0.1
print(f'{"n":>6} {"exact":>10} {"Chebyshev":>10}')
for n in (10, 100, 1000, 5000):
    print(f'{n:6d} {exact_tail(n, t):10.2e} {0.25 / (n * t**2):10.2e}')
```

At $n=10$ the exact tail is $0.75$ while the bound, $2.5$, is trivial. At
$n=100$ the bound drops to $0.25$, roughly four times the exact $0.057$. Beyond
that point the two curves part ways dramatically. At $n=1{,}000$ the exact tail
is $2.7\times10^{-10}$ whereas Chebyshev still allows $0.025$, eight orders of
magnitude larger; by $n=5{,}000$ the gap has widened to forty-two orders of
magnitude. Every tenfold increase in $n$ merely divides Chebyshev's bound by
ten, while the exact tail sheds a fixed number of decimal places for every
hundred flips: on a logarithmic scale it descends along a straight line.

Unfortunately, Chebyshev's bound cannot be tightened without assuming more about
the distribution. As we saw in :numref:`sec_mdl-random_variables`, it holds with
equality for a distribution supported on three points, so no bound relying
solely on the mean and variance can improve upon it in general. A coin flip,
however, has far more structure than two finite moments: its values are bounded,
which keeps all of its higher moments small as well. To harness all moments at
once, we turn to the moment generating function.

### The Chernoff Method

To see how to use higher moments, look at what Markov's and Chebyshev's
inequalities have in common. Suppose a function $g$ lies everywhere above the
indicator $\mathbf 1\{x\geq t\}$, a *majorant* of the indicator. Taking
expectations immediately gives
$P(X\geq t)=\mathbb{E}[\mathbf 1\{X\geq t\}]\leq\mathbb{E}[g(X)]$. Markov's
inequality chooses the linear majorant $g(x)=x/t$ for a nonnegative $X$, while
Chebyshev's chooses the quadratic $g(x)=(x/t)^2$ (applied to the centered
deviation $X-\mu$). Now notice that for any $\lambda>0$, the exponential
$e^{\lambda(x-t)}$ also lies above the indicator across the entire real line.
Panel (a) of :numref:`fig_mdl-conc-majorants` compares all three majorants for a
nonnegative $X$. Since the bound is simply the average of $g(X)$, what determines
its tightness is how small $g$ stays where the probability mass of $X$ actually
sits.

![A tail bound is the expectation of a function lying above the indicator. (a) For a threshold $t=2$, the indicator $\mathbf{1}\{x\geq t\}$ (black) and three functions above it: Markov's $x/t$, Chebyshev's $(x/t)^2$, and Chernoff's $e^{\lambda(x-t)}$ with $\lambda=1.5$. The shaded bump marks where the mass of $X$ sits; each bound is the average of its function under that mass, and raising $\lambda$ makes the exponential small there. (b) The Chernoff exponent $-\lambda nt+n\lambda^2/8$ for the mean of $n=100$ fair coins at $t=0.1$. Every $\lambda>0$ gives a valid bound; the best, at the minimum $\lambda=4t$ of the parabola, is $e^{-2}$.](../img/mdl-prob-conc-majorants.svg)
:label:`fig_mdl-conc-majorants`

The expectation of this exponential has a standard name. The *moment generating
function* (MGF) of $X$ is $\mathbb{E}[e^{\lambda X}]$, viewed as a function of
$\lambda$ wherever the expectation is finite. For a fair coin centered at zero,
$Z\in\{-\tfrac12,\tfrac12\}$, it is simply
$\tfrac12(e^{\lambda/2}+e^{-\lambda/2})=\cosh(\lambda/2)$. The MGF is finite for
every $\lambda$ whenever $X$ is bounded, but it is infinite for every
$\lambda>0$ when the right tail decays only polynomially, as with a Student-$t$
variable; for such heavy-tailed variables, the exponential method below does not
apply.

Applying Markov's inequality to the nonnegative random variable $e^{\lambda X}$
gives the bound in one line. Because $x\mapsto e^{\lambda x}$ is strictly
increasing for $\lambda>0$, the event $\{X\geq t\}$ is identical to
$\{e^{\lambda X}\geq e^{\lambda t}\}$, so

$$
P(X \geq t)
 = P\bigl(e^{\lambda X} \geq e^{\lambda t}\bigr)
 \leq e^{-\lambda t}\,\mathbb{E}\bigl[e^{\lambda X}\bigr],
\qquad\textrm{hence}\qquad
P(X \geq t) \leq \inf_{\lambda > 0}\, e^{-\lambda t}\,
\mathbb{E}\bigl[e^{\lambda X}\bigr].
$$
:eqlabel:`eq_mdl-chernoff`

Every choice of $\lambda>0$ yields a valid upper bound; taking the infimum
simply keeps the tightest one. This two-step recipe is the *Chernoff method*.

There is a second, equally decisive reason why the exponential is the right
majorant. When $S=X_1+\cdots+X_n$ is a sum of independent random variables,
$e^{\lambda S}=\prod_{i=1}^n e^{\lambda X_i}$ becomes a product of independent
factors, and the expectation of a product of independent variables is the
product of their expectations:

$$
\mathbb{E}\bigl[e^{\lambda S}\bigr]=\prod_{i=1}^n\mathbb{E}\bigl[e^{\lambda X_i}\bigr].
$$

Thus the moment generating function of a sum factorizes into the product of the
MGFs of its terms, so the logarithm of the Chernoff bound splits into a sum of
$n$ pieces and scales linearly with $n$. No polynomial majorant enjoys this
property: $\mathbb{E}[S^4]$ does not factor into a product of fourth moments.
All that remains is to bound the MGF of a single summand.

### The Fair Coin

Let us work through the fair coin all the way to the end, since this simplest
case already contains every step of the general proof. A centered flip
$Z=X-\tfrac12$ takes the values $\pm\tfrac12$ with equal probability, so its MGF
is $\mathbb{E}[e^{\lambda Z}]=\cosh(\lambda/2)$. Comparing the Taylor series of
$\cosh u$ and $e^{u^2/2}$ term by term gives

$$
\cosh u=\sum_{k\geq0}\frac{u^{2k}}{(2k)!}
\leq\sum_{k\geq0}\frac{(u^2/2)^k}{k!}=e^{u^2/2},
$$

because $(2k)!\geq2\cdot4\cdots(2k)=2^kk!$: the even factors of $(2k)!$ alone
already produce $2^kk!$, and every odd factor is at least $1$. Setting
$u=\lambda/2$, we see that a single centered flip satisfies
$\mathbb{E}[e^{\lambda Z}]\leq e^{\lambda^2/8}$ for every $\lambda$. By
comparison, a Gaussian $\mathcal N(0,\sigma^2)$ has
$\mathbb{E}[e^{\lambda X}]=e^{\lambda^2\sigma^2/2}$ (found by completing the
square in the integral), so a centered coin's MGF is never larger than that of a
Gaussian with standard deviation $\tfrac12$.

Now consider $n$ independent flips. The event $\hat p-\tfrac12\geq t$ is the
same as $\sum_{i=1}^n Z_i\geq nt$, so the Chernoff method and the product rule
give

$$
P\bigl(\hat p-\tfrac12\geq t\bigr)
\leq e^{-\lambda nt}\prod_{i=1}^n\mathbb{E}\bigl[e^{\lambda Z_i}\bigr]
\leq\exp\Bigl(-\lambda nt+\frac{n\lambda^2}{8}\Bigr).
$$

The exponent is a simple parabola in $\lambda$. Panel (b) of
:numref:`fig_mdl-conc-majorants` plots it for $n=100$ and $t=0.1$, where it
equals $-10\lambda+12.5\lambda^2$. Any $\lambda>0$ gives a valid bound, for
example $e^{-0.875}\approx0.42$ at $\lambda=0.1$. The tightest bound sits at the
vertex of the parabola, $\lambda=4t$, where the exponent simplifies to
$-4nt^2+2nt^2=-2nt^2$. By symmetry the lower tail obeys the same bound, and
adding the two tails yields

$$
P\bigl(|\hat p-\tfrac12|\geq t\bigr)\leq2e^{-2nt^2}.
$$

The probability of straying by $t$ now decays exponentially in $nt^2$. At
$n=100$ and $t=0.1$ this gives $2e^{-2}\approx0.27$, and at $n=1{,}000$ it drops
to $2e^{-20}\approx4.1\times10^{-9}$, a dramatic improvement over Chebyshev's
$0.025$.

Why does the exponential majorant work so well? At the threshold $S=nt$, the
function $e^{\lambda(S-nt)}$ equals $1$, while at the mean $S=0$, where almost
all the probability mass sits, it shrinks to $e^{-\lambda nt}$, exponentially
small in $n$. What prevents us from pushing $\lambda$ to infinity is that the
function grows steeply for $S>0$, so upward fluctuations carry disproportionate
weight. The product of the MGFs, bounded by $e^{n\lambda^2/8}$, measures that
penalty, and the optimal $\lambda=4t$ strikes the exact balance between the two.

Notice that the entire derivation relied on just one property of the coin: the
MGF of a centered flip is bounded by $e^{\lambda^2/8}$, the MGF of a Gaussian
with standard deviation $\tfrac12$, which is half the coin's range. Hoeffding's
lemma generalizes this single step, showing that *every* random variable
confined to an interval $[a,b]$ obeys the same bound with $(b-a)/2$ in place of
$\tfrac12$.

### Hoeffding's Lemma and Hoeffding's Inequality

A test-set loss is rarely a fair coin. An error rate of 8.3% behaves like a
heavily biased coin, and a clipped loss takes a continuum of values rather than
just $0$ and $1$. We would like the fair coin's exponential bound to hold for
*any* bounded random variable. The lemma that delivers this is the main
technical step of the section, and its proof rests on one elementary fact about
variances.

**Proposition (variance of a bounded variable).** *If $Y$ takes values in
$[a,b]$, then $\textrm{Var}(Y)\leq(b-a)^2/4$, with equality when $Y$ takes
the values $a$ and $b$ with probability $\tfrac12$ each.*

**Proof.** The mean minimizes the expected squared deviation, since
$\mathbb{E}[(Y-c)^2]=\textrm{Var}(Y)+(\mathbb{E}[Y]-c)^2$ for every choice of
center $c$. Taking $c=(a+b)/2$, every point of $[a,b]$ lies within $(b-a)/2$ of
the midpoint, so

$$
\textrm{Var}(Y)\leq\mathbb{E}\Bigl[\Bigl(Y-\frac{a+b}{2}\Bigr)^2\Bigr]
\leq\Bigl(\frac{b-a}{2}\Bigr)^2.
$$

Both inequalities become equalities when $Y$ places equal mass at the two
endpoints $a$ and $b$. $\blacksquare$

The fair coin on $[0,1]$ is precisely this extreme case, with variance
$\tfrac14$. With this variance bound in hand, we can prove Hoeffding's lemma.

**Proposition (Hoeffding's lemma).** *Let $X$ be a random variable with
$\mathbb{E}[X]=0$ taking values in $[a,b]$. Then for every
$\lambda\in\mathbb{R}$,*

$$
\mathbb{E}\bigl[e^{\lambda X}\bigr] \leq \exp\!\left(\frac{\lambda^2 (b-a)^2}{8}\right).
$$
:eqlabel:`eq_mdl-hoeffding-lemma`

**Proof.** We want to show that the log-MGF
$\psi(\lambda)=\log\mathbb{E}[e^{\lambda X}]$ grows no faster than the
parabola $\lambda^2(b-a)^2/8$. Because $X$ is bounded, $\psi$ is finite and
smooth everywhere, allowing us to differentiate under the expectation. At
$\lambda=0$, we have $\psi(0)=\log 1=0$ and

$$
\psi'(\lambda) = \frac{\mathbb{E}[X e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]},
\qquad\textrm{so}\qquad
\psi'(0) = \mathbb{E}[X] = 0 .
$$

Just as for the log-partition function of an exponential family
(:numref:`sec_mdl-exponential-family`), the second derivative of $\psi$ is a
variance. Define the *tilted* distribution $q_\lambda$ by reweighting the
distribution of $X$ by $e^{\lambda x}$ and normalizing. Under $q_\lambda$,
$\psi'(\lambda)$ is the mean of $X$, and differentiating once more gives

$$
\psi''(\lambda)
 = \frac{\mathbb{E}[X^2 e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}
 - \left(\frac{\mathbb{E}[X e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}\right)^{\!2}
 = \textrm{Var}_{q_\lambda}(X).
$$

Crucially, exponential tilting only reweights points; it never moves mass
outside $[a,b]$. Our variance bound therefore holds for $q_\lambda$ at every
$\lambda$, giving $\psi''(\lambda)\leq(b-a)^2/4$. Taylor's theorem with the
Lagrange remainder now finishes the proof: for some $\xi$ between $0$ and
$\lambda$,

$$
\psi(\lambda)
 = \psi(0) + \lambda\,\psi'(0) + \frac{\lambda^2}{2}\,\psi''(\xi)
 \leq \frac{\lambda^2 (b-a)^2}{8},
$$

and exponentiating both sides yields :eqref:`eq_mdl-hoeffding-lemma`.
$\blacksquare$

In words, any bounded, centered random variable has a moment generating function
no larger than that of a Gaussian with standard deviation $(b-a)/2$. For a
centered coin, $b-a=1$, so the lemma immediately recovers the bound
$\cosh(\lambda/2)\leq e^{\lambda^2/8}$ that we proved by Taylor series. With
that single ingredient in place, our coin proof carries over verbatim to any
average of independent bounded variables.

**Proposition (Hoeffding's inequality).** *Let $X_1,\ldots,X_n$ be
independent random variables with $X_i\in[a,b]$, and let
$\bar X = \frac1n\sum_i X_i$. Then for every $t>0$,*

$$
P\bigl(|\bar X - \mathbb{E}[\bar X]| \geq t\bigr)
 \leq 2\exp\!\left(-\frac{2 n t^2}{(b-a)^2}\right).
$$
:eqlabel:`eq_mdl-hoeffding`

**Proof.** Let $Z_i = X_i - \mathbb{E}[X_i]$. Each $Z_i$ has mean zero and lies
in an interval of length $b-a$, so Hoeffding's lemma applies to every term. For
any $\lambda>0$, the Chernoff method :eqref:`eq_mdl-chernoff` and the product
rule give

$$
P(\bar X - \mathbb{E}[\bar X] \geq t)
 = P\Bigl(\sum_i Z_i \geq nt\Bigr)
 \leq e^{-\lambda n t} \prod_{i=1}^n \mathbb{E}\bigl[e^{\lambda Z_i}\bigr]
 \leq \exp\!\left(-\lambda n t + \frac{n\lambda^2 (b-a)^2}{8}\right).
$$

This is the same parabola we minimized for the coin, with $(b-a)^2$ in place of
$1$. Its minimum occurs at $\lambda = 4t/(b-a)^2$, where the exponent equals
$-2nt^2/(b-a)^2$. Applying the same argument to $-Z_i$ bounds the lower tail by
the same quantity, and adding the two tails produces the factor of $2$.
$\blacksquare$

Thus the sample mean of $n$ independent bounded variables strays from its
expectation by $t$ with a probability that decays exponentially in $nt^2$,
where $t$ is measured in units of the range $b-a$. Notice what the bound does
*not* require: no Gaussianity, no variance estimate, and no asymptotic limit. It
asks only for boundedness and independence, and it holds at every finite $n$.
Independence, however, is a genuine requirement: the bound does not apply to
the correlated samples of a Markov chain.

Look back at the proof and you will notice we never actually needed all the
intervals to have the same length. If the independent centered variables satisfy
$Y_i\in[a_i,b_i]$, the product of Hoeffding's lemma bounds replaces $n(b-a)^2$
by $\sum_i(b_i-a_i)^2$, and the same quadratic minimization gives

$$
P\Bigl(\Bigl|\sum_iY_i\Bigr|\geq s\Bigr)
\leq2\exp\Bigl(-\frac{2s^2}{\sum_i(b_i-a_i)^2}\Bigr).
$$
:eqlabel:`eq_mdl-hoeffding-unequal`

We will put this unequal-range version to work shortly when we analyze random
directions in high dimensions.

Let us translate Hoeffding's inequality into a practical test-set size. For
losses in $[0,1]$, we have $b-a=1$, so the failure probability is at most
$\delta$ once $n\geq\log(2/\delta)/(2t^2)$. To estimate an error rate within
$t=0.01$ with failure probability $\delta=0.05$, we need
$n\geq\log 40/(2\cdot10^{-4})=18{,}444.4$, or $18{,}445$ test examples. This is
the figure quoted in :numref:`chap_classification_generalization`, roughly twice
the 9,604 examples suggested by the asymptotic Gaussian approximation.

Equivalently, setting the right-hand side of :eqref:`eq_mdl-hoeffding` equal to
$\delta$ and solving for $t$ gives the confidence-interval form used in
practice: with probability at least $1-\delta$,

$$
\bigl|\bar X - \mathbb{E}[\bar X]\bigr|
 \leq (b-a)\sqrt{\frac{\log(2/\delta)}{2n}} .
$$
:eqlabel:`eq_mdl-hoeffding-interval`

Just like the Gaussian interval $\hat\mu\pm1.96\,\hat\sigma/\sqrt n$, the
half-width shrinks at the $1/\sqrt n$ rate, but it comes with a strict
finite-sample guarantee.

Returning to our classifier with 830 errors on 10,000 images at $\delta=0.05$,
Hoeffding gives $\pm1.36\%$, compared with the Gaussian interval
$\pm1.96\sqrt{0.083\cdot0.917/10^4}=\pm0.54\%$. The two differ only in their
constants: Hoeffding replaces the empirical standard deviation $\hat\sigma=0.28$
by its worst-case value $(b-a)/2=0.5$, and replaces the Gaussian quantile $1.96$
by $\sqrt{2\log(2/\delta)}=2.72$, making the interval $2.5$ times wider. Whenever
the actual spread of the data is much smaller than their range, Hoeffding is
inevitably conservative, a gap that Bernstein's inequality will close below.

One might wonder whether the constant $\tfrac18$ in Hoeffding's lemma can be
sharpened. It cannot: a variable placing probability $\tfrac12$ at each endpoint
of $[a,b]$ attains $\tfrac18$ to second order in $\lambda$ (Exercise
:exref:`ex_mdl-concentration-generalization-random-signs`).

### The Fair Coin in Numbers
:label:`subsec_mdl-tail-bounds-numerical`

Let us see how close Hoeffding's bound $2e^{-2nt^2}$ comes to the exact tail of
the fair coin at $t=0.1$. We print both alongside their ratio, and plot them
together with Chebyshev's bound against $n$ on a logarithmic scale.

```{.python .input #mdl-concentration-generalization-the-tail-race-in-code}
print(f'{"n":>6} {"exact":>10} {"Hoeffding":>10} {"ratio":>8}')
for n in (10, 100, 1000, 5000):
    hoeffding = 2 * np.exp(-2 * n * t**2)
    print(f'{n:6d} {exact_tail(n, t):10.2e} {hoeffding:10.2e} '
          f'{hoeffding / exact_tail(n, t):8.1f}')

ns = np.arange(10, 701, 10)
d2l.plot(ns, [np.array([exact_tail(n, t) for n in ns]),
              0.25 / (ns * t**2), 2 * np.exp(-2 * ns * t**2)],
         'n', 'P(|p_hat - 1/2| >= 0.1)',
         legend=['exact', 'Chebyshev', 'Hoeffding'], yscale='log')
```

At $n=10$ both bounds exceed $1$ and are uninformative. At $n=100$, Chebyshev's
$0.25$ is actually a touch tighter than Hoeffding's $0.27$: at small sample
sizes the leading constants decide, and Chebyshev benefits from knowing the
exact variance $\tfrac14$. Once $n$ grows, however, the exponential rate takes
over completely. Hoeffding stays within a factor of 15 of the exact tail at
$n=1{,}000$ and within a factor of 57 at $n=5{,}000$.

On the logarithmic plot, the exact tail and Hoeffding's bound trace nearly
parallel straight lines. The logarithm of Hoeffding's bound drops by
$2t^2=0.02$ per flip, while the exact tail drops by roughly $0.0201$ per flip.
That sharper slope is the Kullback–Leibler divergence
$D_{\textrm{KL}}(0.6\|0.5)=0.6\log1.2+0.4\log0.8$ between a biased coin with
probability $0.6$ and a fair coin (:numref:`sec_mdl-divergences-distances`).
Applying the Chernoff method with the exact MGF of a Bernoulli variable recovers
this exact rate (Exercise :exref:`ex_mdl-concentration-generalization-chernoff-coins`).

Of course, when $n$ is tiny, no inequality can create evidence that is not
there. Recall the coin from the chapter introduction that landed heads 9 times
in 13 flips, a deviation of $t=2.5/13$ from $\tfrac12$. Hoeffding's bound on
such a deviation is $2e^{-2\cdot13\cdot(2.5/13)^2}=2e^{-0.96}\approx0.77$,
which rules out nothing. Even the exact two-sided binomial probability is
$0.27$: thirteen flips simply cannot rule out a fair coin.

### Sub-Gaussian and Sub-Exponential Variables

Hoeffding's interval for our 8.3% classifier was 2.5 times wider than the
Gaussian interval because it knows only the range of a 0/1 loss and ignores its
small variance. *Bernstein's inequality* incorporates both.

**Proposition (Bernstein's inequality).** *Let $X_1,\ldots,X_n$ be
independent with $\textrm{Var}(X_i)\leq\sigma^2$ and
$|X_i-\mathbb{E}[X_i]|\leq M$. Then for every $t>0$,*

$$
P\bigl(|\bar X - \mathbb{E}[\bar X]| \geq t\bigr)
 \leq 2\exp\!\left(-\frac{n t^2}{2\bigl(\sigma^2 + M t/3\bigr)}\right).
$$
:eqlabel:`eq_mdl-bernstein`

Once again, the proof is the Chernoff method
:cite:`Boucheron.Lugosi.Massart.2013`, paired with an MGF bound that tracks both
the variance and the range (worked out in Exercise
:exref:`ex_mdl-concentration-generalization-bernstein`). Look at the two regimes
in the denominator. When $t$ is small, $\sigma^2$ dominates $Mt/3$ and the bound
behaves like a Gaussian tail with the *true* variance $\sigma^2$. When $t$ is
large, the linear term $Mt/3$ takes over and the tail decays only exponentially
in $t$ rather than as a Gaussian in $t^2$.

How much does accounting for the variance help in practice? Consider a
classifier with a 2% error rate on $n=10{,}000$ test examples: each 0/1 loss has
variance $\sigma^2=0.02\cdot0.98$, while its centered deviation satisfies $M=1$.
Setting the right-hand side of :eqref:`eq_mdl-bernstein` equal to $\delta$
produces a quadratic equation in $t$, namely
$nt^2/(2(\sigma^2+t/3))=\log(2/\delta)$. Below, we solve this quadratic at
$\delta=0.05$ for both our 8.3% and 2% error rates alongside Hoeffding and the
Gaussian interval, computing both the plug-in estimate (using the observed rate
in the variance) and a fully guaranteed version that we explain right after the
output.

```{.python .input #mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables}
def half_widths(p, n, delta=0.05, z=1.96):
    """Hoeffding, Bernstein (plug-in, guaranteed), Gaussian half-widths."""
    L = np.log(2 / delta)
    hoeffding = np.sqrt(L / (2 * n))
    c = 2 * L / 3  # Bernstein with M = 1: n t^2 - c t - 2 L var = 0
    bernstein = lambda var: (c + np.sqrt(c**2 + 8 * n * L * var)) / (2 * n)
    plug_in = t = bernstein(p * (1 - p))
    for _ in range(20):  # true rate at most p + t: a fixed point in t
        t = bernstein((p + t) * (1 - p - t))
    return hoeffding, plug_in, t, z * np.sqrt(p * (1 - p) / n)
```

```{.python .input #mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables-1}
for p in (0.083, 0.02):
    hoeffding, plug_in, bernstein, gauss = half_widths(p, 10000)
    print(f'error {p:.1%}: Hoeffding ±{hoeffding:.2%}, Bernstein '
          f'±{plug_in:.2%} (plug-in) ±{bernstein:.2%} (guaranteed), '
          f'Gaussian ±{gauss:.2%}')
```

At a 2% error rate, Bernstein's plug-in half-width shrinks to $\pm0.39\%$,
compared with Hoeffding's $\pm1.36\%$: roughly three and a half times tighter,
and within a factor of $1.4$ of the asymptotic Gaussian $\pm0.27\%$. At 8.3%
error the improvement is smaller but still substantial, narrowing $\pm1.36\%$
down to $\pm0.76\%$.

Strictly speaking, those two plug-in numbers ($\pm0.39\%$ and $\pm0.76\%$) are
not yet true finite-sample guarantees. The code substituted the observed rate
$\hat p$ into the variance $p(1-p)$, whereas Bernstein's inequality requires an
upper bound on the *true* variance, the exact same plug-in objection we raised
against the Gaussian interval. Fortunately, fixing this is inexpensive. Whenever
the bound holds, the true error rate is at most $\hat p+t$, so we simply solve
the same equation with the worst-case variance $(\hat p+t)(1-\hat p-t)$.
Iterating this fixed-point equation (as the cell does) nudges the half-widths
only slightly, to the fully guaranteed $\pm0.43\%$ and $\pm0.79\%$. This gives a
sharp answer to our opening question: for the classifier with 830 errors on
10,000 images, its true error rate lies within about $0.8\%$ of 8.3% with 95%
confidence. Because practical models typically operate at error rates of a few
percent, variance-aware bounds make a real difference.

Stepping back, Hoeffding's proof relied on just one property of the summands:
their moment generating functions were bounded by that of a Gaussian. Naming
this property lets us extend the same proof to unbounded random variables, and
clarifies exactly where it stops working. A random variable $X$ is
**sub-Gaussian** with *variance proxy* $\sigma^2$ if its centered MGF never
exceeds that of $\mathcal N(0,\sigma^2)$:

$$
\mathbb{E}\bigl[e^{\lambda (X - \mathbb{E}[X])}\bigr] \leq \exp\!\left(\frac{\lambda^2\sigma^2}{2}\right)
\qquad\textrm{for all } \lambda\in\mathbb{R}.
$$
:eqlabel:`eq_mdl-subgaussian`

Any sub-Gaussian variable has tails at least as light as a Gaussian's: applying
the Chernoff method with $\lambda=t/\sigma^2$ yields
$P(|X-\mathbb{E}[X]|\geq t)\leq2e^{-t^2/(2\sigma^2)}$. Keep in mind that
$\sigma^2$ here is a *proxy*, not necessarily the true variance. Comparing both
sides of :eqref:`eq_mdl-subgaussian` to second order in $\lambda$ shows that the
proxy is never smaller than $\textrm{Var}(X)$, and it can be far larger: a 0/1
outcome with 2% error has variance $0.0196$, yet Hoeffding's lemma supplies the
proxy $\tfrac14$, which is precisely why Hoeffding's interval was so wide.

Three standard families are sub-Gaussian. A Gaussian $\mathcal N(\mu,\sigma^2)$
has proxy $\sigma^2$ with equality; any variable on $[a,b]$ has proxy
$(b-a)^2/4$ by Hoeffding's lemma; and a uniform random sign $\xi\in\{-1,+1\}$
has proxy $1$, since $\mathbb{E}[e^{\lambda\xi}]=\cosh\lambda\leq e^{\lambda^2/2}$
by our earlier series comparison. Best of all, sub-Gaussianity is preserved by
averaging: if $X_1,\ldots,X_n$ are independent with proxy $\sigma^2$, the
product rule gives $\bar X$ the proxy $\sigma^2/n$, and therefore
$P(|\bar X-\mathbb{E}\bar X|\geq t)\leq2e^{-nt^2/(2\sigma^2)}$.

Not every well-behaved random variable is sub-Gaussian, however, and the very
next quantity we need is not. The squared length of a Gaussian vector is a sum
of squares of standard Gaussians, and for a single standard Gaussian $x$,
completing the square gives

$$
\mathbb{E}\bigl[e^{\lambda(x^2-1)}\bigr]=\frac{e^{-\lambda}}{\sqrt{1-2\lambda}}
\qquad\textrm{for }\lambda<\tfrac12,
$$

while the expectation diverges to infinity for every $\lambda\geq\tfrac12$.
Because the MGF blows up at $\lambda=\tfrac12$, :eqref:`eq_mdl-subgaussian`
cannot hold for $x^2$ at any variance proxy: its right tail decays like
$e^{-t/2}$, too slowly for a Gaussian bound.

Even so, a Gaussian-shaped MGF bound *does* hold on a restricted window around
$\lambda=0$. We say that a random variable $X$ is **sub-exponential** with
parameters $(\nu,\alpha)$ if

$$
\mathbb{E}\bigl[e^{\lambda (X - \mathbb{E}[X])}\bigr] \leq \exp\!\left(\frac{\lambda^2\nu^2}{2}\right)
\qquad\textrm{for } |\lambda| \leq \frac{1}{\alpha}.
$$
:eqlabel:`eq_mdl-subexponential`

Here $\nu^2$ acts as a local variance proxy near $\lambda=0$, while $1/\alpha$
sets the half-width of the admissible window, capping how large a $\lambda$ the
Chernoff method may choose. In :numref:`subsec_mdl-thin-shell` we will see that
$x^2$ is sub-exponential with $\nu=2$ and $\alpha=4$. When we apply the Chernoff
method to an average of sub-exponential variables, the optimization naturally
splits at the boundary of this window. You may skip the proof on a first
reading; the next subsection uses only the statement.

**Proposition (sub-exponential tail bound).** *Let $X_1,\ldots,X_n$ be
independent and $(\nu,\alpha)$-sub-exponential. Then for every $t>0$,*

$$
P\bigl(|\bar X-\mathbb{E}[\bar X]|\geq t\bigr)
\leq2\exp\!\left(-\frac n2\min\Bigl(\frac{t^2}{\nu^2},\,\frac{t}{\alpha}\Bigr)\right).
$$
:eqlabel:`eq_mdl-subexponential-tail`

**Proof.** For any $0<\lambda\leq1/\alpha$, the Chernoff method and the product
rule bound $P(\bar X-\mathbb{E}[\bar X]\geq t)$ by
$\exp(-\lambda nt+n\lambda^2\nu^2/2)$. When $t\leq\nu^2/\alpha$, the
unconstrained minimizer $\lambda=t/\nu^2$ falls inside the window and yields
$e^{-nt^2/(2\nu^2)}$. When $t>\nu^2/\alpha$, that minimizer lies outside the
window, so we clamp $\lambda$ to the largest admissible value,
$\lambda=1/\alpha$. The exponent becomes $-nt/\alpha+n\nu^2/(2\alpha^2)$, which
is at most $-nt/\alpha+nt/(2\alpha)=-nt/(2\alpha)$ because $\nu^2/\alpha<t$.
Combining both regimes gives an exponent of at most
$-\tfrac n2\min(t^2/\nu^2,t/\alpha)$. The lower tail follows identically using
$\lambda\in[-1/\alpha,0)$. $\blacksquare$

The contrast with sub-Gaussian averages is now crystal clear. Averages of
bounded or sub-Gaussian variables enjoy Gaussian-type tails $e^{-ct^2}$ at every
scale. A sub-exponential average has that same Gaussian tail for moderate
deviations below $\nu^2/\alpha$, and then transitions to a slower exponential
tail $e^{-nt/(2\alpha)}$ farther out.

## Probability in High Dimension

Up to now, our bounds have protected a single sample average. Now consider a
random vector in $\mathbb{R}^d$ with $d$ independent coordinates: both its
squared length and its inner products with other vectors are sums over those $d$
coordinates. Concentration inequalities therefore govern the geometry of
high-dimensional space. If you are jumping in here, you only need two results
from the first half of the section: the sub-exponential tail bound
:eqref:`eq_mdl-subexponential-tail` for squared lengths, and the unequal-range
Hoeffding bound :eqref:`eq_mdl-hoeffding-unequal` for inner products.

### The Thin Shell
:label:`subsec_mdl-thin-shell`

Draw a standard Gaussian vector $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf
I_{784})$, which has as many coordinates as a $28\times28$ image. Its
probability density is highest at the origin. Surprisingly, however, the
probability that $\mathbf x$ lands within distance 14 of the origin (half its
typical length) is roughly $10^{-110}$, the chance that a sum of 784 squared
standard Gaussians is at most 196. No sample you could ever draw on a computer
will contain a point anywhere near the mode. Instead, a typical draw has norm
$28.0\pm0.7$.

The explanation is that $\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ is a sum of $d$
independent terms, each with mean $1$, so $\|\mathbf x\|^2/d$ is simply a sample
mean. Because $x_i^2$ is unbounded, Hoeffding's inequality does not apply, but
our sub-exponential bound handles it cleanly.

**Proposition (norm concentration).** *Let
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$. For every
$0<\varepsilon\leq 1$,*

$$
P\!\left(\Bigl|\frac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \geq \varepsilon\right)
 \leq 2\exp\!\left(-\frac{d\,\varepsilon^2}{8}\right).
$$
:eqlabel:`eq_mdl-norm-concentration`

**Proof.** The argument proceeds in three short steps. First, a power-series
comparison (Exercise :exref:`ex_mdl-concentration-generalization-chi-square-tail`)
shows that the MGF of a centered square, $e^{-\lambda}/\sqrt{1-2\lambda}$,
satisfies

$$
\mathbb{E}\bigl[e^{\lambda (x_i^2 - 1)}\bigr] \leq e^{2\lambda^2}
\qquad\textrm{for } |\lambda| \leq \tfrac14 .
$$
:eqlabel:`eq_mdl-chi2-mgf`

In the terminology of :eqref:`eq_mdl-subexponential`, $x_i^2$ is sub-exponential
with $\nu=2$ and $\alpha=4$. Second, applying the sub-exponential tail bound
:eqref:`eq_mdl-subexponential-tail` with $n=d$ and $t=\varepsilon$ gives

$$
P\Bigl(\Bigl|\frac{\|\mathbf x\|^2}{d}-1\Bigr|\geq\varepsilon\Bigr)
\leq2\exp\Bigl(-\frac d2\min\Bigl(\frac{\varepsilon^2}{4},\frac{\varepsilon}{4}\Bigr)\Bigr)
=2e^{-d\varepsilon^2/8}
\qquad\textrm{for }\varepsilon\leq1.
$$

Third, we pass from the squared norm to the norm itself. Let
$z=\|\mathbf x\|/\sqrt d\geq0$. Whenever $|z-1|\geq\varepsilon$, we also have
$|z^2-1|=|z-1|\,(z+1)\geq\varepsilon$, since $z+1\geq1$. Thus the event for the
norm is contained in the event for the squared norm and inherits its upper
bound. $\blacksquare$

In short, the probability that the length of a standard Gaussian vector misses
$\sqrt d$ by a fixed fraction decays exponentially in the dimension $d$. Note
that the restriction $\varepsilon\leq1$ is an artifact of bounding the *squared*
norm, whose upper tail beyond $\varepsilon=1$ decays only exponentially as
$e^{-d\varepsilon/8}$ (Exercise
:exref:`ex_mdl-concentration-generalization-chi-square-tail`). The norm itself
keeps a Gaussian tail at every scale: exceeding $(1+\varepsilon)\sqrt d$ forces
the squared norm above $(1+2\varepsilon+\varepsilon^2)d$, whose probability is
at most $e^{-d(2\varepsilon+\varepsilon^2)/8}\leq e^{-d\varepsilon^2/8}$.

How thick is this shell of typical norms? Since $\mathbb{E}[x^4]=3$, each
coordinate square has $\textrm{Var}(x^2)=3-1=2$, giving
$\textrm{Var}(\|\mathbf x\|^2)=2d$. Linearizing the square root around $d$ gives
$\|\mathbf x\|\approx\sqrt d+(\|\mathbf x\|^2-d)/(2\sqrt d)$, so the standard
deviation of $\|\mathbf x\|$ is approximately $\sqrt{2d}/(2\sqrt d)=1/\sqrt2
\approx0.71$, independent of the dimension. Remarkably, a standard Gaussian in
$\mathbb{R}^d$ concentrates on a thin shell of radius $\sqrt d$ whose thickness
stays of order one no matter how large $d$ becomes.

Why does all the probability mass sit so far from the mode? The joint density
$(2\pi)^{-d/2}e^{-\|\mathbf x\|^2/2}$ depends on $\mathbf x$ solely through its
radius $r=\|\mathbf x\|$, while the surface area of a sphere of radius $r$ grows
like $r^{d-1}$. Multiplying density by surface area, the distribution of the
radius $\|\mathbf x\|$ is proportional to $r^{d-1}e^{-r^2/2}$. As $r$ grows, the
volume factor $r^{d-1}$ explodes while the density $e^{-r^2/2}$ plummets; their
product peaks where the derivative of $(d-1)\log r-r^2/2$ vanishes, namely at
$r=\sqrt{d-1}$. Panel (a) of :numref:`fig_mdl-conc-shell` illustrates this
competition between volume and density at $d=20$.

At a typical point of norm $\sqrt d$, the pointwise density is $e^{-d/2}$ times
its value at the origin (a tiny $10^{-170}$ when $d=784$), yet there is so much
more surface area at radius $\sqrt d$ that virtually all the mass lives there.
Panel (b) displays the exact density of
$\|\mathbf x\|$ from $d=2$ to $d=512$: the peaks march outward like $\sqrt d$
while their widths stay pinned near $1/\sqrt2$.

![The thin shell. (a) At $d=20$ the Gaussian density $e^{-r^2/2}$ falls with the radius $r$, the volume $r^{d-1}$ of the shell at radius $r$ grows, and their product, the density of $\|\mathbf{x}\|$, peaks at $\sqrt{d-1}$, far from the origin where the density is largest; each curve is scaled to maximum 1. (b) The exact density of $\|\mathbf{x}\|$ for $\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ and $d=2,8,32,128,512$. The centers move out like $\sqrt{d}$; the widths stay near $1/\sqrt{2}$.](../img/mdl-prob-conc-shell.svg)
:label:`fig_mdl-conc-shell`

Notice that our bound captures the exponential scaling in $d$, though not the
exact numerical value. A norm below 14 at $d=784$ corresponds to
$\varepsilon=\tfrac12$, for which :eqref:`eq_mdl-norm-concentration` gives
$2e^{-784/32}=2e^{-24.5}\approx5\times10^{-11}$, compared with the exact
$10^{-110}$. The constant $8$ in the denominator is loose, but the qualitative
conclusion, exponential decay in $d$, is right on target.

### Near-Orthogonality of Random Directions

In :numref:`sec_mdl-geometry-linear-algebraic-ops`, we saw that the cosine of
the angle between a fixed unit vector and a random direction in $\mathbb{R}^d$
has variance $1/d$. Paired with Chebyshev's inequality, that gave a weak tail of
order $1/(d\varepsilon^2)$. Concentration sharpens this to an exponential tail,
which in turn lets us count how many random directions can be simultaneously
nearly orthogonal.

Random sign vectors provide the cleanest calculation. Fix any unit vector
$\mathbf u$ and draw independent uniform signs $x_i\in\{-1,+1\}$. Each term
$u_ix_i$ of the inner product $\langle\mathbf x,\mathbf u\rangle$ has mean zero
and lies in $[-|u_i|,|u_i|]$, an interval of length $2|u_i|$. Substituting these
ranges into our unequal-range Hoeffding bound :eqref:`eq_mdl-hoeffding-unequal`,
the denominator becomes $4\sum_i u_i^2=4$, leaving

$$
P\bigl(|\langle\mathbf x,\mathbf u\rangle| \geq s\bigr)
 \leq 2 e^{-s^2/2}.
$$
:eqlabel:`eq_mdl-inner-product-tail`

The inner product of a random sign vector with any fixed unit vector stays of
order one with a Gaussian tail, because the $\pm|u_i|$ contributions largely
cancel. Meanwhile, the norm of $\mathbf x$ is $\sqrt d$. Setting
$s=\varepsilon\sqrt d$ immediately gives
$P(|\cos(\mathbf x,\mathbf u)|\geq\varepsilon)\leq2e^{-d\varepsilon^2/2}$. For a
standard Gaussian vector $\mathbf x$, the same behavior holds: the inner product
$\langle\mathbf x,\mathbf u\rangle$ is itself a standard Gaussian for every unit
vector $\mathbf u$, so it obeys the exact same tail $2e^{-s^2/2}$, and the thin shell
keeps $\|\mathbf x\|$ within a few percent of $\sqrt d$.

With an exponential tail in hand, we can control *many* pairs at once using one
extra tool: the *union bound*, which states that the probability that at least
one of several events occurs is at most the sum of their individual
probabilities, regardless of whether those events are independent. Take $m$
independent random sign vectors. They form $m(m-1)/2<m^2/2$ distinct pairs, and
any single pair has $|\cos|\geq\varepsilon$ with probability at most
$2e^{-d\varepsilon^2/2}$. By the union bound, the probability that *any* pair
exceeds $\varepsilon$ is at most $m^2e^{-d\varepsilon^2/2}$. Consequently, with
probability at least $1-\delta$, all pairwise cosines stay below $\varepsilon$
whenever $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$. In high dimensions, you can
pack *exponentially* many nearly orthogonal vectors into $\mathbb{R}^d$.

Let us plug in $d=784$ and $\delta=0.05$. Even this loose bound guarantees
roughly 570 directions with all pairwise $|\cos|<0.2$, and roughly $10^7$
directions with all pairwise $|\cos|<0.3$. And those are worst-case guarantees
over all pairs; a single typical pair has $|\cos|$ much smaller, around
$\sqrt{2/(\pi d)}\approx0.028$, the mean absolute value of a Gaussian with
standard deviation $1/\sqrt d$.

Of course, learned neural network embeddings are not independent random
vectors, and that is precisely why this calculation is useful. At $d=784$, a
cosine similarity of $0.3$ sits more than eight standard deviations
($1/\sqrt{784}=0.036$) above what independent random directions produce. Random
directions thus serve as the null model against which genuine geometric
structure in trained embeddings is measured.

### The Thin Shell in Numbers
:label:`subsec_mdl-measuring-shell`

Because our theoretical bounds have loose constants, let us measure the true
numbers directly. Our tangent-line argument predicts that the standard deviation
of $\|\mathbf x\|$ stays near $1/\sqrt2\approx0.71$ at every dimension $d$,
while the Gaussian approximation predicts a mean $|\cos|$ of
$\sqrt{2/(\pi d)}$. Below, we draw 10,000 pairs of standard Gaussian vectors at
$d=2,20,200,2000$ to test both predictions; notice that one row will not match.

```{.python .input #mdl-distributions-multivariate-gaussian-1}
for d in (2, 20, 200, 2000):
    x, y = rng.standard_normal((2, 10000, d))
    nx = np.linalg.norm(x, axis=1)
    cos = (x * y).sum(1) / (nx * np.linalg.norm(y, axis=1))
    print(f'd = {d:4d}: mean ||x||/sqrt(d) = {(nx / np.sqrt(d)).mean():.3f}, '
          f'sd ||x|| = {nx.std():.2f}, mean |cos| = {np.abs(cos).mean():.4f}'
          f' vs {np.sqrt(2 / (np.pi * d)):.4f}')
```

As predicted, the standard deviation of the norm sits near $0.65$ at $d=2$ and
settles right around $0.7$ thereafter, even as the norm itself climbs from
roughly 1.3 to 45. The one row that does not match the asymptotic formula is
$d=2$, where the mean $|\cos|$ is about $0.64$: in the plane, the angle between
two isotropic vectors is uniform, giving $\mathbb{E}|\cos\theta|=2/\pi$, whereas
$\sqrt{2/(\pi d)}$ is a high-dimensional approximation. From $d=20$ onward, the
measured values (about $0.18$, $0.057$, and $0.018$) agree with the formula to
within a few percent.

Two follow-up questions make the geometry even more vivid. First, what fraction
of the probability mass lies within 10% of the typical norm, inside the shell
$(1\pm0.1)\sqrt d$? Second, do Euclidean distances still separate nearby points
from distant ones? To measure separation, we compute the *contrast* of a query
point, $(d_{\max}-d_{\min})/d_{\min}$ across its distances to 200 random points,
and report the median contrast over 100 queries.

```{.python .input #mdl-concentration-generalization-measuring-the-shell}
eps = 0.1
for d in (2, 20, 200, 2000):
    x = rng.standard_normal((10000, d))
    ratio = np.linalg.norm(x, axis=1) / np.sqrt(d)
    shell = np.mean(np.abs(ratio - 1) <= eps)          # mass in the 10% shell
    query, points = x[:100], x[100:300]                # 100 queries, 200 points
    dist = np.linalg.norm(query[:, None] - points[None], axis=2)  # (100, 200)
    contrast = (dist.max(1) - dist.min(1)) / dist.min(1)
    print(f'd = {d:4d}: mass in (1 ± 0.1) sqrt(d) = {shell:6.1%}, '
          f'median contrast = {np.median(contrast):.3f}')
```

At $d=2$, the 10% shell captures only about 15% of the mass; by $d=20$ it
captures roughly 47%, and by $d=200$ roughly 95%. That 95% matches our
tangent-line estimate: at $d=200$, the ratio $\|\mathbf x\|/\sqrt d$ has
standard deviation $1/\sqrt{2d}=0.05$, so $\pm0.1$ spans two standard
deviations on either side of the mean. By $d=2000$, all 10,000 draws land
inside the shell. Meanwhile, the median distance contrast collapses from around
30 at $d=2$ (fluctuating between roughly 25 and 40 from run to run) down to
about 1.2, 0.27, and finally 0.08. At $d=2000$, for a typical query, the
farthest of 200 random points is less than 10% farther away than the nearest.

### Consequences for Deep Learning
:label:`subsec_mdl-concentration-dl`

Three standard practices in deep learning rest directly on this high-dimensional
geometry.

**Initialization.** LeCun initialization :cite:`LeCun.Bottou.Orr.ea.1998`
samples the entries of a $d\times d$ weight matrix independently with variance
$1/d$, and He initialization :cite:`He.Zhang.Ren.ea.2015` uses variance $2/d$
for ReLU networks (:numref:`sec_numerical_stability`). The classical derivation
of these scales works in expectation over all random draws of the weights: at
LeCun's scale, each pre-activation $\langle\mathbf w,\mathbf x\rangle$ has
variance $\|\mathbf x\|^2/d$, which norm concentration keeps close to $1$ for
inputs with unit-variance coordinates. In practice, however, we initialize and
train a *single* network, not an average over infinitely many. Concentration
tells us when that expectation actually describes our one draw.

For a single He-initialized ReLU layer of width $d$, the ratio
$\rho=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$ of squared activation norms has
mean $1$ and variance $5/d$ (Exercise
:exref:`ex_mdl-concentration-generalization-depth-norms`), so one layer perturbs
the squared norm by roughly $\sqrt{5/d}$, or about 10% at $d=512$. Across $K$
layers, these multiplicative fluctuations add on a logarithmic scale, giving
$\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ a variance of roughly $5K/d$. The
expected-variance heuristic therefore describes a typical network only while
$K\ll d$. At $K=50$ and $d=512$, the log-variance reaches about $0.5$, meaning
the squared norm of a typical initialized network is already off by a factor of
two in either direction.

**Distances and nearest neighbors.** For two independent standard Gaussian
draws, the difference satisfies
$\mathbf x-\mathbf y\sim\mathcal N(\mathbf 0,2\mathbf I)$, so norm concentration
applies directly to $(\mathbf x-\mathbf y)/\sqrt2$: each pairwise distance falls
in $(1\pm\varepsilon)\sqrt{2d}$ except with probability $2e^{-d\varepsilon^2/8}$.
Applying the same union bound we used for directions places *all* pairwise
distances among $n$ points inside that narrow band simultaneously once
$d\geq(8/\varepsilon^2)\log(n^2/\delta)$, a threshold that grows only
logarithmically with $n$. When every distance is nearly the same, nearest and
farthest neighbors differ by only a small relative margin, just as our contrast
experiment showed. Real learned embeddings retain meaningful distance contrast
only because real data lie near a much lower-dimensional structure. On the
flip side, this same calculation yields a powerful tool: a random projection
down to $k$ dimensions preserves all pairwise distances among $n$ points once
$k$ is of order $\log n$, regardless of the ambient dimension (Exercise
:exref:`ex_mdl-concentration-generalization-johnson-lindenstrauss`).

**Latent codes.** A generative decoder trained on latent codes
$\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ almost exclusively sees inputs
whose norm is close to $\sqrt d$. Both the prior mean (norm $0$) and the
midpoint of the straight chord connecting two independent codes (norm close to
$\sqrt{d/2}$) lie deep in the empty interior where the decoder was never
trained. For this reason, practitioners typically interpolate between latent
codes along a spherical arc rather than a straight line (Exercise
:exref:`ex_mdl-concentration-generalization-latent-interpolation`).

One crucial limitation remains. Every bound in this section protects a single
quantity fixed *before* the data are drawn. A learning algorithm, by contrast,
inspects the data and picks the best model among many candidates *after* the
fact, and the best of many fair coins will always look biased. In
:numref:`sec_mdl-generalization-bounds`, we show how to extend concentration
inequalities to account for that selection.

## Summary

How far can an average of $n$ independent terms stray from its expectation at a
finite sample size $n$? Chebyshev's inequality gives a tail probability that
decays only like $1/n$, far too slowly when we need a guarantee that almost
never fails. The Chernoff method applies Markov's inequality to $e^{\lambda X}$,
turning the moment generating function of an independent sum into a product, and
then optimizes over $\lambda$. Combined with Hoeffding's lemma, this yields the
exponential tail $2e^{-2nt^2}$ for losses in $[0,1]$ and a finite-sample
confidence interval of half-width $\sqrt{\log(2/\delta)/(2n)}$; Bernstein's
inequality tightens this interval substantially whenever the variance is small
relative to the range. Applied to the $d$ coordinates of a random vector, the
same ideas show that a standard Gaussian concentrates on a thin shell of radius
$\sqrt d$ and thickness about $1/\sqrt2$, and that exponentially many random
directions are pairwise nearly orthogonal. Together, these results tell us how
to size test sets, when initialization heuristics hold for a single trained
network, and how to judge cosine similarities and distances in high-dimensional
embeddings.

## Exercises

1. **Sub-Gaussian random signs.** A random sign $\xi$ takes the values $-1$
   and $+1$ with probability $\tfrac12$ each, so that
   $\mathbb{E}[e^{\lambda\xi}]=\cosh\lambda\leq e^{\lambda^2/2}$.
    1. Conclude with the Chernoff method :eqref:`eq_mdl-chernoff` that the
       average $\bar\xi$ of $n$ independent random signs satisfies
       $P(|\bar\xi|\geq t)\leq 2e^{-nt^2/2}$.
    1. Hoeffding's lemma :eqref:`eq_mdl-hoeffding-lemma` with $[a,b]=[-1,1]$
       also gives a variance proxy for $\xi$. Compare it with the bound
       $\cosh\lambda\leq e^{\lambda^2/2}$. Expand
       $\log\cosh\lambda=\lambda^2/2-\lambda^4/12+\cdots$ around
       $\lambda=0$, and explain why no constant smaller than $\tfrac18$ can
       replace the $\tfrac18$ in :eqref:`eq_mdl-hoeffding-lemma` for all
       bounded variables.
   :exlabel:`ex_mdl-concentration-generalization-random-signs`

1. **Latent interpolation.** A generative model decodes latent codes
   $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ with $d=512$. Let
   $\mathbf z_1,\mathbf z_2$ be two independent codes and
   $\mathbf z(\gamma)=(1-\gamma)\mathbf z_1+\gamma\mathbf z_2$ for
   $\gamma\in[0,1]$ their linear interpolation.
    1. Find the distribution of $\mathbf z(\gamma)$ and the approximate
       value of $\|\mathbf z(\gamma)\|$. At which $\gamma$ is the norm
       smallest?
    1. Use :eqref:`eq_mdl-norm-concentration` to bound the probability that
       the midpoint $\mathbf z(\tfrac12)$ has norm at least $0.9\sqrt d$,
       that is, reaches the shell $(1\pm 0.1)\sqrt d$ in which typical codes
       lie. Then compute the exact probability, which is
       $P(\chi^2_{512}\geq 2\cdot0.81\cdot512)$ for a chi-square variable
       with 512 degrees of freedom (`scipy.stats.chi2.sf`). By how many
       orders of magnitude is the bound loose?
    1. Spherical interpolation :cite:`White.2016` replaces the chord by the
       arc

       $$\tilde{\mathbf z}(\gamma)=\frac{\sin((1-\gamma)\theta)}{\sin\theta}\,\mathbf z_1+\frac{\sin(\gamma\theta)}{\sin\theta}\,\mathbf z_2,$$

       where $\theta$ is the angle between $\mathbf z_1$ and $\mathbf z_2$.
       Show that $\|\tilde{\mathbf z}(\gamma)\|=r$ for all $\gamma$ when
       $\|\mathbf z_1\|=\|\mathbf z_2\|=r$. Which value does $\theta$ take
       approximately for $d=512$, and why?
   :exlabel:`ex_mdl-concentration-generalization-latent-interpolation`

1. [code] **Chernoff bound for coins.** ● Let $\hat p$ be the fraction of
   heads in $n$ independent flips of a coin with bias $p$, and let $a>p$.
    1. Apply the Chernoff method :eqref:`eq_mdl-chernoff` to $n\hat p$, whose
       moment generating function is $(1-p+pe^\lambda)^n$. Optimize over
       $\lambda$ and show that $P(\hat p\geq a)\leq e^{-n\,D_{\textrm{KL}}(a\|p)}$,
       where
       $D_{\textrm{KL}}(a\|p)=a\log\frac ap+(1-a)\log\frac{1-a}{1-p}$.
    1. Show that $D_{\textrm{KL}}(a\|p)\geq 2(a-p)^2$ by examining the
       second derivative in $a$, so that the bound of (a) is never weaker
       than one side of Hoeffding's bound. Evaluate both exponents for
       $p=\tfrac12$ and $a=0.6$.
    1. Add the two-sided bound $2e^{-n\,D_{\textrm{KL}}(0.6\|0.5)}$ as a
       column to the table of :numref:`subsec_mdl-tail-bounds-numerical`,
       for $n\in\{10,100,1000,5000\}$. The ratio of the exact tail to this
       bound still decreases with $n$. How does it scale with $n$, and what
       does the scaling say about the factor that the Chernoff method
       loses?
   :exlabel:`ex_mdl-concentration-generalization-chernoff-coins`

1. **Chi-square tail.** ● The proof of :eqref:`eq_mdl-norm-concentration`
   rests on the bound :eqref:`eq_mdl-chi2-mgf` for $x^2-1$ with
   $x\sim\mathcal N(0,1)$, which the proof states without derivation.
    1. Show that $\mathbb{E}[e^{\lambda(x^2-1)}]=e^{-\lambda}/\sqrt{1-2\lambda}$
       for $\lambda<\tfrac12$ by completing the square in the Gaussian
       integral.
    1. Expand $\log\mathbb{E}[e^{\lambda(x^2-1)}]$ into a power series in
       $2\lambda$ and prove :eqref:`eq_mdl-chi2-mgf`. The series starts at
       the quadratic term.
    1. The proof of :eqref:`eq_mdl-norm-concentration` takes
       $\lambda=\varepsilon/4$, which is admissible only for
       $\varepsilon\leq 1$. For $\varepsilon>1$, set $\lambda=\tfrac14$ and
       bound $P(\|\mathbf x\|^2\geq(1+\varepsilon)d)$. Does the result decay
       like $e^{-c\varepsilon^2}$ or like $e^{-c\varepsilon}$ in
       $\varepsilon$, and which regime of
       :eqref:`eq_mdl-subexponential-tail` does it belong to?
   :exlabel:`ex_mdl-concentration-generalization-chi-square-tail`

1. [code] **Activation norms through depth.** ● Consider a ReLU network of
   width $d$ and depth $K$ without biases,
   $\mathbf h_k=\mathrm{ReLU}(\mathbf W_k\mathbf h_{k-1})$, whose weights are
   independent draws from $\mathcal N(0,2/d)$ (He
   initialization, :numref:`sec_numerical_stability`). Let
   $\rho_k=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$. How such norms fluctuate
   with depth and width was analyzed by :citet:`Hanin.Nica.2020`.
    1. Show that, given $\mathbf h_{k-1}\ne\mathbf 0$, the ratio $\rho_k$ is
       distributed as $\frac2d\sum_{j=1}^d\mathrm{ReLU}(g_j)^2$ with
       independent $g_j\sim\mathcal N(0,1)$, whatever the direction of
       $\mathbf h_{k-1}$. Conclude that $\mathbb{E}[\rho_k]=1$ and
       $\textrm{Var}(\rho_k)=5/d$, using $\mathbb{E}[g^2]=1$ and
       $\mathbb{E}[g^4]=3$.
    1. Write $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ as a sum of $K$
       independent terms ($\rho_k=0$ has probability $2^{-d}$ per layer;
       ignore it). Expand $\log\rho$ to second order around $1$ and predict
       the mean and variance of this sum in terms of $K/d$.
    1. Simulate $1{,}000$ networks for each width $d\in\{32,128\}$ and depth
       $K\in\{1,10,50\}$, each with a fresh standard Gaussian input. Compare
       the mean and variance of $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$
       with (b), and report the mean and the median of the ratio
       $\|\mathbf h_K\|^2/\|\mathbf h_0\|^2$ itself. The variance calculation
       of :numref:`sec_numerical_stability` gives
       $\mathbb{E}[\|\mathbf h_K\|^2/\|\mathbf h_0\|^2]=1$ at every depth.
       Repeat the run at $K=50$ and $d=32$ several times. Why does the
       sample mean over $1{,}000$ networks scatter so widely around $1$? Why
       does the expectation not describe a typical network there?
   :exlabel:`ex_mdl-concentration-generalization-depth-norms`

1. [code] **Random projections preserve distances.** ● Let
   $\mathbf A\in\mathbb{R}^{k\times d}$ have independent $\mathcal N(0,1/k)$
   entries.
    1. Show that for a fixed $\mathbf v\neq\mathbf 0$ the ratio
       $\|\mathbf A\mathbf v\|^2/\|\mathbf v\|^2$ has the distribution of
       $\|\mathbf g\|^2/k$ with $\mathbf g\sim\mathcal N(\mathbf 0,\mathbf I_k)$,
       so that by :eqref:`eq_mdl-norm-concentration`
       $P(|\|\mathbf A\mathbf v\|/\|\mathbf v\|-1|\geq\varepsilon)\leq
       2e^{-k\varepsilon^2/8}$. Hint: rotate so that $\mathbf v$ lies
       along the first axis.
    1. For $n$ points, apply (a) to the $n(n-1)/2$ differences and a union
       bound. Show that all pairwise distances are preserved within a
       factor $1\pm\varepsilon$ with probability at least $1-\delta$ once
       $k\geq(8/\varepsilon^2)\log(n^2/\delta)$, a bound that does not
       depend on $d$. This is the Johnson–Lindenstrauss lemma. Evaluate it
       for $n=1{,}000$, $\varepsilon=0.3$ and $\delta=0.05$.
    1. Verify it on $1{,}000$ standard Gaussian points in $d=2{,}000$.
       Project to $k\in\{50,200,800\}$ and report the largest distortion
       $|\|\mathbf A(\mathbf x_i-\mathbf x_j)\|/\|\mathbf x_i-\mathbf x_j\|-1|$
       over all pairs, together with the typical fluctuation
       $1/\sqrt{2k}$. Compute the pairwise distances from the Gram matrix,
       $\|\mathbf x_i-\mathbf x_j\|^2=\|\mathbf x_i\|^2+\|\mathbf x_j\|^2-2\,
       \mathbf x_i^\top\mathbf x_j$, not by broadcasting a
       $1{,}000\times1{,}000\times2{,}000$ array. Explain the ratio between
       the two numbers from the number of pairs. Already at $k=200$ the
       largest distortion is below $\varepsilon=0.3$, while (b) asked for a
       much larger $k$. Which part of the bound is loose: the constant 8 or
       the $\log n$?
   :exlabel:`ex_mdl-concentration-generalization-johnson-lindenstrauss`

1. **Bernstein's inequality.** ● Let $X$ be centered with $|X|\leq M$ and
   $\textrm{Var}(X)=\sigma^2$.
    1. Show that $\mathbb{E}[X^k]\leq\sigma^2M^{k-2}$ for $k\geq2$, and
       hence that
       $\mathbb{E}[e^{\lambda X}]\leq1+\sigma^2(e^{\lambda M}-1-\lambda M)/M^2$
       for $\lambda>0$.
    1. Show that $e^u-1-u\leq u^2/(2(1-u/3))$ for $0\leq u<3$ by comparing
       $u^k/k!$ with $(u^2/2)(u/3)^{k-2}$, that is, by using
       $k!\geq2\cdot3^{k-2}$. Conclude with $1+y\leq e^y$ that
       $\mathbb{E}[e^{\lambda X}]\leq\exp(\lambda^2\sigma^2/(2(1-\lambda M/3)))$
       for $0<\lambda<3/M$.
    1. Apply the Chernoff method to an average of $n$ such variables with
       $\lambda=t/c$, where $c=\sigma^2+Mt/3$. Check that $\lambda<3/M$ and
       that $1-\lambda M/3=\sigma^2/c$, and conclude that the exponent is
       exactly $-nt^2/(2c)$, which proves :eqref:`eq_mdl-bernstein`.
    1. Show that (b) makes $X$ sub-exponential in the sense of
       :eqref:`eq_mdl-subexponential` with $\nu^2=2\sigma^2$ and
       $\alpha=2M/3$.
    1. With $n=10{,}000$ and $\delta=0.05$, show that Bernstein's bound is
       weaker than Hoeffding's at the worst-case variance $\sigma^2=\tfrac14$
       of a 0/1 outcome. Below which variance does it win, and which error
       rates does that correspond to?
   :exlabel:`ex_mdl-concentration-generalization-bernstein`

[Discussions](https://d2l.discourse.group/t/36834)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §36.9]{.kicker}

Concentration Inequalities and High-Dimensional Measure<br>**Chernoff method · Hoeffding & Bernstein bounds · sub-Gaussian & sub-exponential tails · thin Gaussian shell · near-orthogonality**
:::
:::

::: {.slide title="830 Errors in 10,000 Images: Can We Trust the Average?"}
[Motivation & Core Problem]{.kicker}

Suppose a trained classifier makes $830$ mistakes on $n = 10{,}000$ held-out test images, giving an empirical error rate of $\hat{p} = 8.3\%$. How far can the true population error stray from $8.3\%$?

- **Asymptotic Gaussian CLT ($\pm 1.96\,\hat{\sigma}/\sqrt{n} = \pm 0.54\%$):** Convenient and narrow, but exact only as $n \to \infty$ and reliant on plugging in the estimated standard deviation $\hat{\sigma}$.
- **Chebyshev's finite-sample bound ($\sigma^2 / (nt^2) \le \delta$):** Using $\sigma^2 \le 1/4$ gives $\pm 2.2\%$ at $95\%$ confidence ($\delta = 0.05$), and collapses to a useless **$\pm 500\%$** when we demand one-in-a-million reliability ($\delta = 10^{-6}$).
- **Exponential concentration (Hoeffding & Bernstein):** Capturing exponential tail decay in $n$ certifies **$\pm 1.36\%$** at $95\%$ confidence and **$\pm 2.7\%$** at $\delta = 10^{-6}$ (tightened further by Bernstein to **$\pm 0.79\%$**).

::: {.d2l-note}
Why care about $\delta = 10^{-6}$? When a learner compares thousands of candidate models, we need *none* of them to look good by chance—and only exponential tail bounds make union bounds affordable.
:::
:::

::: {.slide title="Why Chebyshev's Polynomial Tail Falls Short"}
[Polynomial vs. Exponential Decay]{.kicker}

Applying Markov's inequality to the squared deviation $(\bar{X} - \mu)^2$ yields Chebyshev's inequality $P(\lvert\bar{X} - \mu\rvert \ge t) \le \frac{\sigma^2}{nt^2}$, which decays only like $1/n$. Comparing Chebyshev's $25/n$ against the exact binomial tail $P(\lvert\hat{p} - 0.5\rvert \ge 0.1)$ for $n$ fair coin flips exposes the gap:

@!mdl-concentration-generalization-chebyshev-s-rate-2

- **Eight orders of magnitude loose at $n = 1{,}000$:** The exact tail is $2.7 \times 10^{-10}$ while Chebyshev allows $2.5 \times 10^{-2}$; by $n = 5{,}000$, the gap reaches $42$ orders of magnitude.
- **Why two moments are not enough:** Chebyshev is tight for discrete three-point distributions; beating $1/n$ requires harnessing *all* higher moments simultaneously via the moment generating function.
:::

::: {.slide title="The Chernoff Method: Exponential Majorants"}
[Harnessing All Moments at Once]{.kicker}

::: {.cols .vc}
::: {.col}
- **Majorizing the step indicator:** Any nonnegative function $g(x) \ge \mathbf{1}\{x \ge t\}$ satisfies $P(X \ge t) \le \mathbb{E}[g(X)]$. Markov uses $x/t$, Chebyshev uses $(x/t)^2$, and Chernoff uses $e^{\lambda(x-t)}$ for $\lambda > 0$:
  $$P(X \ge t) = P\bigl(e^{\lambda X} \ge e^{\lambda t}\bigr) \le \inf_{\lambda > 0}\, e^{-\lambda t}\,\mathbb{E}\bigl[e^{\lambda X}\bigr]$$
- **Independent sums factorize into products:** For independent $X_1, \dots, X_n$ with $S = \sum_{i=1}^n X_i$, $\mathbb{E}[e^{\lambda S}] = \prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]$, turning the exponent into a sum of $n$ terms.
:::

::: {.col .fig}
@fig:mdl-prob-conc-majorants
:::
:::
:::

::: {.slide title="Worked Derivation: The Fair Coin Exponential Tail"}
[Chernoff on Centered Coin Flips]{.kicker}

For a single fair coin centered at zero, $Z_i = X_i - \frac{1}{2} \in \{-\frac{1}{2}, +\frac{1}{2}\}$ with equal probability, its moment generating function is $\mathbb{E}[e^{\lambda Z_i}] = \frac{1}{2}(e^{\lambda/2} + e^{-\lambda/2}) = \cosh(\lambda/2)$:

1. **Taylor series MGF bound:** Since $(2k)! \ge 2 \cdot 4 \cdots (2k) = 2^k k!$, comparing power series term-by-term gives:
   $$\cosh u = \sum_{k=0}^\infty \frac{u^{2k}}{(2k)!} \le \sum_{k=0}^\infty \frac{(u^2/2)^k}{k!} = e^{u^2/2} \implies \mathbb{E}\bigl[e^{\lambda Z_i}\bigr] \le e^{\lambda^2 / 8}$$
2. **Chernoff bound on $n$ flips:** By independence, $P\bigl(\hat{p} - \frac{1}{2} \ge t\bigr) = P\bigl(\sum_{i=1}^n Z_i \ge nt\bigr) \le e^{-\lambda n t}\prod_{i=1}^n \mathbb{E}[e^{\lambda Z_i}] \le \exp\bigl(-\lambda nt + \frac{n\lambda^2}{8}\bigr)$.
3. **Optimize over $\lambda > 0$:** The parabola $-\lambda nt + \frac{n\lambda^2}{8}$ reaches its minimum at $\lambda^* = 4t$, giving $e^{-2nt^2}$ for the upper tail and $P\bigl(\lvert\hat{p} - \frac{1}{2}\rvert \ge t\bigr) \le 2e^{-2nt^2}$ for both tails.
:::

::: {.slide title="Hoeffding's Lemma for General Bounded Variables"}
[Bounding the Cumulant Generating Function]{.kicker}

What if $X$ is an asymmetric or continuous variable confined to $[a, b]$ with $\mathbb{E}[X] = 0$? **Hoeffding's lemma** proves that its MGF never exceeds that of a Gaussian with standard deviation $(b-a)/2$:

$$\mathbb{E}\bigl[e^{\lambda X}\bigr] \le \exp\!\left(\frac{\lambda^2(b-a)^2}{8}\right) \qquad \text{for all } \lambda \in \mathbb{R}$$

- **Step 1 (Maximum variance on $[a,b]$):** For any $Y \in [a,b]$, centering at the midpoint $c = \frac{a+b}{2}$ gives $\operatorname{Var}(Y) \le \mathbb{E}\bigl[\bigl(Y - \frac{a+b}{2}\bigr)^2\bigr] \le \frac{(b-a)^2}{4}$.
- **Step 2 (Curvature of the log-MGF):** Let $\psi(\lambda) = \log \mathbb{E}[e^{\lambda X}]$. Then $\psi(0) = 0$, $\psi'(0) = \mathbb{E}[X] = 0$, and $\psi''(\lambda) = \operatorname{Var}_{q_\lambda}(X)$, where $q_\lambda(dx) \propto e^{\lambda x}p(dx)$ is the exponentially tilted distribution—still supported inside $[a,b]$, so $\psi''(\lambda) \le \frac{(b-a)^2}{4}$ everywhere.
- **Step 3 (Second-order Taylor bound):** $\psi(\lambda) = \psi(0) + \lambda\psi'(0) + \frac{\lambda^2}{2}\psi''(\xi) \le \frac{\lambda^2(b-a)^2}{8}$.
:::

::: {.slide title="Hoeffding's Inequality for Bounded Independent Sums"}
[Finite-Sample Tail Guarantee]{.kicker}

Applying the Chernoff method with Hoeffding's lemma to independent random variables $X_1, \dots, X_n \in [a, b]$ and minimizing $-\lambda nt + \frac{n\lambda^2(b-a)^2}{8}$ at $\lambda^* = \frac{4t}{(b-a)^2}$ establishes **Hoeffding's inequality**:

$$P\bigl(\lvert\bar{X} - \mathbb{E}[\bar{X}]\rvert \ge t\bigr) \le 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)$$

- **Zero distributional assumptions beyond boundedness and independence:** Holds at every finite sample size $n$ without knowing the true variance or invoking the Central Limit Theorem.
- **Unequal-range extension:** When independent centered variables satisfy $Y_i \in [a_i, b_i]$, the individual MGF bounds multiply to replace $n(b-a)^2$ with $\sum_{i=1}^n (b_i - a_i)^2$:
  $$P\Biggl(\Biggl\lvert\sum_{i=1}^n Y_i\Biggr\rvert \ge s\Biggr) \le 2\exp\!\left(-\frac{2s^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$
:::

::: {.slide title="Sizing a Test Set with Hoeffding's Confidence Interval"}
[Sample Complexity of Evaluation]{.kicker}

Setting the failure probability $2\exp\bigl(-\frac{2nt^2}{(b-a)^2}\bigr) = \delta$ and solving for either the half-width $t$ or the sample size $n$ provides a strict finite-sample guarantee with probability at least $1 - \delta$:

$$\bigl\lvert\bar{X} - \mathbb{E}[\bar{X}]\bigr\rvert \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}, \qquad n \ge \frac{(b-a)^2\log(2/\delta)}{2t^2}$$

- **Benchmark test-set sizing ($[a,b] = [0,1]$):** Certifying an error rate within $t = \pm 0.01$ ($1\%$) at $95\%$ confidence ($\delta = 0.05$) requires $n = \frac{\log 40}{2 \cdot 10^{-4}} = 18{,}445$ test examples (roughly twice the asymptotic Gaussian figure of $9{,}604$).
- **Logarithmic cost of extreme confidence:** Raising reliability from $\delta = 0.05$ to $\delta = 10^{-6}$ increases $\log(2/\delta)$ by less than $4\times$ ($n = 72{,}413$ examples), whereas Chebyshev's $1/\delta$ scaling would demand $2.5 \times 10^9$ examples.
:::

::: {.slide title="The Tail Race in Code: Exact Binomial vs. Hoeffding"}
[Empirical Verification]{.kicker}

::: {.cols .vc}
::: {.col}
- **Parallel exponential decay:** On a logarithmic axis, Hoeffding's bound $2e^{-2nt^2}$ and the exact binomial tail trace nearly parallel straight lines.
- **Per-flip decay rate:** Hoeffding drops by $2t^2 = 0.0200$ nats per flip, matching the exact Cramér–Chernoff KL rate $D_{\mathrm{KL}}(0.6 \,\|\, 0.5) = 0.0201$ nats per flip to within $0.5\%$.
- **Small-$n$ regime ($n \le 13$):** For $9$ heads in $13$ flips ($t = 2.5/13$), Hoeffding gives $0.77$ and the exact tail is $0.27$—no inequality can manufacture evidence from $13$ flips.
:::

::: {.col .fig}
@!mdl-concentration-generalization-the-tail-race-in-code
:::
:::
:::

::: {.slide title="Sub-Gaussian Random Variables and Variance Proxies"}
[Generalizing Beyond Bounded Support]{.kicker}

Hoeffding's proof required only that the centered MGF be bounded by a Gaussian MGF. A random variable $X$ is **sub-Gaussian** with *variance proxy* $\sigma^2$ if:

$$\mathbb{E}\bigl[e^{\lambda(X - \mathbb{E}[X])}\bigr] \le \exp\!\left(\frac{\lambda^2 \sigma^2}{2}\right) \qquad \text{for all } \lambda \in \mathbb{R}$$

- **Canonical sub-Gaussian families:** (1) $\mathcal{N}(\mu, \sigma^2)$ has proxy $\sigma^2$ with equality; (2) any bounded variable $X \in [a,b]$ has proxy $\sigma^2 = (b-a)^2/4$ by Hoeffding's lemma; (3) a Rademacher sign $\xi \in \{-1, +1\}$ has proxy $\sigma^2 = 1$.
- **Preservation under averaging:** If $X_1, \dots, X_n$ are independent and $\sigma^2$-sub-Gaussian, then $\bar{X}$ is $(\sigma^2/n)$-sub-Gaussian and satisfies $P(\lvert\bar{X} - \mathbb{E}[\bar{X}]\rvert \ge t) \le 2e^{-nt^2/(2\sigma^2)}$.
- **Why $\sigma^2$ is a proxy:** Expanding both sides to $O(\lambda^2)$ shows $\operatorname{Var}(X) \le \sigma^2$; for a $0/1$ loss with $2\%$ error, $\operatorname{Var}(X) = 0.0196$ is $12.7\times$ smaller than Hoeffding's worst-case proxy $1/4$.
:::

::: {.slide title="Why Squared Gaussians Are Only Sub-Exponential"}
[Two-Regime MGF Bounds]{.kicker}

To analyze the squared norm $\|\mathbf{x}\|^2 = \sum_{i=1}^d x_i^2$ of a Gaussian vector, we need the MGF of $x_i^2 - 1$ for $x_i \sim \mathcal{N}(0, 1)$. Completing the square in the Gaussian integral gives $\mathbb{E}[e^{\lambda(x_i^2 - 1)}] = \frac{e^{-\lambda}}{\sqrt{1 - 2\lambda}}$ for $\lambda < \frac{1}{2}$, which **diverges to $+\infty$** for $\lambda \ge \frac{1}{2}$!

- **Sub-exponential definition:** A variable $X$ is $(\nu, \alpha)$-**sub-exponential** if its MGF is bounded by a Gaussian MGF on a restricted window around the origin:
  $$\mathbb{E}\bigl[e^{\lambda(X - \mathbb{E}[X])}\bigr] \le \exp\!\left(\frac{\lambda^2 \nu^2}{2}\right) \qquad \text{for all } \lvert\lambda\rvert \le \frac{1}{\alpha}$$
- **Two-regime Chernoff tail bound:** Optimizing $-\lambda nt + \frac{n\lambda^2\nu^2}{2}$ over $0 < \lambda \le 1/\alpha$ picks the unconstrained vertex $\lambda^* = t/\nu^2$ for small deviations ($t \le \nu^2/\alpha$) and clamps at the boundary $\lambda^* = 1/\alpha$ for large deviations ($t > \nu^2/\alpha$):
  $$P\bigl(\lvert\bar{X} - \mathbb{E}[\bar{X}]\rvert \ge t\bigr) \le 2\exp\!\left(-\frac{n}{2}\min\!\left(\frac{t^2}{\nu^2},\, \frac{t}{\alpha}\right)\right)$$
:::

::: {.slide title="Bernstein's Inequality: Exploiting Small Variance"}
[Variance-Aware Concentration]{.kicker}

When bounded variables $\lvert X_i - \mathbb{E}[X_i]\rvert \le M$ also have small variance $\operatorname{Var}(X_i) \le \sigma^2 \ll M^2$, **Bernstein's inequality** tracks both moments to bridge the sub-Gaussian and sub-exponential regimes:

$$P\bigl(\lvert\bar{X} - \mathbb{E}[\bar{X}]\rvert \ge t\bigr) \le 2\exp\!\left(-\frac{nt^2}{2\bigl(\sigma^2 + Mt/3\bigr)}\right)$$

@!mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables-1

- **Closing the gap to the CLT:** At a $2\%$ error rate on $n = 10{,}000$ images, solving the fixed-point equation with worst-case true variance $(\hat{p}+t)(1-\hat{p}-t)$ gives a strictly guaranteed half-width of **$\pm 0.43\%$**—more than $3\times$ tighter than Hoeffding's $\pm 1.36\%$.
:::

::: {.slide title="High-Dimensional Paradox: The Empty Mode of a Gaussian"}
[Concentration of Measure in R^d]{.kicker}

::: {.cols .vc}
::: {.col}
- **Where does $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{784})$ live?** Although the density $(2\pi)^{-d/2}e^{-\|\mathbf{x}\|^2/2}$ is highest at the origin $\mathbf{0}$, the probability of landing within distance $14$ of the origin is $\approx 10^{-110}$!
- **Density vs. spherical volume:** Integrating over spherical shells of radius $r = \|\mathbf{x}\|$ multiplies the decaying density $e^{-r^2/2}$ by the exploding surface area $\propto r^{d-1}$:
  $$p(r) \propto r^{d-1}e^{-r^2/2}$$
- **The thin shell:** The radial density peaks at $r_{\mathrm{mode}} = \sqrt{d - 1} \approx \sqrt{d}$. At $d = 784$, virtually all probability mass sits in a thin shell at radius $\sqrt{784} = 28.0 \pm 0.7$.
:::

::: {.col .fig}
@fig:mdl-prob-conc-shell
:::
:::
:::

::: {.slide title="Theorem: Concentration of the Euclidean Norm"}
[The Thin Shell Bound]{.kicker}

Viewing $\frac{1}{d}\|\mathbf{x}\|^2 = \frac{1}{d}\sum_{i=1}^d x_i^2$ as a sample average of $d$ independent squared Gaussians turns high-dimensional geometry into a concentration inequality:

$$P\!\left(\Biggl\lvert\frac{\|\mathbf{x}\|}{\sqrt{d}} - 1\Biggr\rvert \ge \varepsilon\right) \le 2\exp\!\left(-\frac{d\varepsilon^2}{8}\right) \qquad \text{for all } 0 < \varepsilon \le 1$$

1. **Sub-exponential coordinates:** Power-series expansion shows $\mathbb{E}[e^{\lambda(x_i^2 - 1)}] = \frac{e^{-\lambda}}{\sqrt{1-2\lambda}} \le e^{2\lambda^2}$ for $\lvert\lambda\rvert \le \frac{1}{4}$, so each $x_i^2$ is $(\nu = 2, \alpha = 4)$-sub-exponential.
2. **Concentration of the squared norm:** Applying our sub-exponential tail bound with $n = d$ and $t = \varepsilon \le 1$ gives $P\bigl(\bigl\lvert\frac{\|\mathbf{x}\|^2}{d} - 1\bigr\rvert \ge \varepsilon\bigr) \le 2\exp\bigl(-\frac{d}{2}\min(\frac{\varepsilon^2}{4}, \frac{\varepsilon}{4})\bigr) = 2e^{-d\varepsilon^2/8}$.
3. **From squared norm to norm:** Writing $z = \|\mathbf{x}\|/\sqrt{d} \ge 0$, whenever $\lvert z - 1\rvert \ge \varepsilon$ we have $\lvert z^2 - 1\rvert = \lvert z - 1\rvert(z + 1) \ge \varepsilon$ since $z + 1 \ge 1$.
:::

::: {.slide title="Constant Shell Thickness Across All Dimensions"}
[O(1) Width at Radius sqrt(d)]{.kicker}

How thick is the shell around radius $\sqrt{d}$? Because $\mathbb{E}[x_i^4] = 3$, each squared coordinate has variance $\operatorname{Var}(x_i^2) = 3 - 1 = 2$, so $\operatorname{Var}(\|\mathbf{x}\|^2) = 2d$. Linearizing the square root around $d$ shows that the shell thickness is independent of dimension:

$$\|\mathbf{x}\| = \sqrt{\|\mathbf{x}\|^2} \approx \sqrt{d} + \frac{\|\mathbf{x}\|^2 - d}{2\sqrt{d}} \implies \operatorname{sd}\bigl(\|\mathbf{x}\|\bigr) \approx \frac{\sqrt{2d}}{2\sqrt{d}} = \frac{1}{\sqrt{2}} \approx 0.71$$

@!mdl-distributions-multivariate-gaussian-1
:::

::: {.slide title="Near-Orthogonality of High-Dimensional Random Vectors"}
[Angles in High Dimension]{.kicker}

What is the angle between a fixed unit vector $\mathbf{u} \in \mathbb{R}^d$ ($\|\mathbf{u}\|_2 = 1$) and a random sign vector $\mathbf{x} \in \{-1, +1\}^d$ (or a standard Gaussian $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$)?

- **Inner product stays $O(1)$:** Each term $Y_i = u_i x_i$ has mean zero and lies in $[-\lvert u_i\rvert, \lvert u_i\rvert]$ of width $2\lvert u_i\rvert$. Since $\sum_{i=1}^d (2\lvert u_i\rvert)^2 = 4\|\mathbf{u}\|_2^2 = 4$, unequal-range Hoeffding gives:
  $$P\bigl(\lvert\langle\mathbf{x}, \mathbf{u}\rangle\rvert \ge s\bigr) \le 2e^{-s^2 / 2}$$
- **Cosine concentrates around zero at rate $1/\sqrt{d}$:** Since $\|\mathbf{x}\|_2 = \sqrt{d}$, setting $s = \varepsilon\sqrt{d}$ proves:
  $$P\bigl(\lvert\cos(\mathbf{x}, \mathbf{u})\rvert \ge \varepsilon\bigr) \le 2e^{-d\varepsilon^2 / 2}, \qquad \mathbb{E}\bigl[\lvert\cos(\mathbf{x}, \mathbf{y})\rvert\bigr] \approx \sqrt{\frac{2}{\pi d}}$$
- **Null baseline for embeddings:** In $d = 784$, two independent random vectors have typical cosine similarity $\sqrt{2/(784\pi)} \approx 0.028$ ($\mathrm{sd} = 1/\sqrt{784} = 0.036$), so a learned cosine similarity of $0.30$ sits more than $8$ standard deviations above random chance.
:::

::: {.slide title="Packing Exponentially Many Quasi-Orthogonal Vectors"}
[The Union Bound in R^d]{.kicker}

In $\mathbb{R}^d$, strictly orthogonal vectors can number at most $d$. How many *nearly* orthogonal vectors ($\lvert\cos(\mathbf{x}_i, \mathbf{x}_j)\rvert < \varepsilon$ for all pairs $i \neq j$) can we pack simultaneously?

- **Applying the union bound over all $\binom{m}{2} < \frac{m^2}{2}$ pairs:** Drawing $m$ independent random sign vectors in $\mathbb{R}^d$, the probability that *any* pair exceeds cosine similarity $\varepsilon$ is bounded by the sum of pairwise failure probabilities:
  $$P\Bigl(\max_{i < j} \lvert\cos(\mathbf{x}_i, \mathbf{x}_j)\rvert \ge \varepsilon\Bigr) \le \frac{m(m-1)}{2}\cdot 2e^{-d\varepsilon^2 / 2} < m^2 e^{-d\varepsilon^2 / 2}$$
- **Exponential packing capacity:** Setting $m^2 e^{-d\varepsilon^2/2} \le \delta$ guarantees that all $\binom{m}{2}$ pairs have $\lvert\cos\rvert < \varepsilon$ with probability at least $1 - \delta$ whenever:
  $$m \le \sqrt{\delta}\,\exp\!\left(\frac{d\varepsilon^2}{4}\right)$$
- **Concrete capacity at $d = 784, \delta = 0.05$:** Even this worst-case union bound certifies $m \approx 570$ vectors with all pairwise $\lvert\cos\rvert < 0.2$, and **$m \approx 10^7$ vectors** with all pairwise $\lvert\cos\rvert < 0.3$.
:::

::: {.slide title="Collapse of Pairwise Euclidean Distance Contrast"}
[Curse of Dimensionality for Nearest Neighbors]{.kicker}

For two independent Gaussian vectors $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, their difference follows $\mathbf{x} - \mathbf{y} \sim \mathcal{N}(\mathbf{0}, 2\mathbf{I}_d)$, so $\|\mathbf{x} - \mathbf{y}\| / \sqrt{2d}$ concentrates around $1$ with tail $2e^{-d\varepsilon^2/8}$:

@!mdl-concentration-generalization-measuring-the-shell

- **All random points become equidistant:** By a union bound over $\binom{n}{2}$ pairs, all pairwise distances among $n$ random points lie inside $(1 \pm \varepsilon)\sqrt{2d}$ once $d \ge \frac{8}{\varepsilon^2}\log(n^2/\delta)$. At $d = 2000$, the farthest of $200$ random neighbors is only $8\%$ farther away than the nearest!
:::

::: {.slide title="Consequence 1: Weight Initialization Across Depth"}
[LeCun & He Initialization Fluctuations]{.kicker}

Classical initialization schemes choose weight variance $\frac{1}{d}$ (LeCun) or $\frac{2}{d}$ (He for ReLU) so that expected activation energy is preserved ($\mathbb{E}[\|\mathbf{h}_k\|^2 / \|\mathbf{h}_{k-1}\|^2] = 1$). Concentration reveals when a *single* initialized network actually obeys that expectation:

- **Single-layer norm fluctuation:** For a width-$d$ ReLU layer $\mathbf{h}_k = \mathrm{ReLU}(\mathbf{W}_k \mathbf{h}_{k-1})$ with $W_{ij} \sim \mathcal{N}(0, 2/d)$, the squared-norm multiplier $\rho_k = \|\mathbf{h}_k\|^2 / \|\mathbf{h}_{k-1}\|^2$ is distributed as $\frac{2}{d}\sum_{j=1}^d \mathrm{ReLU}(g_j)^2$ ($g_j \sim \mathcal{N}(0,1)$), which has mean $1$ and variance:
  $$\operatorname{Var}(\rho_k) = \frac{5}{d} \implies \operatorname{sd}(\rho_k) = \sqrt{\frac{5}{d}} \approx 10\% \text{ at } d = 512$$
- **Accumulation across depth $K$:** Across $K$ independent layers, $\log\frac{\|\mathbf{h}_K\|^2}{\|\mathbf{h}_0\|^2} = \sum_{k=1}^K \log \rho_k$ accumulates variance $\approx \frac{5K}{d}$.
- **Aspect ratio rule ($K \ll d$):** Initialization preserves norms in a single network only when depth $K \ll$ width $d$; at $K = 50, d = 512$, $\frac{5K}{d} \approx 0.5$, so activation norms already drift by $2\times$ before step $1$ of training.
:::

::: {.slide title="Johnson–Lindenstrauss and Spherical Latent Arcs"}
[Random Projections & Generative Interpolation]{.kicker}

High-dimensional norm concentration directly powers two foundational algorithms in representation learning and generative modeling:

- **1. Johnson–Lindenstrauss Random Projections:** Multiplying $n$ points in $\mathbb{R}^d$ by a random projection matrix $\mathbf{A} \in \mathbb{R}^{k \times d}$ with entries $\mathcal{N}(0, 1/k)$ preserves every pairwise Euclidean distance within $1 \pm \varepsilon$ with probability $\ge 1 - \delta$ once:
  $$k \ge \frac{8}{\varepsilon^2}\log\frac{n^2}{\delta}$$
  completely independent of the ambient dimension $d$.
- **2. Spherical Interpolation (Slerp) in Generative Latent Spaces:** A decoder trained on $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ only ever sees latent codes on the thin shell of radius $\sqrt{d}$. Because two independent codes $\mathbf{z}_1, \mathbf{z}_2$ are nearly orthogonal ($\theta \approx \pi/2$), their linear midpoint $\frac{1}{2}(\mathbf{z}_1 + \mathbf{z}_2) \sim \mathcal{N}(\mathbf{0}, \frac{1}{2}\mathbf{I}_d)$ has norm $\approx \sqrt{d/2}$—plunging into the empty interior where the decoder was never trained. Interpolating along the great-circle arc $\tilde{\mathbf{z}}(\gamma) = \frac{\sin((1-\gamma)\theta)}{\sin\theta}\mathbf{z}_1 + \frac{\sin(\gamma\theta)}{\sin\theta}\mathbf{z}_2$ keeps $\|\tilde{\mathbf{z}}(\gamma)\| \approx \sqrt{d}$ for all $\gamma \in [0, 1]$.
:::

::: {.slide title="Pitfall: Single-Average Bounds Fail After Selection"}
[The Bridge to Uniform Convergence]{.kicker}

Every concentration inequality in this section—Hoeffding, Bernstein, and sub-exponential norm bounds—rests on one crucial hypothesis:

::: {.d2l-note .warn}
**The Fixed-Variable Hypothesis:** The random variable being averaged (the classifier $f$ on a test set, or the vector $\mathbf{v}$ being projected) must be chosen and fixed **before** the random sample $S$ is drawn.
:::

- **What happens during training and model selection?** A learning algorithm inspects the dataset $S$, evaluates thousands or infinitely many candidate functions $f \in \mathcal{F}$, and selects the empirical winner $\hat{f} = \operatorname{argmin}_{f \in \mathcal{F}} \hat{R}_S(f)$.
- **Selection bias of the minimum:** Even if every candidate has true risk $R(f) = 0.5$, the best of $1{,}000$ empirical averages on $n = 100$ examples systematically lands near $0.34$. Certifying a data-dependent choice $\hat{f}$ requires **uniform convergence** over the entire class $\mathcal{F}$ (§36.10).
:::

::: {.slide title="Comparing Concentration Tail Bounds"}
[Markov, Chebyshev, Hoeffding, Bernstein, Sub-Exponential]{.kicker}

Each concentration inequality trades stronger moment or boundedness assumptions for dramatically sharper tail decay:

| Inequality | Assumptions on Independent $X_1, \dots, X_n$ | Tail Bound on $P(\lvert\bar{X} - \mu\rvert \ge t)$ | Scaling of $n$ with Failure $\delta$ |
| :--- | :--- | :--- | :--- |
| **Markov** | Nonnegative $X \ge 0$ (single variable) | $P(X \ge t) \le \frac{\mathbb{E}[X]}{t}$ | Linear in $1/\delta$ |
| **Chebyshev** | Finite variance $\operatorname{Var}(X_i) \le \sigma^2$ | $\frac{\sigma^2}{nt^2}$ | Polynomial: $n \propto \frac{1}{\delta t^2}$ |
| **Hoeffding** | Bounded range $X_i \in [a, b]$ | $2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)$ | Logarithmic: $n \propto \frac{(b-a)^2\log(2/\delta)}{t^2}$ |
| **Sub-Gaussian** | $\mathbb{E}[e^{\lambda(X_i-\mu)}] \le e^{\lambda^2\sigma^2/2}$ ($\forall \lambda \in \mathbb{R}$) | $2\exp\!\left(-\frac{nt^2}{2\sigma^2}\right)$ | Logarithmic: $n \propto \frac{\sigma^2\log(2/\delta)}{t^2}$ |
| **Sub-Exponential** | $\mathbb{E}[e^{\lambda(X_i-\mu)}] \le e^{\lambda^2\nu^2/2}$ ($\lvert\lambda\rvert \le 1/\alpha$) | $2\exp\!\left(-\frac{n}{2}\min\!\left(\frac{t^2}{\nu^2},\, \frac{t}{\alpha}\right)\right)$ | Gaussian for $t \le \frac{\nu^2}{\alpha}$; exponential for $t > \frac{\nu^2}{\alpha}$ |
| **Bernstein** | $\lvert X_i - \mu\rvert \le M$ and $\operatorname{Var}(X_i) \le \sigma^2$ | $2\exp\!\left(-\frac{nt^2}{2(\sigma^2 + Mt/3)}\right)$ | Replaces worst-case range $(b-a)^2$ with true $\sigma^2$ |
:::

::: {.slide title="Concept Check: Concentration and High Dimensions"}
[Check Your Understanding]{.kicker}

1. **Test-set sizing for $99\%$ confidence:** Using Hoeffding's inequality for a $0/1$ classification loss, how many held-out test examples are needed to guarantee that the empirical accuracy is within $\pm 1\%$ ($t = 0.01$) of the true accuracy with $99\%$ confidence ($\delta = 0.01$)?
2. **Latent chord midpoint vs. shell:** Two independent latent vectors $\mathbf{z}_1, \mathbf{z}_2 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{512})$ are linearly averaged to form the midpoint $\mathbf{z}_{\mathrm{mid}} = \frac{1}{2}(\mathbf{z}_1 + \mathbf{z}_2)$. What are the typical Euclidean norms of $\mathbf{z}_1$, $\mathbf{z}_2$, and $\mathbf{z}_{\mathrm{mid}}$?
3. **Why Bernstein beats Hoeffding at low error rates:** For a classifier with a $1\%$ error rate on $n = 10{,}000$ images, why is Hoeffding's confidence interval more than $3\times$ wider than Bernstein's?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $n \ge \frac{\log(2/\delta)}{2t^2} = \frac{\log(200)}{2 \times 10^{-4}} \approx \frac{5.2983}{0.0002} = 26{,}492$ test examples.
2. Typical draws have $\|\mathbf{z}_1\| \approx \|\mathbf{z}_2\| \approx \sqrt{512} \approx 22.6 \pm 0.7$, whereas $\mathbf{z}_{\mathrm{mid}} \sim \mathcal{N}(\mathbf{0}, \frac{1}{2}\mathbf{I}_{512})$ has typical norm $\sqrt{256} = 16.0 \pm 0.5$—More than $9$ shell standard deviations below the training shell!
3. Hoeffding bounds every $[0,1]$ variable using the worst-case variance proxy $\sigma^2 = 1/4 = 0.25$ (attained only at $p = 0.5$), whereas Bernstein uses the true Bernoulli variance $\sigma^2 = 0.01 \times 0.99 \approx 0.01$, which is $25\times$ smaller.
:::
:::

::: {.slide title="Summary: Concentration and High-Dimensional Geometry"}
[Key Takeaways]{.kicker}

| Concept | Mathematical Form | Statistical & Deep Learning Role |
| :--- | :--- | :--- |
| **Chernoff Method** | $P(S \ge nt) \le \inf_{\lambda > 0} e^{-\lambda nt}\prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]$ | Turns independent sums into products of MGFs for exponential decay |
| **Hoeffding's Inequality** | $\lvert\bar{X} - \mu\rvert \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$ w.p. $\ge 1-\delta$ | Distribution-free finite-sample error bar; $\log(1/\delta)$ enables union bounds |
| **Bernstein's Inequality** | $P(\lvert\bar{X} - \mu\rvert \ge t) \le 2\exp\!\left(-\frac{nt^2}{2(\sigma^2 + Mt/3)}\right)$ | Tightens confidence intervals when error variance $\sigma^2 \ll (b-a)^2/4$ |
| **Gaussian Thin Shell** | $P\bigl(\bigl\lvert\frac{\|\mathbf{x}\|}{\sqrt{d}} - 1\bigr\rvert \ge \varepsilon\bigr) \le 2e^{-d\varepsilon^2/8}$; $\operatorname{sd}(\|\mathbf{x}\|) \approx \frac{1}{\sqrt{2}}$ | $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ lives on a shell of radius $\sqrt{d}$ and $O(1)$ thickness; requires Slerp |
| **Near-Orthogonality** | $P(\lvert\cos(\mathbf{x}, \mathbf{u})\rvert \ge \varepsilon) \le 2e^{-d\varepsilon^2/2}$ | Exponentially many ($e^{d\varepsilon^2/4}$) quasi-orthogonal directions fit in $\mathbb{R}^d$ |
| **Depth & Random Projections** | $\operatorname{Var}\bigl(\log\frac{\|\mathbf{h}_K\|^2}{\|\mathbf{h}_0\|^2}\bigr) \approx \frac{5K}{d}$; JL $k \ge \frac{8}{\varepsilon^2}\log\frac{n^2}{\delta}$ | Explains why He initialization needs $K \ll d$ and why $O(\log n)$ projections preserve distances |
:::

