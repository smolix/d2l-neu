# Concentration Inequalities
:label:`sec_mdl-concentration-generalization`

A test accuracy is a measurement, and :numref:`sec_mdl-statistics` put an
error bar on it. That error bar, $\pm1.96\,\hat\sigma/\sqrt n$, comes from a
Gaussian approximation to the sampling distribution, which is exact only as
$n\to\infty$. :numref:`sec_mdl-random_variables` gave a bound that holds at
every $n$, Chebyshev's inequality, but it is weak. How far can a test-set
error be from the truth, with a guarantee that holds at the sample size we
actually have and stays useful when we demand that it almost never fails?

Take a classifier that makes 830 mistakes on 10,000 test images, an error
rate of 8.3%. Chebyshev's inequality bounds the chance that the average of
$n$ independent draws misses its mean by $t$ or more by $\sigma^2/(nt^2)$,
and a 0/1 outcome has variance $\sigma^2\leq\tfrac14$. So the true error
lies within $\pm2.2\%$ of 8.3% with 95% confidence. Ask for a guarantee that
fails only once in a million, and Chebyshev answers $\pm500\%$, which says
nothing. Yet the exact probability that an average of 0/1 outcomes strays
far falls exponentially in $n$, and a bound that captures this answers
$\pm1.4\%$ and $\pm2.7\%$ for the same two questions.

The idea behind such bounds is to apply Markov's inequality not to the
average but to its exponential, which turns a sum of independent terms into
a product. Sergei Bernstein found bounds of this kind in the 1920s, Herman
Chernoff used the method in 1952 to measure how fast statistical tests
become reliable, and Wassily Hoeffding gave in 1963 the form used
throughout machine learning :cite:`Hoeffding.1963`.

Why ask for one failure in a million? A learner that compares many models and
keeps the best needs a guarantee for all of them at once, and only a tail that
shrinks exponentially gives one at a practical sample size; that is the
subject of :numref:`sec_mdl-generalization-bounds`. This section sizes a test
set from a required precision. The same calculation, applied to the
coordinates of a random vector, shows that a standard Gaussian vector in
$\mathbb{R}^{784}$ has a length within a few percent of 28 and that two random
directions are nearly orthogonal. Initialization scales and embeddings rest on
both facts.

Three results are worth remembering: the Chernoff method
:eqref:`eq_mdl-chernoff`, Hoeffding's interval
:eqref:`eq_mdl-hoeffding-interval`, and the concentration of the norm
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

The Gaussian interval is narrow because Gaussian tails fall like $e^{-z^2/2}$,
a rate it takes from the central limit theorem and that therefore holds only
in the limit. We want a rate of this kind together with a guarantee at every
finite $n$. First we measure how far Chebyshev's bound is from the truth.

### Chebyshev's Rate

Chebyshev's bound is Markov's inequality :eqref:`eq_mdl-markov` applied to a
squared deviation. Here the deviation is that of $\bar X$, the mean of $n$
independent draws with mean $\mu$ and finite variance $\sigma^2$. The
expectation of $(\bar X-\mu)^2$ is $\textrm{Var}(\bar X)=\sigma^2/n$, so

$$
P\bigl(|\bar X-\mu|\geq t\bigr)\leq\frac{\sigma^2}{nt^2}.
$$

The miss probability falls like $1/n$. To push it from $1$ to $10^{-6}$ at a
fixed deviation $t$, the sample must grow a million-fold.

How small is the true miss probability? The smallest case where we can
compute it exactly is a fair coin. Let $\hat p$ be the fraction of heads in
$n$ flips, and ask for the probability that it misses $\tfrac12$ by $t=0.1$ or more. A
flip has variance $\tfrac14$, so Chebyshev's bound is
$0.25/(0.01\,n)=25/n$. The exact probability is a binomial sum, which we
evaluate in log space so that $n=5{,}000$ does not underflow.

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

At $n=10$ the exact tail is $0.75$ and the bound, $2.5$, says nothing. At
$n=100$ the bound is $0.25$, about four times the exact $0.057$. Beyond that
the two part ways. At $n=1{,}000$ the exact tail is $2.7\times10^{-10}$
while Chebyshev allows $0.025$, eight orders of magnitude more, and at
$n=5{,}000$ the gap is forty-two orders. Each tenfold increase of $n$
divides Chebyshev's bound by ten, while the exact tail loses a fixed number
of decimal places per hundred flips: on a log scale it falls along a
straight line.

Chebyshev's bound cannot be tightened without more information about the
distribution. :numref:`sec_mdl-random_variables` showed that it holds with
equality for a three-point law, so no bound that uses only the mean and the
variance can improve on it for every distribution.
The coin has more structure than two moments: it is bounded, so all its
moments are small. A bound that uses all moments at once needs the moment
generating function.

### The Chernoff Method

To use more than two moments, look at what Markov's and Chebyshev's
inequalities have in common. If a function $g$ lies above the indicator
$\mathbf 1\{x\geq t\}$ everywhere, a *majorant* of the indicator, then
$P(X\geq t)=\mathbb{E}[\mathbf 1\{X\geq t\}]\leq\mathbb{E}[g(X)]$. Markov's
inequality uses $g(x)=x/t$ for a nonnegative $X$. Chebyshev's uses
$g(x)=(x/t)^2$, in practice with the deviation $X-\mu$ in place of $X$. The
exponential $e^{\lambda(x-t)}$, for any $\lambda>0$, lies above the indicator
on the whole real line. Panel (a) of :numref:`fig_mdl-conc-majorants` draws
the three functions for one nonnegative $X$. The bound is the expectation of
$g$, so what matters is the value of $g$ where the mass of $X$ sits.

![A tail bound is the expectation of a function lying above the indicator. (a) For a threshold $t=2$, the indicator $\mathbf{1}\{x\geq t\}$ (black) and three functions above it: Markov's $x/t$, Chebyshev's $(x/t)^2$, and Chernoff's $e^{\lambda(x-t)}$ with $\lambda=1.5$. The shaded bump marks where the mass of $X$ sits; each bound is the average of its function under that mass, and raising $\lambda$ makes the exponential small there. (b) The Chernoff exponent $-\lambda nt+n\lambda^2/8$ for the mean of $n=100$ fair coins at $t=0.1$. Every $\lambda>0$ gives a valid bound; the best, at the minimum $\lambda=4t$ of the parabola, is $e^{-2}$.](../img/mdl-prob-conc-majorants.svg)
:label:`fig_mdl-conc-majorants`

The expectation of the exponential has a name. The *moment generating
function* of $X$ is $\mathbb{E}[e^{\lambda X}]$, as a function of $\lambda$,
wherever the expectation is finite. For a fair coin centered at zero,
$Z\in\{-\tfrac12,\tfrac12\}$, it is $\tfrac12(e^{\lambda/2}+e^{-\lambda/2})
=\cosh(\lambda/2)$. It is finite for every $\lambda$ when $X$ is bounded,
and infinite for every $\lambda>0$ when the right tail falls only
polynomially, as for a Student-$t$ variable; for such variables the method
below gives nothing.

Markov's inequality applied to the nonnegative variable $e^{\lambda X}$
gives the bound. The map $x\mapsto e^{\lambda x}$ is increasing, so the
event is unchanged, and for every $\lambda>0$

$$
P(X \geq t)
 = P\bigl(e^{\lambda X} \geq e^{\lambda t}\bigr)
 \leq e^{-\lambda t}\,\mathbb{E}\bigl[e^{\lambda X}\bigr],
\qquad\textrm{hence}\qquad
P(X \geq t) \leq \inf_{\lambda > 0}\, e^{-\lambda t}\,
\mathbb{E}\bigl[e^{\lambda X}\bigr].
$$
:eqlabel:`eq_mdl-chernoff`

Every $\lambda>0$ gives a valid bound, and we keep the best one. This is the
*Chernoff method*.

The exponential is the right majorant for a second reason. If
$S=X_1+\cdots+X_n$ is a sum of independent terms, then
$e^{\lambda S}=\prod_ie^{\lambda X_i}$ is a product of independent factors,
and the expectation of such a product is the product of the expectations:

$$
\mathbb{E}\bigl[e^{\lambda S}\bigr]=\prod_{i=1}^n\mathbb{E}\bigl[e^{\lambda X_i}\bigr].
$$

The moment generating function of a sum is the product of those of its
terms, so the logarithm of the Chernoff bound for a sum has $n$ terms and
grows in proportion to $n$. No polynomial majorant factorizes this way:
$\mathbb{E}[S^4]$ is not the product of the fourth moments. What remains is
to bound the moment generating function of one summand.

### The Fair Coin

The fair coin is the smallest case, and working it to the end shows every
step of the general proof. A centered flip $Z=X-\tfrac12$ takes the values
$\pm\tfrac12$ with probability $\tfrac12$ each, so
$\mathbb{E}[e^{\lambda Z}]=\cosh(\lambda/2)$. Compare the Taylor series of
$\cosh u$ and of $e^{u^2/2}$ term by term:

$$
\cosh u=\sum_{k\geq0}\frac{u^{2k}}{(2k)!}
\leq\sum_{k\geq0}\frac{(u^2/2)^k}{k!}=e^{u^2/2},
$$

because $(2k)!\geq2\cdot4\cdots(2k)=2^kk!$: the even factors of $(2k)!$
alone give $2^kk!$, and the odd ones are at least $1$. With $u=\lambda/2$,
one centered flip has $\mathbb{E}[e^{\lambda Z}]\leq e^{\lambda^2/8}$ for
every $\lambda$. A Gaussian $\mathcal N(0,\sigma^2)$ has
$\mathbb{E}[e^{\lambda X}]=e^{\lambda^2\sigma^2/2}$ (complete the square in
the integral), so a flip's moment generating function is no larger than that
of a Gaussian with standard deviation $\tfrac12$.

For $n$ independent flips, the event $\hat p-\tfrac12\geq t$ is the event
$\sum_iZ_i\geq nt$. The Chernoff method and the product rule give

$$
P\bigl(\hat p-\tfrac12\geq t\bigr)
\leq e^{-\lambda nt}\prod_{i=1}^n\mathbb{E}\bigl[e^{\lambda Z_i}\bigr]
\leq\exp\Bigl(-\lambda nt+\frac{n\lambda^2}{8}\Bigr).
$$

The exponent is a parabola in $\lambda$. Panel (b) of
:numref:`fig_mdl-conc-majorants` draws it for $n=100$ and $t=0.1$, where it
reads $-10\lambda+12.5\lambda^2$. Any $\lambda>0$ gives a bound, for instance
$e^{-0.875}\approx0.42$ at $\lambda=0.1$. The best one sits at the minimum,
$\lambda=4t$, where the exponent is $-4nt^2+2nt^2=-2nt^2$. The lower tail is
the same by symmetry, and adding the two tails gives

$$
P\bigl(|\hat p-\tfrac12|\geq t\bigr)\leq2e^{-2nt^2}.
$$

The probability of missing by $t$ falls exponentially in $nt^2$. At $n=100$
and $t=0.1$ the bound is $2e^{-2}\approx0.27$. At $n=1{,}000$ it is
$2e^{-20}\approx4.1\times10^{-9}$, where Chebyshev allowed $0.025$.

Why is the exponential majorant so effective? The function $e^{\lambda(S-nt)}$
equals $1$ at the threshold, but at the mean of the sum, where the mass is, it
equals $e^{-\lambda nt}$, exponentially small in $n$. What limits $\lambda$ is
that the function grows with $S$, so outcomes above the mean count for more
than $e^{-\lambda nt}$. The product of the moment generating functions, at
most $e^{n\lambda^2/8}$, measures by how much, and the best $\lambda$ balances
the two factors.

The proof used one fact about the coin: the moment generating function of a
centered flip is at most $e^{\lambda^2/8}$, that of a Gaussian with standard
deviation $\tfrac12$. The $\tfrac12$ is half the range of a flip. Hoeffding's
lemma says that every variable confined to an interval of length $b-a$
satisfies the same bound with $(b-a)/2$ in place of $\tfrac12$.

### Hoeffding's Lemma and Hoeffding's Inequality

A test-set loss is rarely a fair coin. An error rate of 8.3% is a biased coin,
and a clipped loss takes many values. We want the coin's bound for every
bounded variable. The lemma that gives it is the hard step of this section,
and its proof rests on one fact about variances.

**Proposition (variance of a bounded variable).** *If $Y$ takes values in
$[a,b]$, then $\textrm{Var}(Y)\leq(b-a)^2/4$, with equality when $Y$ takes
the values $a$ and $b$ with probability $\tfrac12$ each.*

**Proof.** The mean minimizes the expected squared deviation, since
$\mathbb{E}[(Y-c)^2]=\textrm{Var}(Y)+(\mathbb{E}[Y]-c)^2$ for every center
$c$. Take $c$ to be the midpoint. Every point of $[a,b]$ lies within
$(b-a)/2$ of it, so

$$
\textrm{Var}(Y)\leq\mathbb{E}\Bigl[\Bigl(Y-\frac{a+b}{2}\Bigr)^2\Bigr]
\leq\Bigl(\frac{b-a}{2}\Bigr)^2.
$$

Both inequalities are equalities for the two-point law with equal mass at
$a$ and $b$. $\blacksquare$

The fair coin is this extreme case, with variance $\tfrac14$ on $[0,1]$.

**Proposition (Hoeffding's lemma).** *Let $X$ be a random variable with
$\mathbb{E}[X]=0$ taking values in $[a,b]$. Then for every
$\lambda\in\mathbb{R}$,*

$$
\mathbb{E}\bigl[e^{\lambda X}\bigr] \leq \exp\!\left(\frac{\lambda^2 (b-a)^2}{8}\right).
$$
:eqlabel:`eq_mdl-hoeffding-lemma`

**Proof.** We show that $\psi(\lambda)=\log\mathbb{E}[e^{\lambda X}]$ grows
no faster than the parabola $\lambda^2(b-a)^2/8$. Since $X$ is bounded,
$\psi$ is finite and smooth, and we may differentiate under the
expectation. At zero, $\psi(0)=0$, and

$$
\psi'(\lambda) = \frac{\mathbb{E}[X e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]},
\qquad\textrm{so}\qquad
\psi'(0) = \mathbb{E}[X] = 0 .
$$

The second derivative is a variance, as it is for the log-partition function
of an exponential family (:numref:`sec_mdl-exponential-family`); $\psi$ is
one. Let $q_\lambda$ be the *tilted* law, which reweights the law of $X$ by
$e^{\lambda x}$ and renormalizes. Then $\psi'(\lambda)$ is the mean of $X$
under $q_\lambda$, and

$$
\psi''(\lambda)
 = \frac{\mathbb{E}[X^2 e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}
 - \left(\frac{\mathbb{E}[X e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}\right)^{\!2}
 = \textrm{Var}_{q_\lambda}(X).
$$

The tilt moves no mass outside $[a,b]$, so the variance bound applies at
every $\lambda$: $\psi''(\lambda)\leq(b-a)^2/4$. Taylor's theorem with the
Lagrange remainder finishes the proof. For some $\xi$ between $0$ and
$\lambda$,

$$
\psi(\lambda)
 = \psi(0) + \lambda\,\psi'(0) + \frac{\lambda^2}{2}\,\psi''(\xi)
 \leq \frac{\lambda^2 (b-a)^2}{8},
$$

and exponentiating gives :eqref:`eq_mdl-hoeffding-lemma`. $\blacksquare$

A bounded, centered variable has a moment generating function no larger
than that of a Gaussian with standard deviation $(b-a)/2$. For the centered
coin, $b-a=1$, and the lemma returns the bound $\cosh(\lambda/2)\leq
e^{\lambda^2/8}$ that we proved by series. The proof of the coin's tail now
goes through for every bounded variable.

**Proposition (Hoeffding's inequality).** *Let $X_1,\ldots,X_n$ be
independent random variables with $X_i\in[a,b]$, and let
$\bar X = \frac1n\sum_i X_i$. Then for every $t>0$,*

$$
P\bigl(|\bar X - \mathbb{E}[\bar X]| \geq t\bigr)
 \leq 2\exp\!\left(-\frac{2 n t^2}{(b-a)^2}\right).
$$
:eqlabel:`eq_mdl-hoeffding`

**Proof.** Write $Z_i = X_i - \mathbb{E}[X_i]$. Each $Z_i$ is centered and
lies in an interval of length $b-a$, so the lemma applies. For any
$\lambda>0$, the Chernoff method :eqref:`eq_mdl-chernoff` and the product
rule give

$$
P(\bar X - \mathbb{E}[\bar X] \geq t)
 = P\Bigl(\sum_i Z_i \geq nt\Bigr)
 \leq e^{-\lambda n t} \prod_{i=1}^n \mathbb{E}\bigl[e^{\lambda Z_i}\bigr]
 \leq \exp\!\left(-\lambda n t + \frac{n\lambda^2 (b-a)^2}{8}\right).
$$

This is the coin's parabola with $(b-a)^2$ in place of $1$. It is minimized at
$\lambda = 4t/(b-a)^2$, where its value is $-2nt^2/(b-a)^2$. The same argument
applied to $-Z_i$ bounds the lower tail, and adding the two tails gives the
factor $2$. $\blacksquare$

The mean of $n$ independent bounded terms misses its expectation by $t$ with a
probability that falls exponentially in $nt^2$, with $t$ measured in units of
the range. The bound needs no Gaussianity, no variance estimate and no
asymptotics. It needs only boundedness and independence, and it holds at every
finite $n$. Independence is a real restriction: the bound does not cover the
correlated draws of a Markov chain.

The proof never used that the ranges agree. For independent centered
$Y_i\in[a_i,b_i]$, the product of the lemma's bounds has $\sum_i(b_i-a_i)^2$
where it had $n(b-a)^2$, and the same optimization gives for the sum

$$
P\Bigl(\Bigl|\sum_iY_i\Bigr|\geq s\Bigr)
\leq2\exp\Bigl(-\frac{2s^2}{\sum_i(b_i-a_i)^2}\Bigr).
$$
:eqlabel:`eq_mdl-hoeffding-unequal`

We need this form for random directions below.

For losses in $[0,1]$, $b-a=1$, and the bound is at most $\delta$ once
$n\geq\log(2/\delta)/(2t^2)$. To estimate an error rate within $t=0.01$ with
failure probability $\delta=0.05$, this gives
$\log 40/(2\cdot10^{-4})=18{,}444.4$, so $18{,}445$ examples. This is the
number that :numref:`chap_classification_generalization` quoted, about twice
the 9,604 examples that the Gaussian approximation asks for.

Setting the right-hand side of :eqref:`eq_mdl-hoeffding` equal to $\delta$
and solving for $t$ gives the form used in practice: with probability at
least $1-\delta$,

$$
\bigl|\bar X - \mathbb{E}[\bar X]\bigr|
 \leq (b-a)\sqrt{\frac{\log(2/\delta)}{2n}} .
$$
:eqlabel:`eq_mdl-hoeffding-interval`

This is a confidence interval whose half-width shrinks like $1/\sqrt n$, the
same rate as the Gaussian interval $\hat\mu\pm1.96\,\hat\sigma/\sqrt n$, but
it is valid at every $n$.

For the classifier with 830 errors on 10,000 images, at $\delta=0.05$,
Hoeffding gives $\pm1.36\%$ and the Gaussian interval
$\pm1.96\sqrt{0.083\cdot0.917/10^4}=\pm0.54\%$. The two differ in their
constants. Hoeffding's half-width is the Gaussian one with $\hat\sigma$
replaced by its worst case $(b-a)/2=0.5$ and the quantile $1.96$ replaced by
$\sqrt{2\log(2/\delta)}=2.72$. With $\hat\sigma=0.28$ the ratio is $2.5$.
When the spread of the data is much smaller than their range, Hoeffding is
conservative, and Bernstein's inequality below repairs this.

The constant $\tfrac18$ in Hoeffding's lemma cannot be improved for all
bounded variables: a variable with mass $\tfrac12$ at each endpoint of its
interval attains it to second order in $\lambda$ (Exercise
:exref:`ex_mdl-concentration-generalization-random-signs`).

### The Fair Coin in Numbers
:label:`subsec_mdl-tail-bounds-numerical`

We add Hoeffding's bound $2e^{-2nt^2}$ as a third column to the table for the
fair coin at $t=0.1$, and plot all three quantities against $n$ on a log
scale.

```{.python .input #mdl-concentration-generalization-the-tail-race-in-code}
print(f'{"n":>6} {"exact":>10} {"Chebyshev":>10} {"Hoeffding":>10}')
for n in (10, 100, 1000, 5000):
    print(f'{n:6d} {exact_tail(n, t):10.2e} {0.25 / (n * t**2):10.2e} '
          f'{2 * np.exp(-2 * n * t**2):10.2e}')

ns = np.arange(10, 701, 10)
d2l.plot(ns, [np.array([exact_tail(n, t) for n in ns]),
              0.25 / (ns * t**2), 2 * np.exp(-2 * ns * t**2)],
         'n', 'P(|p_hat - 1/2| >= 0.1)',
         legend=['exact', 'Chebyshev', 'Hoeffding'], yscale='log')
```

At $n=10$ both bounds exceed $1$. At $n=100$ Chebyshev's $0.25$ is a shade
tighter than Hoeffding's $0.27$: at small $n$ the constants decide, and
Chebyshev uses the true variance $\tfrac14$. For large $n$ the rate decides,
and Hoeffding stays within a factor of 15 of the exact tail at $n=1{,}000$ and
of 57 at $n=5{,}000$.

On the log plot the exact tail and Hoeffding's bound are nearly parallel
lines. The logarithm of Hoeffding's bound falls by
$2t^2=0.02$ per flip, and that of the exact tail by about $0.0201$. The
second number is the Kullback–Leibler divergence
$D_{\textrm{KL}}(0.6\|0.5)=0.6\log1.2+0.4\log0.8$ between a coin of bias
$0.6$ and the fair coin (:numref:`sec_mdl-divergences-distances`), and the
Chernoff method with the exact moment generating function of a coin attains
this rate (Exercise :exref:`ex_mdl-concentration-generalization-chernoff-coins`).

At small $n$ no bound helps. The coin of the chapter introduction landed heads
9 times in 13 flips, a deviation of $t=2.5/13$ from $\tfrac12$. Hoeffding's
bound on a deviation that large is
$2e^{-2\cdot13\cdot(2.5/13)^2}=2e^{-0.96}\approx0.77$, which rules out
nothing. The exact two-sided probability is $0.27$, so with 13 flips not even
the exact tail rules out a fair coin.

### Sub-Gaussian and Sub-Exponential Variables

Hoeffding's interval for the 8.3% classifier was 2.5 times wider than the
Gaussian one because it uses the range of a 0/1 outcome and ignores its
spread. Bernstein's inequality uses both.

**Proposition (Bernstein's inequality).** *Let $X_1,\ldots,X_n$ be
independent with $\textrm{Var}(X_i)\leq\sigma^2$ and
$|X_i-\mathbb{E}[X_i]|\leq M$. Then for every $t>0$,*

$$
P\bigl(|\bar X - \mathbb{E}[\bar X]| \geq t\bigr)
 \leq 2\exp\!\left(-\frac{n t^2}{2\bigl(\sigma^2 + M t/3\bigr)}\right).
$$
:eqlabel:`eq_mdl-bernstein`

The proof is the Chernoff method once more
:cite:`Boucheron.Lugosi.Massart.2013`, with a bound on the moment generating
function that uses the variance as well as the range; Exercise
:exref:`ex_mdl-concentration-generalization-bernstein` carries it out. For
small $t$ the denominator is close to $2\sigma^2$, and the bound is the
Gaussian tail with the *true* variance. For large $t$ the term $Mt/3$ takes
over and the bound decays only exponentially in $t$.

How much does the variance help? A classifier with an error rate of 2% on
$n=10{,}000$ examples has per-example variance $\sigma^2=0.02\cdot0.98$, and
its centered losses satisfy $M=1$. Setting the bound equal to $\delta$ gives
$nt^2/(2(\sigma^2+t/3))=\log(2/\delta)$, a quadratic in $t$. We solve it at
$\delta=0.05$ for both error rates we have met, next to Hoeffding and the
Gaussian interval.

```{.python .input #mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables}
def half_widths(p, n, delta=0.05, z=1.96):
    """Hoeffding, Bernstein and Gaussian half-widths for an error rate p."""
    L, var = np.log(2 / delta), p * (1 - p)
    hoeffding = np.sqrt(L / (2 * n))
    c = 2 * L / 3  # Bernstein with M = 1: n t^2 - c t - 2 L var = 0
    bernstein = (c + np.sqrt(c**2 + 8 * n * L * var)) / (2 * n)
    return hoeffding, bernstein, z * np.sqrt(var / n)
```

```{.python .input #mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables-1}
for p in (0.083, 0.02):
    hoeffding, bernstein, gauss = half_widths(p, 10000)
    print(f'error {p:.1%}: Hoeffding ±{hoeffding:.2%}, '
          f'Bernstein ±{bernstein:.2%}, Gaussian ±{gauss:.2%}')
```

At an error rate of 2%, Bernstein's half-width is $\pm0.39\%$ against
Hoeffding's $\pm1.36\%$, three and a half times tighter, and within a factor
of $1.4$ of the asymptotic Gaussian $\pm0.27\%$. At 8.3% the gain is smaller,
$\pm0.76\%$ against $\pm1.36\%$.

These two half-widths are not yet guarantees. The cell put the measured rate
into the variance $p(1-p)$, whereas Bernstein's inequality asks for a bound on
the true variance: the substitution of $\hat\sigma$ for $\sigma$ that we held
against the Gaussian interval. The repair is cheap. The true rate is at most
$\hat p+t$, so we solve the same equation with variance $(\hat p+t)(1-\hat
p-t)$, which gives $0.43\%$ and $0.79\%$ in place of $0.39\%$ and $0.76\%$.
This answers the opening question for the classifier with 830 errors: with 95%
confidence its true error lies within about $0.8\%$ of the measured 8.3%.
Error rates of a few percent are where practitioners work, and there the
variance-aware bound matters most.

Hoeffding's proof used one property of the summands: a moment generating
function bounded by a Gaussian's. Giving that property a name lets us apply
the proof to unbounded variables, and shows where it stops working. A random
variable $X$ is **sub-Gaussian** with *variance proxy* $\sigma^2$ if its
centered moment generating function is no larger than that of $\mathcal
N(0,\sigma^2)$:

$$
\mathbb{E}\bigl[e^{\lambda (X - \mathbb{E}[X])}\bigr] \leq \exp\!\left(\frac{\lambda^2\sigma^2}{2}\right)
\qquad\textrm{for all } \lambda\in\mathbb{R}.
$$
:eqlabel:`eq_mdl-subgaussian`

The tails of such a variable are no heavier than a Gaussian's: the Chernoff
method with $\lambda=t/\sigma^2$ gives $P(|X-\mathbb{E}[X]|\geq
t)\leq2e^{-t^2/(2\sigma^2)}$. Here $\sigma^2$ is the proxy and not the
variance. The proxy is never smaller than the variance (compare the two sides
of :eqref:`eq_mdl-subgaussian` to second order in $\lambda$), and it can be
much larger: a 0/1 outcome with error rate 2% has variance $0.0196$, while
Hoeffding's lemma supplies the proxy $\tfrac14$. That is why Hoeffding's
interval was wide.

A Gaussian $\mathcal N(\mu,\sigma^2)$ has proxy $\sigma^2$, with equality. A
variable on $[a,b]$ has proxy $(b-a)^2/4$ by Hoeffding's lemma. A random sign
$\xi\in\{-1,+1\}$ has proxy $1$, since
$\mathbb{E}[e^{\lambda\xi}]=\cosh\lambda\leq e^{\lambda^2/2}$ by the series
comparison above. Sub-Gaussianity also survives averaging: if $X_1,\ldots,X_n$
are independent with proxy $\sigma^2$, the product rule gives $\bar X$ the
proxy $\sigma^2/n$, and hence $P(|\bar X-\mathbb{E}\bar X|\geq
t)\leq2e^{-nt^2/(2\sigma^2)}$.

Not every variable with all moments finite is sub-Gaussian, and the one we
need next is not. The squared length of a Gaussian vector is a sum of squares
of standard Gaussians, and the square of a standard Gaussian $x$ has

$$
\mathbb{E}\bigl[e^{\lambda(x^2-1)}\bigr]=\frac{e^{-\lambda}}{\sqrt{1-2\lambda}}
\qquad\textrm{for }\lambda<\tfrac12,
$$

and the expectation is infinite for $\lambda\geq\tfrac12$. So
:eqref:`eq_mdl-subgaussian` fails for $x^2$ at every variance proxy. Its right
tail falls like $e^{-t/2}$, too slowly for a Gaussian bound.

A Gaussian-type bound does hold on a window of $\lambda$ around zero. A random
variable $X$ is **sub-exponential** with parameters $(\nu,\alpha)$ if

$$
\mathbb{E}\bigl[e^{\lambda (X - \mathbb{E}[X])}\bigr] \leq \exp\!\left(\frac{\lambda^2\nu^2}{2}\right)
\qquad\textrm{for } |\lambda| \leq \frac{1}{\alpha}.
$$
:eqlabel:`eq_mdl-subexponential`

Here $\nu^2$ plays the part of the variance proxy near $\lambda=0$, and
$1/\alpha$ is the half-width of the window, which limits how large a $\lambda$
the Chernoff method may use. :numref:`subsec_mdl-thin-shell` shows that $x^2$
is sub-exponential with $\nu=2$ and $\alpha=4$. For an average of such
variables, the Chernoff optimization splits at the edge of the window. The
proof can be skipped on a first reading; the next part uses only the
statement.

**Proposition (sub-exponential tail bound).** *Let $X_1,\ldots,X_n$ be
independent and $(\nu,\alpha)$-sub-exponential. Then for every $t>0$,*

$$
P\bigl(|\bar X-\mathbb{E}[\bar X]|\geq t\bigr)
\leq2\exp\!\left(-\frac n2\min\Bigl(\frac{t^2}{\nu^2},\,\frac{t}{\alpha}\Bigr)\right).
$$
:eqlabel:`eq_mdl-subexponential-tail`

**Proof.** For $0<\lambda\leq1/\alpha$, the Chernoff method and the product
rule bound $P(\bar X-\mathbb{E}[\bar X]\geq t)$ by
$\exp(-\lambda nt+n\lambda^2\nu^2/2)$. If $t\leq\nu^2/\alpha$, the
parabola's minimizer $\lambda=t/\nu^2$ lies in the window and gives
$e^{-nt^2/(2\nu^2)}$. If $t>\nu^2/\alpha$, the minimizer lies outside the
window, and we take the largest admissible value, $\lambda=1/\alpha$. The
exponent becomes $-nt/\alpha+n\nu^2/(2\alpha^2)$, and since
$\nu^2/\alpha<t$ it is at most $-nt/\alpha+nt/(2\alpha)=-nt/(2\alpha)$. In
both cases the exponent is at most $-\tfrac n2\min(t^2/\nu^2,t/\alpha)$. The lower
tail follows in the same way from $\lambda\in[-1/\alpha,0)$. $\blacksquare$

Averages of bounded or sub-Gaussian terms have tails $e^{-ct^2}$ at every
scale. A sub-exponential average has such a tail only for deviations below
$\nu^2/\alpha$, and an exponential tail, $e^{-nt/(2\alpha)}$, beyond it.

## Probability in High Dimension

The bounds so far protect a single average. A random vector in $\mathbb{R}^d$
has $d$ independent coordinates, and its length and its angles with other
vectors are averages over those coordinates. Concentration therefore governs
the geometry of high-dimensional random vectors. A reader who starts here
needs two results from above: the sub-exponential tail bound
:eqref:`eq_mdl-subexponential-tail` for squared lengths and Hoeffding's
inequality in the form :eqref:`eq_mdl-hoeffding-unequal` for inner products.

### The Thin Shell
:label:`subsec_mdl-thin-shell`

Draw $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_{784})$, a Gaussian vector
with as many coordinates as a $28\times28$ image. Its density is largest at
the origin. Yet the probability that $\mathbf x$ lands within distance 14 of
the origin, half its typical length, is about $10^{-110}$, the probability
that a sum of 784 squared standard Gaussians is at most 196. No sample we
could ever draw will contain such a point. A typical draw has norm $28.0\pm0.7$.

The reason is that $\|\mathbf x\|^2=\sum_{i=1}^dx_i^2$ is a sum of $d$
independent terms with mean $1$, so $\|\mathbf x\|^2/d$ is a sample mean. Its
terms are unbounded, so Hoeffding's inequality does not apply, but they are
sub-exponential.

**Proposition (norm concentration).** *Let
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$. For every
$0<\varepsilon\leq 1$,*

$$
P\!\left(\Bigl|\frac{\|\mathbf x\|}{\sqrt d} - 1\Bigr| \geq \varepsilon\right)
 \leq 2\exp\!\left(-\frac{d\,\varepsilon^2}{8}\right).
$$
:eqlabel:`eq_mdl-norm-concentration`

**Proof.** The proof has three steps. First, the moment generating function
of a centered square, $e^{-\lambda}/\sqrt{1-2\lambda}$, satisfies

$$
\mathbb{E}\bigl[e^{\lambda (x_i^2 - 1)}\bigr] \leq e^{2\lambda^2}
\qquad\textrm{for } |\lambda| \leq \tfrac14 ,
$$
:eqlabel:`eq_mdl-chi2-mgf`

by a series comparison (Exercise
:exref:`ex_mdl-concentration-generalization-chi-square-tail`). In the
language of :eqref:`eq_mdl-subexponential`, $x_i^2$ is sub-exponential with
$\nu=2$ and $\alpha=4$. Second, the sub-exponential tail bound
:eqref:`eq_mdl-subexponential-tail` with $n=d$ and $t=\varepsilon$ gives

$$
P\Bigl(\Bigl|\frac{\|\mathbf x\|^2}{d}-1\Bigr|\geq\varepsilon\Bigr)
\leq2\exp\Bigl(-\frac d2\min\Bigl(\frac{\varepsilon^2}{4},\frac{\varepsilon}{4}\Bigr)\Bigr)
=2e^{-d\varepsilon^2/8}
\qquad\textrm{for }\varepsilon\leq1.
$$

Third, we pass from squares to norms. Write $z=\|\mathbf x\|/\sqrt d\geq0$.
If $|z-1|\geq\varepsilon$, then $|z^2-1|=|z-1|\,(z+1)\geq\varepsilon$,
because $z+1\geq1$. So the event for the norm is contained in the event for
the squared norm and inherits its bound. $\blacksquare$

The norm of a standard Gaussian vector misses $\sqrt d$ by a fixed fraction
with a probability that falls exponentially in $d$. The restriction
$\varepsilon\leq1$ matters only for the squared norm, whose tail beyond
$\varepsilon=1$ is exponential, $e^{-d\varepsilon/8}$ (Exercise
:exref:`ex_mdl-concentration-generalization-chi-square-tail`). The norm keeps
its Gaussian tail at every scale: a norm above $(1+\varepsilon)\sqrt d$ means
a squared norm above $(1+2\varepsilon+\varepsilon^2)d$, which has probability
at most $e^{-d(2\varepsilon+\varepsilon^2)/8}\leq e^{-d\varepsilon^2/8}$.

How thick is the shell? Since $\mathbb{E}[x^4]=3$, each square has
$\textrm{Var}(x^2)=3-1=2$, so $\textrm{Var}(\|\mathbf x\|^2)=2d$. The tangent
line of the square root at $d$ gives $\|\mathbf x\|\approx\sqrt d+(\|\mathbf
x\|^2-d)/(2\sqrt d)$, so the standard deviation of $\|\mathbf x\|$ is about
$\sqrt{2d}/(2\sqrt d)=1/\sqrt2\approx0.71$, whatever the dimension. A standard
Gaussian in $\mathbb{R}^d$ is a shell of radius $\sqrt d$ and thickness of
order one.

Why is the mass so far from the mode? The density
$(2\pi)^{-d/2}e^{-\|\mathbf x\|^2/2}$ depends on $\mathbf x$ only through
$r=\|\mathbf x\|$, and the sphere of radius $r$ has a surface area
proportional to $r^{d-1}$. The density of $\|\mathbf x\|$ is therefore
proportional to $r^{d-1}e^{-r^2/2}$. The first factor grows with $r$, the
second falls, and their product peaks where the derivative of
$(d-1)\log r-r^2/2$ vanishes, at $r=\sqrt{d-1}$. Panel (a) of
:numref:`fig_mdl-conc-shell` shows the two factors and their product at
$d=20$.

At a typical point, of norm $\sqrt d$, the density is $e^{-d/2}$
times its value at the origin, $10^{-170}$ at $d=784$, but there are vastly
more such points. Panel (b) shows the exact density of $\|\mathbf x\|$ for
$d$ from 2 to 512. The centers move out like $\sqrt d$ while the widths stay
near $1/\sqrt2$.

![The thin shell. (a) At $d=20$ the Gaussian density $e^{-r^2/2}$ falls with the radius $r$, the volume $r^{d-1}$ of the shell at radius $r$ grows, and their product, the density of $\|\mathbf{x}\|$, peaks at $\sqrt{d-1}$, far from the origin where the density is largest; each curve is scaled to maximum 1. (b) The exact density of $\|\mathbf{x}\|$ for $\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ and $d=2,8,32,128,512$. The centers move out like $\sqrt{d}$; the widths stay near $1/\sqrt{2}$.](../img/mdl-prob-conc-shell.svg)
:label:`fig_mdl-conc-shell`

The bound captures the form of the number we started from, but not its
size. A norm below 14 at $d=784$ means $\varepsilon=\tfrac12$, and
:eqref:`eq_mdl-norm-concentration` gives
$2e^{-784/32}=2e^{-24.5}\approx5\times10^{-11}$, against the exact
$10^{-110}$. The constant 8 in the exponent is loose; the exponential
dependence on $d$ is right.

### Near-Orthogonality of Random Directions

:numref:`sec_mdl-geometry-linear-algebraic-ops` showed that the cosine between
a fixed unit vector and a random direction in $\mathbb{R}^d$ has variance
$1/d$, which with Chebyshev's inequality gives a tail of order
$1/(d\varepsilon^2)$. Concentration replaces it by an exponential tail, and
with that tail we can count how many random directions are nearly orthogonal
at once.

Vectors of random signs are the cleanest case. Fix a unit vector $\mathbf u$
and let the $x_i\in\{-1,+1\}$ be independent uniform signs. Each term $u_ix_i$
of $\langle\mathbf x,\mathbf u\rangle$ is centered and lies in
$[-|u_i|,|u_i|]$, an interval of length $2|u_i|$. In
:eqref:`eq_mdl-hoeffding-unequal` the denominator is then $4\sum_iu_i^2=4$,
and

$$
P\bigl(|\langle\mathbf x,\mathbf u\rangle| \geq s\bigr)
 \leq 2 e^{-s^2/2}.
$$
:eqlabel:`eq_mdl-inner-product-tail`

The inner product of a random sign vector with any fixed unit vector is of
order one, with a Gaussian tail: each coordinate contributes $\pm|u_i|$, and
the signs cancel. The norm, meanwhile, is exactly $\sqrt d$, so with
$s=\varepsilon\sqrt d$ we get $P(|\cos(\mathbf x,\mathbf
u)|\geq\varepsilon)\leq2e^{-d\varepsilon^2/2}$. For a Gaussian $\mathbf x$ the
same rate holds approximately: $\langle\mathbf x,\mathbf u\rangle$ is a
standard Gaussian whatever $\mathbf u$ is, with the same tail $2e^{-s^2/2}$
(Exercise :exref:`ex_mdl-concentration-generalization-unequal-ranges`), and
the norm is within a few percent of $\sqrt d$ by the thin shell.

The exponential tail lets us count, and the count needs one more tool. The
*union bound* says that the probability that at least one of several events
occurs is at most the sum of their probabilities, whether or not the events
are independent. Take $m$ independent sign vectors. They form $m(m-1)/2<m^2/2$
pairs, and for each pair the cosine exceeds $\varepsilon$ in absolute value
with probability at most $2e^{-d\varepsilon^2/2}$. By the union bound, the
probability that any pair does is below $m^2e^{-d\varepsilon^2/2}$. So with
probability at least $1-\delta$ all pairwise cosines are below $\varepsilon$
whenever $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$, a number that grows
exponentially in $d$.

At $d=784$ and $\delta=0.05$ this guarantees about 570 directions with all
pairwise $|\cos|<0.2$, and about $10^7$ with $|\cos|<0.3$. These are
guarantees from a loose bound, not estimates of how many such directions
exist. A typical $|\cos|$ is much smaller, $\sqrt{2/(\pi d)}\approx0.028$, the
mean absolute value of a Gaussian with standard deviation $1/\sqrt d$.

Trained embeddings are not independent random directions. A cosine of $0.3$ at
$d=784$ lies more than eight standard deviations ($1/\sqrt{784}=0.036$) above
what independent directions produce, so they serve as the null model against
which structure in embeddings is measured.

### The Thin Shell in Numbers
:label:`subsec_mdl-measuring-shell`

The bounds above have loose constants, so we measure. The tangent-line
calculation predicts a standard deviation of the norm near $1/\sqrt2$ at every
$d$, and a Gaussian with standard deviation $1/\sqrt d$ predicts a mean
$|\cos|$ of $\sqrt{2/(\pi d)}$. We draw 10,000 pairs of vectors at
$d=2,20,200,2000$; one row will not fit.

```{.python .input #mdl-distributions-multivariate-gaussian-1}
for d in (2, 20, 200, 2000):
    x, y = rng.standard_normal((2, 10000, d))
    nx = np.linalg.norm(x, axis=1)
    cos = (x * y).sum(1) / (nx * np.linalg.norm(y, axis=1))
    print(f'd = {d:4d}: mean ||x||/sqrt(d) = {(nx / np.sqrt(d)).mean():.3f}, '
          f'sd ||x|| = {nx.std():.2f}, mean |cos| = {np.abs(cos).mean():.4f}'
          f' vs {np.sqrt(2 / (np.pi * d)):.4f}')
```

The standard deviation of the norm is about $0.65$ at $d=2$ and about $0.7$
after that, while the norm itself grows from about 1.3 to about 45. The row
that does not fit is $d=2$: its mean $|\cos|$ is about $0.64$. That is
$2/\pi$, the mean of $|\cos\theta|$ for a uniform angle; the formula
$\sqrt{2/(\pi d)}$ describes high dimensions only. From $d=20$ on, the
measured values, about $0.18$, $0.057$ and $0.018$, agree with it to within a
few percent.

Two further questions. How much of the mass lies within 10% of the typical
norm, in the shell $(1\pm0.1)\sqrt d$? And do distances still separate near
points from far ones? We measure the second by the *contrast* of a query
point, $(d_{\max}-d_{\min})/d_{\min}$ over its distances to 200 random points,
and report the median over 100 queries.

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

At $d=2$ the shell holds about 15% of the mass, at $d=20$ about 47%, and at
$d=200$ about 95%. The last number is what the tangent-line calculation
predicts: the ratio $\|\mathbf x\|/\sqrt d$ has standard deviation
$1/\sqrt{2d}=0.05$ at $d=200$, so the shell spans two standard deviations on
either side. At $d=2000$ every one of the 10,000 draws lies in the shell. The
median contrast falls from about 30 at $d=2$ (between roughly 25 and 40 from
run to run) to about 1.2, 0.27 and 0.08. At $d=2000$ the farthest of 200
random points is, for a typical query, less than 10% farther from it than the
nearest.

### Consequences for Deep Learning
:label:`subsec_mdl-concentration-dl`

Three practices in deep learning rest on these facts.

**Initialization.** LeCun initialization :cite:`LeCun.Bottou.Orr.ea.1998`
draws the entries of a $d\times d$ weight matrix independently with variance
$1/d$, and He initialization :cite:`He.Zhang.Ren.ea.2015` uses $2/d$ for ReLU
networks (:numref:`sec_numerical_stability`). The argument for these scales
computes an expectation over all draws of the weights: at LeCun's scale each
pre-activation $\langle\mathbf w,\mathbf x\rangle$ has variance $\|\mathbf
x\|^2/d$, which norm concentration puts close to $1$ for inputs with
coordinates of unit variance. We train one draw. Concentration says when the
expectation describes it.

For one He-initialized ReLU layer of width $d$, the ratio
$\rho=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$ of squared activation norms
has mean $1$ and variance $5/d$ (Exercise
:exref:`ex_mdl-concentration-generalization-depth-norms`), so one layer
changes the squared norm by about $\sqrt{5/d}$, 10% at $d=512$. Through $K$
layers the fluctuations add up in the logarithm, and
$\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ has variance about $5K/d$. The
variance calculation therefore describes a typical network only while
$K\ll d$. At $K=50$ and $d=512$ the log-variance is about $0.5$, so the
squared norm of a typical network is off by a factor of two in either
direction.

**Distances and nearest neighbors.** For two independent draws, $\mathbf
x-\mathbf y\sim\mathcal N(\mathbf 0,2\mathbf I)$, so norm concentration
applies to $(\mathbf x-\mathbf y)/\sqrt2$: each distance lies in
$(1\pm\varepsilon)\sqrt{2d}$ except with probability $2e^{-d\varepsilon^2/8}$.
The pair count used for directions then puts all distances among $n$ points in
that band at once when $d\geq(8/\varepsilon^2)\log(n^2/\delta)$, a dimension
that grows only with the logarithm of the number of points. When all distances
are nearly equal, the nearest and the farthest neighbors of a query differ by
a small relative margin, as the contrast measured above shows. Real embeddings
keep their contrast because their data lie near a set of low intrinsic
dimension. The same calculation also gives a construction: a random projection
to $k$ dimensions preserves all pairwise distances among $n$ points once $k$
is of order $\log n$, whatever the original dimension (Exercise
:exref:`ex_mdl-concentration-generalization-johnson-lindenstrauss`).

**Latent codes.** A decoder trained on codes
$\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ sees inputs of norm close
to $\sqrt d$. The prior mean, of norm $0$, and the midpoint of the chord
between two codes, of norm close to $\sqrt{d/2}$, are inputs it never saw
during training. For this reason, latent codes are usually interpolated
along the sphere rather than the chord (Exercise
:exref:`ex_mdl-concentration-generalization-latent-interpolation`).

Every bound in this section protects a quantity fixed *before* the data were
drawn. A learner picks the best of many candidates *after* seeing the data,
and the best of many fair coins looks biased.
:numref:`sec_mdl-generalization-bounds` shows how to account for that choice.

## Summary

How far can an average of $n$ independent terms be from its expectation, at
the $n$ we have? Chebyshev's inequality answers with a probability that falls
like $1/n$, too slowly for a guarantee that must almost never fail. The
Chernoff method applies Markov's inequality to $e^{\lambda X}$, where
independence turns the moment generating function of a sum into a product, and
optimizes over $\lambda$. With Hoeffding's lemma it gives, for losses in
$[0,1]$, the tail $2e^{-2nt^2}$ and an interval of half-width
$\sqrt{\log(2/\delta)/(2n)}$ that is valid at every $n$; Bernstein's
inequality tightens it when the variance is small. Applied to the $d$
coordinates of a random vector, the same calculation shows that a Gaussian
vector lies in a shell of radius $\sqrt d$ and thickness about $1/\sqrt2$, and
that exponentially many random directions are pairwise nearly orthogonal.
These results size test sets, tell when an initialization scale computed in
expectation describes the one network we train, and give the null model for
cosines between embeddings.

## Exercises

1. **Hoeffding with unequal ranges.** The proof of :eqref:`eq_mdl-hoeffding`
   assumed that every $X_i$ takes values in the same interval $[a,b]$.
    1. Redo the proof for independent $X_i\in[a_i,b_i]$ and show that

       $$P\bigl(|\bar X-\mathbb{E}[\bar X]|\geq t\bigr)\leq 2\exp\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right).$$

       Check that this bound reduces to :eqref:`eq_mdl-hoeffding` when all
       ranges agree.
    1. Let $x_1,\ldots,x_d$ be independent uniform random signs and let
       $\mathbf u\in\mathbb R^d$ be a unit vector. Apply the bound of (a)
       to the sum $\sum_i u_i x_i$ and derive the tail bound for the
       inner product, :eqref:`eq_mdl-inner-product-tail` in the text.
    1. Now let the $x_i$ be independent standard Gaussians. Derive the same
       bound $2e^{-s^2/2}$ from the Gaussian moment generating function
       $\mathbb{E}[e^{\lambda g}]=e^{\lambda^2/2}$ and the Chernoff method
       :eqref:`eq_mdl-chernoff`, and explain why the bound of (a) cannot be
       applied here.
   :exlabel:`ex_mdl-concentration-generalization-unequal-ranges`

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

1. **Comparing two classifiers on one test set.** ● Classifiers A and B are
   scored on the same $n=10{,}000$ examples. For 0/1 losses let
   $D_i=\ell_A(i)-\ell_B(i)\in\{-1,0,1\}$, so that the mean of the $D_i$ is
   the difference of the two error rates.
    1. Apply Hoeffding's inequality to the $D_i$, whose range has length 2.
       Show that the half-width at $\delta=0.05$ is about $\pm2.7\%$, so a
       difference of one percentage point cannot be certified this way.
    1. Suppose the two models disagree on a fraction $q$ of the examples.
       Show that $\textrm{Var}(D_i)\leq q$ and
       $|D_i-\mathbb{E}[D_i]|\leq2$, and apply :eqref:`eq_mdl-bernstein`
       with $M=2$. Compute the half-widths for $q=0.05$, $0.10$ and $0.20$.
       Below which disagreement rate can a one-point gap be certified?
    1. Compare with the asymptotic Gaussian interval $1.96\sqrt{q/n}$. What
       does Bernstein give up in exchange for validity at every $n$?
    1. Why is scoring both models on the *same* examples better than
       scoring them on two independent test sets of 10,000 examples each?
       Hint: compare the variance $2p(1-p)/n$ of a difference of two
       independent error estimates at $p=0.083$ with $q/n$ at $q=0.05$.
   :exlabel:`ex_mdl-concentration-generalization-two-classifiers`

1. [code] **Mode versus mass.** Let
   $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$.
    1. Show that the density at a point of norm $\sqrt d$ is $e^{-d/2}$
       times the density at the origin, and evaluate this ratio at
       $d=784$.
    1. Show that the density of $\|\mathbf x\|$ is proportional to
       $r^{d-1}e^{-r^2/2}$ and peaks at $r=\sqrt{d-1}$.
    1. Compute $P(\|\mathbf x\|\leq\sqrt d/2)$ at $d=784$ in three ways:
       with the bound :eqref:`eq_mdl-norm-concentration`, exactly with
       `scipy.stats.chi2.cdf`, and by Monte Carlo with $10^6$ draws. Which
       statement would you quote, and why does the exponent rather than the
       constant carry the message?
    1. A decoder with $d=512$ latent dimensions is evaluated at the prior
       mean $\mathbf 0$. Using the standard deviation $1/\sqrt2$ of the
       norm, how many standard deviations is that input from the norm of a
       typical code?
   :exlabel:`ex_mdl-concentration-generalization-mode-vs-mass`

[Discussions](https://d2l.discourse.group/t/36834)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.5]{.kicker}

Concentration inequalities<br>**why averages can be trusted**
:::
:::

::: {.slide title="830 Errors in 10,000: Chebyshev Says ±2.2% at 95% and ±500% at One in a Million"}
[The problem]{.kicker}

- A classifier errs on 830 of 10,000 test images: 8.3%.
- Chebyshev: a mean of $n$ draws with variance $\leq\tfrac14$ misses by $t$
  with probability $\leq1/(4nt^2)$.
- 95%: $\pm2.2\%$. One in a million: $\pm500\%$, which says nothing.
- An exponential bound (Hoeffding): $\pm1.4\%$ and $\pm2.7\%$.
- One in a million matters: a learner compares thousands of models.

::: {.d2l-note}
Markov's inequality on $e^{\lambda X}$ makes the tail exponential in $n$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- size a test set from a precision $t$ and a failure probability $\delta$;
- turn a bound on $\mathbb{E}[e^{\lambda X}]$ into an exponential tail;
- tell sub-Gaussian averages from sub-exponential squared norms;
- say where a 784-dimensional Gaussian vector lives, and at what angles;
- extend one tail bound to many objects with a union bound.

**Running examples:** the 8.3% classifier on 10,000 images; a fair coin at
deviation $t=0.1$; $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_{784})$.
:::

::: {.slide title="Exponential Tails for Averages and for Random Vectors"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Random Variables | densities, moments, covariance |
| Distributions | the named laws and their losses |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP, EM |
| Bayesian Computation | averaging over a posterior |
| **Concentration Inequalities** | **exponential tails of averages** |
| Uniform Convergence | guarantees after selection |
| Naive Bayes | one classifier, end to end |
:::

::: {.col .narrow}
- **Uses:** Markov, Chebyshev, the Gaussian interval.
- **Skips:** martingale methods; models chosen after seeing the data.
:::
:::
:::

::: {.slide title="Chebyshev Is Eight Orders of Magnitude Loose at n = 1,000"}
[Chebyshev's rate]{.kicker}

Fair coin, $P(|\hat p-\tfrac12|\geq0.1)$: the exact binomial sum against
Chebyshev's $\sigma^2/(nt^2)=25/n$.

@!mdl-concentration-generalization-chebyshev-s-rate-2

::: {.d2l-note}
Chebyshev falls like $1/n$; the exact tail falls along a straight line on a
log scale.
:::
:::

::: {.slide title="A Tail Bound Is the Expectation of a Function Above the Indicator"}
[Chernoff]{.kicker}

::: {.cols .vc}
::: {.col}
- $P(X\geq t)\leq\mathbb{E}[g(X)]$ for any $g\geq\mathbf 1\{x\geq t\}$.
- The exponential can be made small where the mass is, and it factorizes
  over sums.
:::

::: {.col .fig .big}
@fig:mdl-prob-conc-majorants
:::
:::
:::

::: {.slide title="Chernoff: Every λ > 0 Gives a Bound; Keep the Best"}
[Chernoff]{.kicker}

$$P(X\geq t)\leq\inf_{\lambda>0}e^{-\lambda t}\,\mathbb{E}\bigl[e^{\lambda X}\bigr]$$

- Markov on the increasing transform $e^{\lambda X}$.
- Independent sum $S$: $\mathbb{E}[e^{\lambda S}]=\prod_i\mathbb{E}[e^{\lambda X_i}]$,
  so the exponent grows like $n$.
- Fair coin, $n=100$, $t=0.1$: exponent $-10\lambda+12.5\lambda^2$.
- Minimum $-2$ at $\lambda=0.4$: each tail $\leq e^{-2}=0.14$.
- $\lambda=0.1$ is valid too, but gives only $e^{-0.875}=0.42$.
:::

::: {.slide title="One Fair Coin: cosh(λ/2) ≤ exp(λ²/8), So n Coins Give 2exp(−2nt²)"}
[The fair coin]{.kicker}

**I.** Centered flip $Z=\pm\tfrac12$: $\mathbb{E}[e^{\lambda Z}]=\cosh(\lambda/2)$, and
$\cosh u=\sum_k\frac{u^{2k}}{(2k)!}\leq\sum_k\frac{(u^2/2)^k}{k!}=e^{u^2/2}$,
since $(2k)!\geq2^kk!$.

. . .

**II.** $n$ flips: $P(\hat p-\tfrac12\geq t)\leq\exp(-\lambda nt+n\lambda^2/8)$.

. . .

**III.** Minimize at $\lambda=4t$, add both tails:
$\boxed{P(|\hat p-\tfrac12|\geq t)\leq2e^{-2nt^2}}$

::: {.d2l-note}
$n=1{,}000$, $t=0.1$: $2e^{-20}=4\times10^{-9}$, exact $2.7\times10^{-10}$.
:::
:::

::: {.slide title="Hoeffding's Lemma: a Bounded Variable's MGF Is No Worse Than a Gaussian's"}
[Hoeffding]{.kicker}

Centered $X\in[a,b]$, $\psi(\lambda)=\log\mathbb{E}[e^{\lambda X}]$:

**I.** Any $Y\in[a,b]$ has $\textrm{Var}(Y)\leq\mathbb{E}[(Y-\tfrac{a+b}2)^2]\leq(\tfrac{b-a}2)^2$.

. . .

**II.** $\psi(0)=\psi'(0)=0$; $\psi''(\lambda)$ is the variance of $X$ under the
law reweighted by $e^{\lambda x}$, still on $[a,b]$: $\psi''\leq(b-a)^2/4$.

. . .

**III.** Taylor: $\psi(\lambda)\leq\lambda^2(b-a)^2/8$, so
$\boxed{\mathbb{E}[e^{\lambda X}]\leq e^{\lambda^2(b-a)^2/8}}$

::: {.d2l-note}
The fair coin's $e^{\lambda^2/8}$ is the case $b-a=1$.
:::
:::

::: {.slide title="Hoeffding's Inequality in Four Lines"}
[Hoeffding]{.kicker}

Independent $X_i\in[a,b]$, $Z_i=X_i-\mathbb{E}X_i$, mean $\bar X$:

$$P(\bar X-\mathbb{E}\bar X\geq t)\leq e^{-\lambda nt}\prod_i\mathbb{E}[e^{\lambda Z_i}]\leq e^{-\lambda nt+n\lambda^2(b-a)^2/8}$$

- Minimize at $\lambda=4t/(b-a)^2$: exponent $-2nt^2/(b-a)^2$.
- Same for the lower tail; the union bound adds the two.

$$\boxed{P(|\bar X-\mathbb{E}\bar X|\geq t)\leq2e^{-2nt^2/(b-a)^2}}$$

::: {.d2l-note .rule}
Only boundedness and independence, at every finite $n$.
:::
:::

::: {.slide title="Hoeffding Certifies ±1% at 95% with 18,445 Examples"}
[Hoeffding]{.kicker}

With probability $1-\delta$:
$\;|\bar X-\mathbb{E}\bar X|\leq(b-a)\sqrt{\log(2/\delta)/(2n)}$.

- Losses in $[0,1]$, $t=0.01$, $\delta=0.05$:
  $n=\log40/(2\cdot10^{-4})=18{,}445$.
- 830 errors in 10,000: Hoeffding $\pm1.36\%$, Gaussian
  $1.96\sqrt{0.083\cdot0.917/10^4}=\pm0.54\%$.
- Ratio $2.5$: $\hat\sigma=0.28$ becomes $0.5$, and $1.96$ becomes
  $\sqrt{2\log40}=2.72$.

::: {.d2l-note}
The range replaces $\hat\sigma$: safe at every $n$, conservative for small spreads.
:::
:::

::: {.slide title="How Many Examples for ±1% at 99%? What Do 9 Heads in 13 Flips Show?"}
[Check]{.kicker}

Use $n=\log(2/\delta)/(2t^2)$ for losses in $[0,1]$, and
$2e^{-2nt^2}$ for the coin with $t=9/13-\tfrac12$.

. . .

- $\delta=0.01$: $n=\log200/(2\cdot10^{-4})\approx26{,}500$.
- At one in a million: Hoeffding $72{,}500$; Chebyshev $2.5$ billion.
- 13 flips: bound $2e^{-0.96}=0.77$, nothing certified (exact: $0.27$).
:::

::: {.slide title="On a Log Scale the Exact Tail and Hoeffding Are Parallel Lines" layout="tight"}
[The fair coin in numbers]{.kicker}

@!mdl-concentration-generalization-the-tail-race-in-code

Slopes per flip: Hoeffding $0.02$, exact $0.0201$; Chebyshev wins for $18\leq n\leq107$.
:::

::: {.slide title="Sub-Gaussian Means an MGF Bound That Survives Averaging"}
[Sub-Gaussian]{.kicker}

$X$ is sub-Gaussian with proxy $\sigma^2$ if
$\mathbb{E}[e^{\lambda(X-\mathbb{E}X)}]\leq e^{\lambda^2\sigma^2/2}$ for all $\lambda$.

- Tail: $P(|X-\mathbb{E}X|\geq t)\leq2e^{-t^2/(2\sigma^2)}$.
- Gaussian: proxy $\sigma^2$, with equality.
- 0/1 coin: proxy $\tfrac14$; random sign $\pm1$: proxy $1$.
- Mean of $n$ independent terms: proxy $\sigma^2/n$.
:::

::: {.slide title="Squares Are Only Sub-Exponential: Gaussian Rate Near the Mean, Exponential Far Out"}
[Sub-exponential]{.kicker}

- $x\sim\mathcal N(0,1)$: $\mathbb{E}[e^{\lambda(x^2-1)}]=e^{-\lambda}/\sqrt{1-2\lambda}$,
  infinite for $\lambda\geq\tfrac12$.
- Sub-exponential $(\nu,\alpha)$: the bound $e^{\lambda^2\nu^2/2}$ holds only
  for $|\lambda|\leq1/\alpha$.
- Chernoff: minimizer $t/\nu^2$ if inside the window, else clamp at $1/\alpha$.

$$\boxed{P(|\bar X-\mathbb{E}\bar X|\geq t)\leq2\exp\Bigl(-\tfrac n2\min\bigl(\tfrac{t^2}{\nu^2},\tfrac t\alpha\bigr)\Bigr)}$$
:::

::: {.slide title="Bernstein Uses the True Variance: ±0.39% Instead of ±1.36% at 2% Error"}
[Bernstein]{.kicker}

$|X_i-\mathbb{E}X_i|\leq M$, $\textrm{Var}(X_i)\leq\sigma^2$:
$\;P(|\bar X-\mathbb{E}\bar X|\geq t)\leq2\exp\bigl(-\tfrac{nt^2}{2(\sigma^2+Mt/3)}\bigr)$

@!mdl-concentration-generalization-sub-gaussian-and-sub-exponential-variables-1

::: {.d2l-note}
Small $t$: the Gaussian tail with the true $\sigma^2$. Large $t$: only exponential.
:::
:::

::: {.slide title="A 784-Dimensional Gaussian Never Comes Within 14 of Its Mode"}
[The thin shell]{.kicker}

::: {.cols .vc}
::: {.col}
- Density of $\|\mathbf x\|\propto r^{d-1}e^{-r^2/2}$, peak at $\sqrt{d-1}$.
- $d=784$: $P(\|\mathbf x\|\leq14)\approx10^{-110}$; a typical norm is $28.0\pm0.7$.
:::

::: {.col .fig .big}
@fig:mdl-prob-conc-shell
:::
:::
:::

::: {.slide title="Norm Concentration Is an Instance of the Sub-Exponential Bound"}
[The thin shell]{.kicker}

**I.** $\mathbb{E}[e^{\lambda(x^2-1)}]\leq e^{2\lambda^2}$ for $|\lambda|\leq\tfrac14$:
$(\nu,\alpha)=(2,4)$.

. . .

**II.** Average of $d$ squares, $t=\varepsilon\leq1$: exponent
$\tfrac d2\min(\tfrac{\varepsilon^2}4,\tfrac\varepsilon4)=\tfrac{d\varepsilon^2}8$.

. . .

**III.** $|z-1|\geq\varepsilon\Rightarrow|z^2-1|=|z-1|(z+1)\geq\varepsilon$ for $z\geq0$:

$$\boxed{P\bigl(\bigl|\|\mathbf x\|/\sqrt d-1\bigr|\geq\varepsilon\bigr)\leq2e^{-d\varepsilon^2/8}}$$

::: {.d2l-note}
$d=784$, norm below 14: $\varepsilon=\tfrac12$, bound $2e^{-24.5}\approx5\times10^{-11}$.
:::
:::

::: {.slide title="The Shell's Thickness Is 1/√2 Whatever d"}
[The thin shell]{.kicker}

$\textrm{Var}(\|\mathbf x\|^2)=2d$ and
$\|\mathbf x\|\approx\sqrt d+(\|\mathbf x\|^2-d)/(2\sqrt d)$, so
$\textrm{sd}(\|\mathbf x\|)\approx\sqrt{2d}/(2\sqrt d)=1/\sqrt2$.

@!mdl-distributions-multivariate-gaussian-1

::: {.d2l-note}
Radius $\sqrt d$, thickness about $0.71$; $|\cos|$ matches $\sqrt{2/(\pi d)}$ from $d=20$ on.
:::
:::

::: {.slide title="Random Directions Are Nearly Orthogonal"}
[Near-orthogonality]{.kicker}

- Gaussian $\mathbf x$, unit $\mathbf u$: $\langle\mathbf x,\mathbf u\rangle\sim\mathcal N(0,1)$ exactly.
- Signs $x_i=\pm1$: $u_ix_i\in[-|u_i|,|u_i|]$; Hoeffding gives
  $P(|\langle\mathbf x,\mathbf u\rangle|\geq s)\leq2e^{-s^2/2}$.
- $\|\mathbf x\|=\sqrt d$, so $P(|\cos|\geq\varepsilon)\leq2e^{-d\varepsilon^2/2}$.
- $m$ vectors, $<m^2/2$ pairs: all $|\cos|<\varepsilon$ with probability
  $\geq1-\delta$ if $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$.
:::

::: {.slide title="d = 784: How Small Is a Typical |cos|, and How Many Directions Stay Below 0.2?"}
[Check]{.kicker}

Use $\mathbb{E}|\cos|\approx\sqrt{2/(\pi d)}$ and $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$
with $\delta=0.05$.

. . .

- Typical $|\cos|\approx\sqrt{2/(784\pi)}=0.028$.
- $\varepsilon=0.2$: $0.22\cdot e^{7.84}\approx570$ directions.
- $\varepsilon=0.3$: $0.22\cdot e^{17.64}\approx10^7$ directions.
:::

::: {.slide title="A Union Bound Flattens All Pairwise Distances Once d ≳ (8/ε²) log(n²/δ)"}
[Distances]{.kicker}

$\mathbf x-\mathbf y\sim\mathcal N(\mathbf 0,2\mathbf I)$: each distance in
$(1\pm\varepsilon)\sqrt{2d}$ except with probability $2e^{-d\varepsilon^2/8}$.

@!mdl-concentration-generalization-measuring-the-shell

::: {.d2l-note .rule}
One exponential tail, many objects: the dimension grows only like $\log n$.
:::
:::

::: {.slide title="Three Consequences: Initialization, Nearest Neighbors, Latent Codes"}
[Deep learning]{.kicker}

- One He-initialized ReLU layer: $\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$ has
  mean $1$, variance $5/d$: about 10% at $d=512$.
- $K$ layers: log-variance $5K/d$; the variance calculation describes a
  typical network only while $K\ll d$.
- Random points: median contrast below 0.1 at $d=2000$; real embeddings have low
  intrinsic dimension.
- Codes of norm $\sqrt d$: the prior mean (norm 0) and a chord midpoint
  (norm $\sqrt{d/2}$) are unseen inputs, so interpolate on the sphere.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Chernoff:** Markov on $e^{\lambda X}$; independent sums multiply MGFs.
- **Hoeffding:** $2e^{-2nt^2/(b-a)^2}$; 8.3% on 10,000 images: $\pm2.7\%$ at
  one in a million, where Chebyshev says $\pm500\%$; 18,445 examples for
  $\pm1\%$ at 95%.
- **Fair coin:** $2e^{-20}$ at $n=1{,}000$, within a factor 15 of the truth.
- **Bernstein:** $\pm0.39\%$ at a 2% error rate.
- **High dimension:** norm $28.0\pm0.7$ at $d=784$; $|\cos|\approx0.03$.

**Next:** the best of 1,000 fair coins shows 66% heads; guarantees for a
model chosen after the data.
:::
