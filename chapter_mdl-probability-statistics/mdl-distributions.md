# Distributions
:label:`sec_mdl-distributions`

The previous section gave us a language for any random quantity: a mass
function or a density, a mean, a variance. A model that predicts a label
uses this language whether or not it says so. A binary classifier outputs one
number, the probability that the label is 1. A language model outputs a
probability for every token in its vocabulary, and the numbers sum to one. A
regression network outputs a predicted value and is trained with squared
error. Each of these models has chosen a particular distribution for its
labels, and that choice fixes the loss it minimizes. Which distributions are
these, where do they come from, and what are their means and variances?

A coin will carry us through most of the answer. Suppose it lands heads 9
times in 13 flips. If the coin is fair, how likely is that? To answer, we need the
distribution of the number of heads, and the same distribution describes how
many test examples a classifier gets right. We will find that the answer is
about 9%.

The idea of this section is that a handful of distributions grow out of a
single coin flip. Counting heads gives the binomial. Many flips that each
rarely come up heads give the Poisson, and the time until the next rare event
gives the exponential. Many ordinary flips, added up and rescaled, give the
Gaussian. Because each law is built from simpler pieces, its mean and variance
follow from the construction, by a short argument that is different each time.

The choice of law matters in practice. Each law, read as a model of the label,
turns into a familiar loss: binary cross-entropy, softmax cross-entropy,
squared error, absolute error. A wrong choice has measurable consequences. A
count model whose variance is too small, for example, makes every error bar
computed from it too narrow.

We build on the densities, means and variances of
:numref:`sec_mdl-random_variables`. Two questions stay open at the end: why
almost all of these laws share one algebraic form, and how well 13 flips
determine the bias of a coin whose bias we do not know.
:numref:`fig_mdl-prob-family-tree` shows how the laws of this section are
built from one another.

![Every law of this section grows out of one coin flip. Arrows with filled heads construct a new law (a sum, a waiting time, a random sign, a vector of coordinates, equal probabilities); dashed arrows with open heads take a limit as the number of flips or points grows. The Poisson and the exponential describe one stream of events, counted and timed.](../img/mdl-prob-family-tree.svg)
:label:`fig_mdl-prob-family-tree`

```{.python .input #distributions-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
import numpy as np
```

```{.python .input #distributions-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #distributions-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import numpy as np
```

```{.python .input #distributions-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Discrete Distributions

A discrete random variable takes values in a countable set such as $\{0,1\}$
or $\{0,1,2,\dots\}$. Its distribution is a probability mass function
$P(X=k)\ge0$ with $\sum_k P(X=k)=1$, its mean is
$\mathbb{E}[X]=\sum_k k\,P(X=k)$, and its variance is
$\textrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$. The laws below are built
from one another. The binomial counts Bernoulli trials, the Poisson is a limit
of the binomial, the categorical extends the Bernoulli to $K$ outcomes, the
multinomial counts categorical draws, and the discrete uniform is the
categorical with equal probabilities. :numref:`fig_mdl-prob-discrete-pmfs`
shows them in the order of the text.

![Four probability mass functions on the running examples. Bernoulli$(9/13)$: one flip of a coin with the observed frequency of heads. Binomial$(13,\frac{1}{2})$: the number of heads in 13 fair flips; the orange stem is $P(X=9)=715/8192\approx0.087$. Poisson$(4)$: requests per minute at an average of four. A categorical over four classes from the logits $(2, 1, 0.1, -1)$, against the uniform $1/4$. The binomial and the Poisson peak near their means, $6.5$ and $4$, and share a vertical scale; the two panels with few outcomes share another.](../img/mdl-prob-discrete-pmfs.svg)
:label:`fig_mdl-prob-discrete-pmfs`

### Bernoulli

The simplest random quantity has two outcomes: a flip comes up heads or tails,
a user clicks or does not, a prediction is right or wrong. One number describes
it completely, the probability $p$ of the outcome we call $1$.

A *Bernoulli* random variable $X\sim\mathrm{Bernoulli}(p)$ takes the value $1$
with probability $p$ and $0$ with probability $1-p$, where $0\le p\le1$. Both
cases fit in one formula,

$$
P(X=x) = p^{x}(1-p)^{1-x}, \qquad x\in\{0,1\}.
$$
:eqlabel:`eq_mdl-bernoulli_pmf`

The exponent selects the factor: for $x=1$ the formula reads $p$, and for $x=0$
it reads $1-p$.

Because $0^2=0$ and $1^2=1$, the variable equals its own square, $X^2=X$.
Both moments follow at once:

$$
\mathbb{E}[X] = 1\cdot p + 0\cdot(1-p) = p,
\qquad
\textrm{Var}(X) = \mathbb{E}[X^2]-\mathbb{E}[X]^2 = p - p^2 = p(1-p).
$$

The mean of a $0/1$ variable is the probability of a $1$, and its variance is
the product of the two probabilities. A fair coin has mean $\tfrac12$ and
variance $\tfrac14$. The variance $p(1-p)$ is $0.25$ at $p=\tfrac12$, $0.09$ at
$p=0.1$ and at $p=0.9$, and $0$ at $p\in\{0,1\}$: the fair coin is the least
predictable, and a coin with $p=0$ or $p=1$ is a constant. If we use the
observed frequency $9/13\approx0.69$ of our coin as $p$, the variance is
$\tfrac{9}{13}\cdot\tfrac{4}{13}\approx0.21$.

A binary classifier outputs an estimate $\hat p$ of the probability that the
label $y$ is $1$. By :eqref:`eq_mdl-bernoulli_pmf`, the negative
log-probability it assigns to the observed label is
$-\log P(y)=-y\log\hat p-(1-y)\log(1-\hat p)$, the *binary cross-entropy*. A
classifier trained with this loss is fitting a Bernoulli distribution to each
label.

Does the frequency of heads in many flips approach $p$, and is the variance
largest at the fair coin? The cell flips the running coin $10{,}000$ times.

```{.python .input #distributions-bernoulli}
rng = np.random.default_rng()
p = 9 / 13
flips = rng.random(10000) < p          # True (heads) with probability p
print(f'p(1-p) = {p * (1 - p):.3f}')
print(f'frequency of heads in 10,000 flips: {flips.mean():.3f}   (p = {p:.3f})')
print('p(1-p) at p = 0.1, 0.5, 0.9:', [round(q * (1 - q), 2) for q in (0.1, 0.5, 0.9)])
```

The frequency lands within about $0.01$ of $p\approx0.692$ (its standard
deviation is $\sqrt{0.21/10{,}000}\approx0.005$), and $p(1-p)$ is largest at
$p=\tfrac12$.

### Binomial

Our question about the coin is about a count: how many heads in 13 flips, and
how likely are exactly 9?

Start with two flips. The outcomes are TT, HT, TH and HH, so

$$
P(X=0)=(1-p)^2,\qquad P(X=1)=2p(1-p),\qquad P(X=2)=p^2 .
$$

The factor $2$ counts the two orderings of one head and one tail, and the three
probabilities sum to $(p+1-p)^2=1$.

In general, the number of heads in $n$ flips is a sum of $n$ indicator
variables, one per flip. If the flips $X_1,\dots,X_n$ are independent and each
is $\mathrm{Bernoulli}(p)$ with the same $p$, their sum has the *binomial*
distribution,

$$
X = \sum_{i=1}^{n} X_i \sim \mathrm{Binomial}(n,p).
$$
:eqlabel:`eq_mdl-binomial_sum`

A binomial count is a sum of $n$ independent coin flips with a common bias.
Exactly $k$ heads can occur in $\binom nk$ orderings, one for each choice of
which $k$ flips are heads, and by independence each ordering has probability
$p^k(1-p)^{n-k}$. Hence

$$
P(X=k) = \binom{n}{k}\,p^k(1-p)^{n-k}, \qquad k=0,1,\dots,n.
$$
:eqlabel:`eq_mdl-binomial_pmf`

The probability of a count is the number of orderings times the probability of
one ordering. For $n=2$ it gives the three numbers above. For our coin, 9 heads
in 13 flips can occur in $\binom{13}{9}=715$ orderings, each of probability
$2^{-13}=1/8192$, so

$$
P(X=9) = \frac{715}{8192} \approx 0.087 .
$$

Nine heads *or more* has probability $(715+286+78+13+1)/8192=1093/8192\approx0.133$.
Whether a tail probability of $0.133$ is reason to doubt the coin is a
question for statistics, which this section does not take up.

**Mean and variance.** Summing $k$ times the pmf would be laborious. The
construction :eqref:`eq_mdl-binomial_sum` gives both moments in one line each.
Expectation is linear for any random variables, dependent or not, so

$$
\mathbb{E}[X] = \sum_{i=1}^n \mathbb{E}[X_i] = np .
$$

Variance adds over *independent* summands, because the covariance of two
independent variables is zero (:numref:`sec_mdl-random_variables`), so

$$
\textrm{Var}(X) = \sum_{i=1}^n \textrm{Var}(X_i) = np(1-p).
$$

Thus $n$ flips have $n$ times the mean and $n$ times the variance of one flip.
For the coin the mean is $6.5$, the variance $3.25$ and the standard deviation
$\sqrt{3.25}\approx1.80$, so 9 heads lie $2.5/1.80\approx1.4$ standard
deviations above the mean.

The two formulas need different hypotheses. The mean $np$ holds even if the
flips are dependent, and with different probabilities $p_i$ it becomes
$\sum_ip_i$. The variance needs independence. If the 13 "flips" were one flip
copied 13 times, $X$ would be $0$ or $13$, and its variance would be
$13^2\cdot\tfrac14=42.25$, thirteen times $np(1-p)$.

**The successive ratio.** The pmf can be computed without any factorial.
When we divide consecutive binomial coefficients, the factorials cancel:

$$
\frac{\binom{n}{k+1}}{\binom nk}
= \frac{k!\,(n-k)!}{(k+1)!\,(n-k-1)!} = \frac{n-k}{k+1},
\qquad\text{so}\qquad
\frac{P(X=k+1)}{P(X=k)} = \frac{n-k}{k+1}\cdot\frac{p}{1-p}.
$$

The ratio counts orderings: a set of $k+1$ heads arises from $k+1$ different
sets of $k$ heads by adding one of the $n-k$ remaining flips, and the factor
$p/(1-p)$ trades the probability of a tail for that of a head. Starting from
$P(X=0)=(1-p)^n$ and multiplying by the ratio builds the whole pmf. The ratio
exceeds $1$ while $k+1<(n+1)p$ and falls below $1$ afterwards, so the pmf rises
to a single peak near the mean and then falls.

How likely are 9 heads in 13 fair flips? The cell builds the pmf by the ratio,
in a function we reuse below, and compares it with direct counting.

```{.python .input #distributions-binomial}
def binomial_pmf(n, p, kmax):
    """P(X=k) for k = 0..kmax, built from P(X=0) by the successive ratio."""
    pmf = [(1 - p) ** n]
    for k in range(kmax):
        pmf.append(pmf[-1] * max(n - k, 0) / (k + 1) * p / (1 - p))
    return np.array(pmf)

n, p = 13, 0.5
pmf = binomial_pmf(n, p, n)
print(f'P(X=9): by the ratio {pmf[9]:.4f}, by counting {math.comb(13, 9) / 2**13:.4f}')
print(f'P(X>=9) = {pmf[9:].sum():.4f}')
print(f'mean np = {n * p}, variance np(1-p) = {n * p * (1 - p)}')
```

Both routes give $0.0873$, and the tail $P(X\ge9)$ is $0.1334$.

### Poisson

Some counts have no natural number of trials. A web server receives requests
at an average rate of $\lambda=4$ per minute, and nobody fixes in advance how
many requests may arrive. What is the chance of a minute with no request at
all, or of a burst of 10 or more?

Slice the minute into $n$ short pieces, so short that no piece holds two
requests, and treat each piece as an independent $\mathrm{Bernoulli}(4/n)$
trial. The count is then $\mathrm{Binomial}(n,4/n)$. Its mean is
$n\cdot\tfrac4n=4$ for every $n$, and its variance $4(1-\tfrac4n)$ approaches
$4$. The chance of a silent minute is $(1-\tfrac4n)^n$, which is $0.0115$ at
$n=20$, $0.0169$ at $n=100$ and $0.0182$ at $n=1000$, approaching
$e^{-4}\approx0.0183$. The numbers settle as the slices shrink, which suggests
that the binomial has a limit.

**Derivation of the limit.** Take $\mathrm{Binomial}(n,\lambda/n)$ and fix $k$.
The pmf splits into four factors, three of which have simple limits:

$$
\binom{n}{k}\Bigl(\frac\lambda n\Bigr)^k\Bigl(1-\frac\lambda n\Bigr)^{n-k}
= \underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to\,1}
  \cdot \frac{\lambda^k}{k!}
  \cdot \underbrace{\Bigl(1-\frac\lambda n\Bigr)^{n}}_{\to\,e^{-\lambda}}
  \cdot \underbrace{\Bigl(1-\frac\lambda n\Bigr)^{-k}}_{\to\,1} .
$$

The first factor is a product of $k$ ratios that each tend to $1$, the third is
the limit $(1-\lambda/n)^n\to e^{-\lambda}$ that defines the exponential
function, and the last tends to $1$ because $k$ is fixed. The same computation
works for any success probabilities $p_n$ with $np_n\to\lambda$, since then
$(1-p_n)^n\to e^{-\lambda}$ as well. This limit is the *law of rare events*. The
surviving factors are the *Poisson* pmf with rate $\lambda>0$,

$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k=0,1,2,\dots
$$
:eqlabel:`eq_mdl-poisson_pmf`

and we write $X\sim\mathrm{Poisson}(\lambda)$. The Poisson describes a count
made of many independent chances, each small, with a fixed expected total
$\lambda$. Its probabilities sum to one because
$\sum_{k\ge0}\lambda^k/k!=e^{\lambda}$.

**Mean and variance.** We compute the moments from the pmf directly. The
factor $k$ cancels against the $k!$, and the index shift $j=k-1$ turns the
remaining sum into the total probability of a Poisson:

$$
\mathbb{E}[X] = \sum_{k\ge1} k\,\frac{e^{-\lambda}\lambda^k}{k!}
= \lambda\sum_{k\ge1}\frac{e^{-\lambda}\lambda^{k-1}}{(k-1)!}
= \lambda\sum_{j\ge0}\frac{e^{-\lambda}\lambda^{j}}{j!} = \lambda .
$$

The same shift applied twice gives $\mathbb{E}[X(X-1)]=\lambda^2$, so

$$
\textrm{Var}(X) = \mathbb{E}[X(X-1)] + \mathbb{E}[X] - \mathbb{E}[X]^2
= \lambda^2+\lambda-\lambda^2 = \lambda .
$$

The Poisson's mean and variance are equal. For the server,
$P(X=0)=e^{-4}\approx0.018$, $P(X=4)=\tfrac{4^4}{4!}e^{-4}\approx0.195$ and
$P(X\ge10)\approx0.008$: a silent minute occurs about once an hour, and a burst
of ten or more about once in two hours. If request counts measured over many
minutes have a variance several times their mean, the Poisson model is wrong
for them, and error bars computed from it are too narrow. The usual repair, a
Poisson whose rate is itself random, is
exercise :exref:`ex_mdl-distributions-gamma-poisson-mixture`.

**The Poisson process.** Run the slicing over the whole timeline, with rate
$\lambda$ per unit time. A window of length $t$ holds $nt$ slices, so its count
is $\mathrm{Binomial}(nt,\lambda/n)$, which tends to
$\mathrm{Poisson}(\lambda t)$. Windows that do not overlap use disjoint sets of
slices, so their counts are independent. This stream of events is the
*Poisson process* with rate $\lambda$, and the exponential distribution below
describes the times between its events.

We build the Poisson pmf by its own successive ratio,
$P(X=k+1)/P(X=k)=\lambda/(k+1)$, read off the three probabilities for the
server, and check the mean and variance on simulated counts.

```{.python .input #distributions-poisson}
def poisson_pmf(lam, kmax):
    """P(X=k) for k = 0..kmax, built from P(X=0) = e^-lam by the ratio lam/(k+1)."""
    pmf = [math.exp(-lam)]
    for k in range(kmax):
        pmf.append(pmf[-1] * lam / (k + 1))
    return np.array(pmf)

pmf = poisson_pmf(4.0, 60)
print(f'P(0) = {pmf[0]:.3f}, P(4) = {pmf[4]:.3f}, P(X>=10) = {pmf[10:].sum():.3f}')
counts = rng.poisson(4.0, 100000)
print(f'100,000 simulated minutes: mean {counts.mean():.2f}, '
      f'variance {counts.var(ddof=1):.2f}')
```

The pmf gives $0.018$, $0.195$ and $0.008$, and the simulated counts have mean
and variance both close to $4$.

**The binomial-to-Poisson limit.** How fast does
$\mathrm{Binomial}(n,4/n)$ approach $\mathrm{Poisson}(4)$? The plot overlays
the two, and the cell measures the gap by the *total-variation distance*
$\tfrac12\sum_k|P_{\mathrm{Bin}}(k)-P_{\mathrm{Poi}}(k)|$, which is the largest
difference between the probabilities that the two laws give to any one event.

```{.python .input #distributions-binomial-to-poisson}
ns, ks = (5, 20, 100), np.arange(15)
d2l.plot(ks, [binomial_pmf(n, 4 / n, 14) for n in ns] + [poisson_pmf(4.0, 14)],
         xlabel='k', ylabel='P(X=k)', fmts=('o-', 's--', '^-.', 'k*:'),
         legend=[f'Binomial({n}, 4/{n})' for n in ns] + ['Poisson(4)'],
         figsize=(5, 3))
for n in (20, 100, 1000):
    tv = 0.5 * np.abs(binomial_pmf(n, 4 / n, 60) - poisson_pmf(4.0, 60)).sum()
    print(f'n = {n:4d}: total-variation distance {tv:.3f}')
```

At $n=5$ the binomial cannot exceed five requests, and its peak is too high.
By $n=100$ the two pmfs coincide to plotting accuracy. The distance is $0.056$
at $n=20$, $0.010$ at $n=100$ and $0.001$ at $n=1000$: each tenfold increase
in $n$ cuts it about tenfold, so the error of the Poisson approximation falls
like $1/n$.

### Categorical and Multinomial

A classifier with $K$ classes, or a language model choosing the next token,
has more than two outcomes. The categorical distribution is the coin with $K$
faces.

A *categorical* random variable takes values in $\{1,\dots,K\}$ with
probabilities given by a vector $\mathbf p=(p_1,\dots,p_K)$, where $p_k\ge0$ and
$\sum_kp_k=1$:

$$
P(X=k) = p_k .
$$
:eqlabel:`eq_mdl-categorical_pmf`

For $K=2$ it is the Bernoulli. A network produces the $p_k$ from real-valued
scores $z_1,\dots,z_K$, the *logits*, through the *softmax*

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} .
$$
:eqlabel:`eq_mdl-softmax`

Exponentiating makes every score positive, and dividing by the sum makes the
scores add to one. The logits $\mathbf z=(2,1,0.1,-1)$, for example, give
$\mathbf p\approx(0.638,0.235,0.095,0.032)$. With the label written as a one-hot
vector $\mathbf y$, the negative log-probability of the observed class is
$-\log p_y=-\sum_ky_k\log p_k$, the *softmax cross-entropy*. Training a
classifier with it fits a categorical distribution to each label.

**The discrete uniform.** When all probabilities are equal, the categorical
on $\{1,\dots,n\}$ with $p_k=1/n$ is the *discrete uniform*
$\mathrm{Unif}\{1,\dots,n\}$: a fair die, or an index chosen uniformly at
random. Its moments follow from the sums $\sum_{i=1}^n i=\tfrac{n(n+1)}2$ and
$\sum_{i=1}^n i^2=\tfrac{n(n+1)(2n+1)}6$:

$$
\mathbb{E}[X] = \frac1n\sum_{i=1}^n i = \frac{n+1}{2},
\qquad
\textrm{Var}(X) = \frac{(n+1)(2n+1)}{6} - \frac{(n+1)^2}{4}
= \frac{(n+1)\bigl(2(2n+1)-3(n+1)\bigr)}{12} = \frac{n^2-1}{12}.
$$

The mean is the midpoint, and the variance grows like $n^2$ because the values
spread over a range of length $n-1$. A die has mean $3.5$ and variance
$35/12\approx2.92$.

**The multinomial.** Draw $n$ independent categorical samples and count how
often each class occurs. The counts $\mathbf x=(x_1,\dots,x_K)$, with
$\sum_kx_k=n$, have the *multinomial* pmf

$$
P(\mathbf x) = \frac{n!}{x_1!\cdots x_K!}\,\prod_{k=1}^{K} p_k^{x_k}.
$$
:eqlabel:`eq_mdl-multinomial_pmf`

As for the binomial, the coefficient counts the orderings of the $n$ draws
that produce these counts, and the product is the probability of one
ordering. For $K=2$ the coefficient is $\binom{n}{x_1}$, and the multinomial is
the binomial again: $13!/(9!\,4!)=715$. Each count on its own is
$\mathrm{Binomial}(n,p_k)$, since a draw either lands in class $k$ or not, so
$\mathbb{E}[x_k]=np_k$ and $\textrm{Var}(x_k)=np_k(1-p_k)$. The whole count
vector has mean $n\mathbf p$ and covariance
$n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$, which
exercise :exref:`ex_mdl-distributions-multinomial-covariance` derives; its
off-diagonal entries are negative because the classes compete for the same
$n$ draws. With $n=1$ the count vector is the one-hot vector of a single
categorical draw.

Does a sampler reproduce the softmax probabilities of the example logits, and
does the two-class multinomial coefficient equal the binomial coefficient
$\binom{13}{9}$?

```{.python .input #distributions-categorical}
z = np.array([2.0, 1.0, 0.1, -1.0])          # logits for K = 4 classes
p_cat = np.exp(z) / np.exp(z).sum()           # softmax
draws = rng.choice(4, size=10000, p=p_cat)
print('softmax p   =', p_cat.round(3))
print('frequencies =', (np.bincount(draws, minlength=4) / 10000).round(3))
print('13! / (9! 4!) =', math.factorial(13) // (math.factorial(9) * math.factorial(4)))
```

The frequencies of $10{,}000$ draws agree with $\mathbf p$ to within about a
percentage point, and the coefficient is $715$.

Do the closed-form sums give the mean and variance of a die?

```{.python .input #distributions-discrete-uniform}
n = 6
k = np.arange(1, n + 1)
p_k = np.full(n, 1 / n)
mean = (k * p_k).sum()
print(f'mean:     sum of k p_k = {mean:.3f},   (n+1)/2 = {(n + 1) / 2:.3f}')
print(f'variance: sum of k^2 p_k - mean^2 = {(k**2 * p_k).sum() - mean**2:.3f},'
      f'   (n^2-1)/12 = {(n**2 - 1) / 12:.3f}')
```

Both pairs agree: $3.5$ and $2.917$.

## Continuous Distributions

A continuous random variable has a density $p(x)\ge0$ with $\int p(x)\,dx=1$.
Probabilities are areas under the density, and means and variances are
integrals, as in :numref:`sec_mdl-random_variables`. The laws below come
in the order in which they are built. The exponential comes from the Poisson
process. The uniform is the raw material of every sampler. The Gaussian is the
limit of sums of ordinary terms, and the Laplace is the exponential folded to
both sides.

### Exponential

Return to the server. The Poisson tells us how many requests arrive in a
minute. An engineer choosing a timeout asks a different question about the
same stream of requests: how long until the next one?

The two questions share an answer. The wait for the next request exceeds one
minute exactly when the minute contains no request, so
$P(\text{wait}>1)=P(N=0)=e^{-4}\approx0.018$, the same number as the
Poisson's $P(0)$. :numref:`fig_mdl-prob-poisson-timeline` shows the one stream
of events that both laws describe.

![One stream of requests at an average rate of four per minute, seen two ways. (a) Event times over three minutes; the unit windows hold $N = 4$, $2$ and $5$ requests, and $T_1$, $T_2$, $T_3$ are the first three waits. (b) The number of requests in a minute is Poisson$(4)$; a silent minute has probability $e^{-4}\approx0.018$. (c) The probability that a wait exceeds $t$ minutes is $e^{-4t}$, a straight line on the logarithmic axis; a wait longer than a minute and a silent minute are the same event.](../img/mdl-prob-poisson-timeline.svg)
:label:`fig_mdl-prob-poisson-timeline`

**Derivation.** The same argument works for a wait of any length $t\ge0$. In a Poisson process with
rate $\lambda$, the wait $T$ from time $0$ to the first event exceeds $t$
exactly when the window $[0,t]$ is empty, and the count in that window is
$\mathrm{Poisson}(\lambda t)$. Hence

$$
P(T>t) = P\bigl(N(t)=0\bigr) = e^{-\lambda t}, \qquad t\ge0 .
$$

The cumulative distribution function is $F(t)=1-e^{-\lambda t}$, and its
derivative is the density of the *exponential* distribution with rate
$\lambda>0$,

$$
p(t) = \lambda e^{-\lambda t}\quad (t\ge0),
\qquad
p(t) = 0\quad (t<0),
\qquad
F(t) = 1 - e^{-\lambda t},
\qquad T\sim\mathrm{Exp}(\lambda).
$$
:eqlabel:`eq_mdl-exponential_pdf`

The density is the rate at which the probability of an empty window decays,
so the process determines it. After each event the future
slices of the process are fresh independent trials, so the same law holds for
the wait after any event: the gaps between consecutive events are independent
$\mathrm{Exp}(\lambda)$ variables.

**Mean and variance.** Two integrals by parts do the work:
$\int_0^\infty u\,e^{-u}\,du=[-ue^{-u}]_0^\infty+\int_0^\infty e^{-u}\,du=1$
and
$\int_0^\infty u^2e^{-u}\,du=[-u^2e^{-u}]_0^\infty+2\int_0^\infty u\,e^{-u}\,du=2$.
Substituting $u=\lambda t$,

$$
\mathbb{E}[T] = \int_0^\infty t\,\lambda e^{-\lambda t}\,dt = \frac1\lambda,
\qquad
\mathbb{E}[T^2] = \frac{2}{\lambda^2},
\qquad
\textrm{Var}(T) = \frac{2}{\lambda^2}-\frac1{\lambda^2} = \frac1{\lambda^2}.
$$

More frequent events mean shorter waits. At four requests per minute the mean
wait is $\tfrac14$ minute, or 15 seconds. The median $m$ solves
$F(m)=\tfrac12$, so $m=\log2/\lambda\approx10.4$ seconds: half of all waits are
shorter than about 10 seconds even though the mean is 15. The density is
largest at $t=0$ and has a long right tail, which pulls the mean above the
median.

**Memorylessness.** Having already waited $s$ minutes without a request, the
chance of waiting $t$ minutes more is the same as at the start:

$$
P(T>s+t \mid T>s)
= \frac{P(T>s+t)}{P(T>s)}
= \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}
= e^{-\lambda t}
= P(T>t).
$$

Time already spent waiting tells us nothing about the time still to wait. Among
continuous distributions on $(0,\infty)$, only the exponential has this
property; exercise :exref:`ex_mdl-distributions-memoryless-converse` proves
it. The property also marks the limit of the model. A light bulb whose
filament thins with use is more likely to fail in the next hour the older it
is, so its lifetime is not exponential.

**The Gamma distribution.** The time $T_k$ of the $k$-th request is the sum of
$k$ independent waits. By the binomial's argument, linearity and independence,
its mean is $k/\lambda$ and its variance $k/\lambda^2$. Its density also follows
from the process. The $k$-th event comes after $t$ exactly when at most $k-1$
events fall in $[0,t]$:

$$
P(T_k>t) = P\bigl(N(t)\le k-1\bigr) = \sum_{j=0}^{k-1} e^{-\lambda t}\frac{(\lambda t)^j}{j!}.
$$

Differentiating the $j$-th term gives
$-\lambda e^{-\lambda t}\frac{(\lambda t)^j}{j!}+\lambda e^{-\lambda t}\frac{(\lambda t)^{j-1}}{(j-1)!}$,
whose second part cancels the first part of the term before it. The sum
telescopes, and only the first part of the last term survives:

$$
p(t) = -\frac{d}{dt}P(T_k>t) = \frac{\lambda^k\,t^{k-1}e^{-\lambda t}}{(k-1)!},
\qquad t>0 .
$$

For $k=1$ this is the exponential density. To allow values of $k$ that are not
integers, replace $(k-1)!$ by the *Gamma function*
$\Gamma(z)=\int_0^\infty u^{z-1}e^{-u}\,du$ for $z>0$. Integration by parts
gives $\Gamma(z+1)=z\,\Gamma(z)$, and $\Gamma(1)=1$, so $\Gamma(n+1)=n!$. The
*Gamma distribution* with shape $\alpha>0$ and rate $\beta>0$ has the density

$$
p(t) = \frac{\beta^{\alpha}\,t^{\alpha-1}e^{-\beta t}}{\Gamma(\alpha)},
\qquad t>0,
\qquad T\sim\mathrm{Gamma}(\alpha,\beta).
$$
:eqlabel:`eq_mdl-gamma_pdf`

The substitution $u=\beta t$ turns its integral into $\Gamma(\alpha)$, so it
integrates to one. The same substitution gives
$\mathbb{E}[T]=\Gamma(\alpha+1)/(\beta\,\Gamma(\alpha))=\alpha/\beta$ and
$\textrm{Var}(T)=\alpha/\beta^2$, which for $\alpha=k$, $\beta=\lambda$ are the
moments of the $k$-th waiting time. For the server, the 4th request arrives
after a mean of $1$ minute with standard deviation $0.5$ minutes, and
$P(T_4>1)=P(N(1)\le3)\approx0.43$: in 43% of minutes, fewer than four requests
arrive, although four is the average.

**Sampling.** The inverse-transform method of
:numref:`sec_mdl-random_variables` turns a uniform draw $U$ into a draw from any
continuous distribution with an invertible cumulative distribution function
$F$, by setting $X=F^{-1}(U)$. For the exponential,
$F^{-1}(q)=-\log(1-q)/\lambda$, so $T=-\log(1-U)/\lambda$. The variable $1-U$
has the same distribution as $U$. We use it because generators return values
in $[0,1)$, so $U=0$ can occur, and $-\log U$ would then be infinite.

Does one simulated stream of events give both laws at once? The cell draws
$60{,}000$ exponential gaps at rate $4$, adds them up into event times, counts
the events in each of the first $10{,}000$ minutes, and checks the Poisson
count, the exponential wait and the Gamma probability $P(T_4>1)$.

```{.python .input #distributions-exponential}
lam = 4.0                                       # requests per minute
gaps = -np.log(1 - rng.random(60000)) / lam     # inverse transform: Exp(lam)
times = np.cumsum(gaps)                         # event times, about 15,000 min
counts = np.bincount(times[times < 10000].astype(int), minlength=10000)
print(f'requests per minute: mean {counts.mean():.2f}, variance {counts.var(ddof=1):.2f}')
print(f'silent minutes {np.mean(counts == 0):.4f},   waits over 1 min '
      f'{np.mean(gaps > 1):.4f},   e^-4 = {math.exp(-4):.4f}')
print(f'mean wait {gaps.mean():.3f} min,   minutes with at most 3 requests '
      f'{np.mean(counts <= 3):.3f}')
```

The counts have mean and variance close to $4$, and the mean wait is close to
$0.25$ minutes. The fraction of silent minutes and the fraction of waits longer
than a minute are both close to $e^{-4}$. The fraction of minutes with at most
three requests is close to the Gamma value $P(T_4>1)\approx0.43$. Gaps drawn
as exponentials produce Poisson counts.

### Uniform

Every sampler in this section starts from the same raw material, a number drawn
uniformly from $[0,1)$. Its law, and the law of a number drawn uniformly from
any other interval, is the continuous uniform.

The *uniform* distribution $U(a,b)$ on an interval $[a,b]$ with $a<b$ has
constant density, fixed by $\int p=1$:

$$
p(x) = \begin{cases}\frac{1}{b-a} & x\in[a,b],\\ 0 & \text{otherwise.}\end{cases}
$$
:eqlabel:`eq_mdl-cont_uniform_pdf`

Every subinterval of the same length has the same probability, its length
divided by $b-a$. (Every single value has probability zero, as for any
density.) The moments follow by direct integration; for the variance, center
the interval at its midpoint, so that it runs from $-c$ to $c$ with
$c=\tfrac{b-a}2$:

$$
\mathbb{E}[X] = \int_a^b \frac{x}{b-a}\,dx = \frac{b^2-a^2}{2(b-a)} = \frac{a+b}{2},
\qquad
\textrm{Var}(X) = \int_{-c}^{c} \frac{y^2}{2c}\,dy = \frac{c^2}{3} = \frac{(b-a)^2}{12}.
$$

$U(0,1)$ has mean $\tfrac12$ and variance $\tfrac1{12}$, and
$U(-\sqrt3,\sqrt3)$ has variance $(2\sqrt3)^2/12=1$. The formula continues the
discrete uniform: $n$ equally likely points spaced $h$ apart have variance
$h^2(n^2-1)/12$, which tends to $(b-a)^2/12$ as the points fill $[a,b]$ with
$nh=b-a$. Through the inverse transform, $U(0,1)$ becomes any continuous
distribution; the exponential sampler above is the example. The uniform has
hard edges and no tails, so it looks nothing like a Gaussian. For that reason
we use it below as the summand that the central limit theorem turns into one.

Is every subinterval of length $0.3$ equally likely under $U(-\sqrt3,\sqrt3)$,
and is the variance $1$?

```{.python .input #distributions-continuous-uniform}
a, b = -math.sqrt(3), math.sqrt(3)
u = a + (b - a) * rng.random(100000)            # shift and scale U(0, 1)
for lo in (-1.5, 1.0):
    print(f'fraction in [{lo}, {lo + 0.3:.1f}): {np.mean((u >= lo) & (u < lo + 0.3)):.3f}')
print(f'0.3 / (b-a) = {0.3 / (b - a):.3f};  sample variance {u.var(ddof=1):.3f},'
      f'  (b-a)^2/12 = {(b - a)**2 / 12:.3f}')
```

Both subintervals catch about $0.087$ of the draws, which is $0.3/(2\sqrt3)$,
and the sample variance is close to $1$.

### Gaussian
:label:`subsec_mdl-gaussian-clt`

We found $P(X=9)$ for the coin by counting 715 orderings. A test set of
$10{,}000$ examples also has a binomial number of errors, and counting
orderings there is hopeless. Is there a formula for the shape of the binomial
near its peak?

**The log-ratio near the mean.** Write $q=1-p$ and measure the
count from the mean, $k=np+x$. The successive ratio of the binomial becomes

$$
\frac{P(X=k+1)}{P(X=k)} = \frac{(n-k)\,p}{(k+1)\,q}
= \frac{(nq-x)\,p}{(np+x+1)\,q}
= \frac{1-x/(nq)}{1+(x+1)/(np)} .
$$

For $x$ of the order of the standard deviation $\sqrt{npq}$, both $x/(nq)$ and
$(x+1)/(np)$ are small, and $\log(1+\epsilon)\approx\epsilon$ gives

$$
\log\frac{P(X=k+1)}{P(X=k)} \approx -\frac{x}{nq}-\frac{x+1}{np}
= -\frac{x}{npq}-\frac{1}{np},
$$

where we used $\frac1{nq}+\frac1{np}=\frac{p+q}{npq}=\frac{1}{npq}$. The step
from $\log P(X=k)$ to $\log P(X=k+1)$ is, to leading order, linear in the
distance $x$ from the mean, with slope $-1/(npq)$; the term $-1/(np)$ is
smaller by a factor of order $1/\sqrt n$. A function whose steps grow
linearly is a parabola. Summing the steps from the mean to $np+x$,

$$
\log P(X=np+x) \approx C-\sum_{j=0}^{x-1}\frac{j}{npq} = C-\frac{x(x-1)}{2npq}
\approx C-\frac{x^2}{2npq},
$$

with $C$ the logarithm of the peak value, which the normalizer below fixes.
Near its peak the binomial is a bell, $e^{-x^2/(2npq)}$, centered at the mean
and as wide as its standard deviation $\sqrt{npq}$. This is the
*de Moivre–Laplace theorem*, the first central limit theorem. With the
normalizer it reads $P(X=np+x)\approx e^{-x^2/(2npq)}/\sqrt{2\pi npq}$. For the
coin, $npq=3.25$ and $x=2.5$ give $e^{-2.5^2/6.5}/\sqrt{2\pi\cdot3.25}\approx0.0846$,
against the exact $0.0873$.

How good is the bell at $n=13$, and how fast does it improve? The cell compares
it with the exact pmf for fair coins.

```{.python .input #mdl-distributions-gaussian-2}
for n in (13, 100, 1000):
    k = np.arange(n + 1)
    bell = np.exp(-(k - n / 2)**2 / (n / 2)) / np.sqrt(np.pi * n / 2)
    exact = binomial_pmf(n, 0.5, n)
    peak = n // 2
    print(f'n = {n:4d}: relative error at the peak {abs(bell[peak] / exact[peak] - 1):.2%}')
print(f'n = 13, k = 9: bell {np.exp(-2.5**2 / 6.5) / np.sqrt(np.pi * 6.5):.4f},'
      f' exact {binomial_pmf(13, 0.5, 13)[9]:.4f}')
```

The bell is 3% low at $k=9$ for 13 flips, and its relative error at the peak is
$1.7\%$ at $n=13$, $0.25\%$ at $n=100$ and $0.03\%$ at $n=1000$, falling like
$1/n$.

**The standard Gaussian.** The parabola suggests rescaling. The standardized
count $Y^{(n)}=(X-np)/\sqrt{npq}$ has mean $0$ and variance $1$ for every $n$,
and its values are spaced $1/\sqrt{npq}$ apart. The bell says that
$P(X=k)\approx\phi(y)/\sqrt{npq}$ at $y=(k-np)/\sqrt{npq}$, a density
$\phi(y)$ times the spacing, with

$$
\phi(y) = \frac{1}{\sqrt{2\pi}}\,e^{-y^2/2}.
$$

This is the density of the *standard Gaussian* $\mathcal N(0,1)$, the limit of
the standardized binomial. Shifting by $\mu$ and scaling by $\sigma>0$ gives
the variable $X=\mu+\sigma Y$, whose density by the change-of-variables formula
of :numref:`sec_mdl-random_variables` is $\phi((x-\mu)/\sigma)/\sigma$, the
*Gaussian* (or normal) distribution

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr),
\qquad X\sim\mathcal N(\mu,\sigma^2).
$$
:eqlabel:`eq_mdl-gaussian_pdf`

The parameter $\mu$ moves the center of the bell and $\sigma$ sets its width.
The variance computation below shows that they are the mean and the standard
deviation.

**The central limit theorem.** The derivation used little about coin flips
beyond their mean and variance, and the general theorem needs only these two
moments. Let $X_1,X_2,\dots$ be
independent and identically distributed with mean $\mu$ and variance
$0<\sigma^2<\infty$, and let $S_n=\sum_{i=1}^nX_i$. Then for every $y$,

$$
\lim_{n\to\infty} P\Bigl(\frac{S_n-n\mu}{\sigma\sqrt n}\le y\Bigr) = \Phi(y)
= \int_{-\infty}^{y}\phi(u)\,du .
$$
:eqlabel:`eq_mdl-clt`

The standardized sum of many independent copies of any law with finite
variance has, in the limit, the standard Gaussian distribution. For Bernoulli
summands this is the de Moivre–Laplace theorem above; we take the general proof
as given :cite:`Wasserman.2013`. Only the mean and the variance of the summands
enter the limit, which is why sums of many small independent effects, such as
measurement noise, so often look Gaussian.

**Sums of uniforms.** The theorem is easy to watch. The cell starts from
$U(0,1)$, flat with hard edges and no tails, forms sums of $n$ draws,
standardizes them by their mean $n/2$ and standard deviation $\sqrt{n/12}$,
and compares their histograms with $\phi$.

```{.python .input #mdl-distributions-gaussian-1}
ns, x = (1, 2, 8, 32), np.arange(-4, 4, 0.01)
curves = []
for n in ns:                                   # standardized sums of n uniforms
    s = (rng.random((100000, n)).sum(1) - n / 2) / np.sqrt(n / 12)
    hist, edges = np.histogram(s, bins=80, range=(-4, 4), density=True)
    curves.append(np.interp(x, (edges[:-1] + edges[1:]) / 2, hist))
curves.append(np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))     # N(0, 1) density
d2l.plot(x, curves, xlabel='standardized sum', ylabel='density',
         legend=[f'n = {n}' for n in ns] + ['N(0, 1)'])
```

At $n=1$ we see the uniform's plateau and at $n=2$ a triangle. By $n=8$ the
histogram is close to the Gaussian, and at $n=32$ the two cannot be told apart
at the resolution of the plot.

**The normalizer.** Why the constant $1/\sqrt{2\pi}$? Let
$I=\int_{-\infty}^\infty e^{-x^2/2}\,dx$. Squaring $I$ gives an integral over
the plane, and polar coordinates $x=r\cos\theta$, $y=r\sin\theta$, with area
element $dx\,dy=r\,dr\,d\theta$, make it elementary:

$$
I^2 = \int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\,dx\,dy
= \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta
= 2\pi\,\Bigl[-e^{-r^2/2}\Bigr]_0^\infty = 2\pi .
$$

So $I=\sqrt{2\pi}$, and $\phi$ integrates to one. The factor $r$ of the polar
area element gives $r\,e^{-r^2/2}$ the simple antiderivative $-e^{-r^2/2}$,
which $e^{-x^2/2}$ alone does not have.

**The variance.** The mean of $\mathcal N(0,1)$ is $0$ by symmetry. For the
variance, integrate by parts with $u=x$ and $dv=x\,e^{-x^2/2}\,dx$:

$$
\int_{-\infty}^{\infty} x^2\,\frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx
= \frac{1}{\sqrt{2\pi}}\Bigl(\bigl[-x\,e^{-x^2/2}\bigr]_{-\infty}^{\infty}
+ \int_{-\infty}^{\infty} e^{-x^2/2}\,dx\Bigr) = 0 + 1 = 1 .
$$

The standard Gaussian has variance $1$, so $\mathcal N(\mu,\sigma^2)$ has mean
$\mu$ and variance $\sigma^2$: the two parameters are the first two moments.

**The Cauchy counterexample.** The theorem needs a finite variance. The previous
section met the standard Cauchy density $1/(\pi(1+x^2))$, whose tails are so
heavy that its variance is infinite. The average of $n$ independent standard
Cauchy draws is again standard Cauchy, a fact we state without proof, so
averaging more draws never narrows the spread. In particular
$P(|\bar X_n|>1)=\tfrac12$ for every $n$. For Gaussian draws the average is
$\mathcal N(0,1/n)$, and the same probability falls from $0.32$ at $n=1$ to
$0.0016$ at $n=10$, and it is negligible at $n=1000$.

Does averaging help when the variance is infinite? The cell averages $n$
Cauchy draws and $n$ Gaussian draws, $10{,}000$ times each.

```{.python .input #distributions-gaussian}
for n in (1, 10, 1000):
    cauchy = rng.standard_cauchy((10000, n)).mean(axis=1)
    gauss = rng.standard_normal((10000, n)).mean(axis=1)
    print(f'n = {n:4d}: P(|average| > 1):  Cauchy {np.mean(np.abs(cauchy) > 1):.3f},'
          f'  Gaussian {np.mean(np.abs(gauss) > 1):.4f}')
```

The Cauchy fraction stays near $\tfrac12$ at every $n$. The Gaussian fraction
is about $0.32$ at $n=1$, one or two in a thousand at $n=10$, and zero at
$n=1000$.

**Gaussian noise and squared error.** A regression network that predicts $\hat y$ and
assumes the noise $y\sim\mathcal N(\hat y,\sigma^2)$ has the negative
log-probability $-\log p(y)=(y-\hat y)^2/(2\sigma^2)+\tfrac12\log(2\pi\sigma^2)$,
which is the squared error up to a scale and a constant. The Gaussian is also
the default prior and latent distribution of generative models.

### Laplace

Squared error punishes a residual of $10$ a hundred times more than a
residual of $1$, so a few outliers can dominate a fit. A loss that grows only
linearly, the absolute error, corresponds to a different noise law.

**Construction.** Let $S$ be a fair random sign, $P(S=1)=P(S=-1)=\tfrac12$, and
let $E$ be an exponential variable with mean $b>0$ (rate $1/b$), independent of
$S$. The variable $X=\mu+SE$ lies above $\mu$ exactly when $S=1$, and then
$X-\mu=E$, so for $x>\mu$ its density is $\tfrac12\cdot\tfrac1b e^{-(x-\mu)/b}$;
by symmetry the same holds below $\mu$ with $|x-\mu|$. This is the *Laplace*
distribution,

$$
p(x) = \frac{1}{2b}\,e^{-|x-\mu|/b},
\qquad X\sim\mathrm{Laplace}(\mu,b).
$$
:eqlabel:`eq_mdl-laplace_pdf`

Each side carries half the mass of an exponential. The density is symmetric,
has a sharp peak at $\mu$, and decays exponentially on both sides.

**Mean and variance.** The construction gives both. The mean is $\mu$ because
$\mathbb{E}[SE]=\mathbb{E}[S]\,\mathbb{E}[E]=0$. Because $S^2=1$, the variance
is the exponential's second moment, $2/\lambda^2$ with $\lambda=1/b$:

$$
\textrm{Var}(X) = \mathbb{E}\bigl[S^2E^2\bigr] = \mathbb{E}[E^2] = 2b^2 .
$$

**Laplace noise and absolute error.** A regression network that assumes Laplace noise has the
negative log-probability $-\log p(y)=|y-\hat y|/b+\log 2b$, the absolute error
plus a constant. As a prior on weights, the Laplace density gives the
$\ell_1$ penalty $\|\mathbf w\|_1$ used by the LASSO :cite:`Tibshirani.1996`.
For a location fitted to data $x_1,\dots,x_m$, the Laplace likelihood is
largest where $\sum_i|x_i-\mu|$ is smallest, which is at the sample median. A
single outlier can move the sample mean arbitrarily far, but it moves the
median by at most one data point.

**Tails.** :numref:`fig_mdl-prob-continuous-pdfs` compares the uniform, the
Gaussian and the Laplace at the same variance. At matched variance
$\sigma^2=2b^2$, the Laplace probability of landing more than three standard
deviations from the center is $P(|X-\mu|>3\sigma)=e^{-3\sigma/b}=e^{-3\sqrt2}\approx0.014$,
against $0.0027$ for the Gaussian: about five times more.

![Three densities with variance one: the uniform on $[-\sqrt{3},\sqrt{3}]$, the standard Gaussian, and the Laplace with $b=1/\sqrt{2}$. (a) The densities. (b) The same on a logarithmic axis, where the Gaussian is a parabola, the Laplace two straight lines, and the uniform a plateau that ends at $\pm\sqrt{3}$. At three standard deviations the Laplace carries about five times the Gaussian's tail probability.](../img/mdl-prob-continuous-pdfs.svg)
:label:`fig_mdl-prob-continuous-pdfs`

How much heavier is the Laplace tail at matched variance? The cell samples the
Laplace by its construction, a random sign times an exponential.

```{.python .input #distributions-laplace}
b, m = 1.0, 200000
lap = rng.choice([-1.0, 1.0], m) * rng.exponential(b, m)   # S * E
sigma = math.sqrt(2) * b                                    # matched sd
gau = rng.normal(0.0, sigma, m)
print(f'sample variances: Laplace {lap.var(ddof=1):.2f}, '
      f'Gaussian {gau.var(ddof=1):.2f}   (2b^2 = 2)')
print(f'P(|X| > 3 sd): Laplace {np.mean(np.abs(lap) > 3 * sigma):.4f}, '
      f'Gaussian {np.mean(np.abs(gau) > 3 * sigma):.4f}')
```

Both sample variances are close to $2$, and the tail fractions are about
$0.014$ and $0.003$.

## The Multivariate Gaussian

The noise in an image, the latent code of a generative model, and the
predictions of a model at many inputs are vectors, and their coordinates are
correlated. The multivariate Gaussian is the distribution of such vectors that
we can compute with. Our running example has two coordinates with variance $2$
each and covariance $1$,

$$
\boldsymbol\Sigma = \begin{pmatrix}2&1\\1&2\end{pmatrix}.
$$

### Contours and Eigenvectors

The *multivariate Gaussian* $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ on
$\mathbb R^d$ has a mean vector $\boldsymbol\mu$ and a covariance matrix
$\boldsymbol\Sigma$ that is symmetric and positive definite. Its density is

$$
p(\mathbf x) = \frac{1}{(2\pi)^{d/2}|\boldsymbol\Sigma|^{1/2}}
\exp\!\Bigl(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Bigr),
$$
:eqlabel:`eq_mdl-mvn_pdf`

where $|\boldsymbol\Sigma|$ is the determinant. For $d=1$ it is
:eqref:`eq_mdl-gaussian_pdf`. The density depends on $\mathbf x$ only through
the quadratic form in the exponent, so it is constant on the sets where that
form is constant. What do these sets look like?

**The running example by hand.** Turn the axes by $45^\circ$ and use the sum
and difference coordinates $u=(x_1+x_2)/\sqrt2$ and $w=(x_1-x_2)/\sqrt2$, with
$\boldsymbol\mu=\mathbf 0$. Their variances and covariance follow from the
entries of $\boldsymbol\Sigma$:

$$
\textrm{Var}(u) = \frac{2+2+2\cdot1}{2} = 3,
\qquad
\textrm{Var}(w) = \frac{2+2-2\cdot1}{2} = 1,
\qquad
\textrm{Cov}(u,w) = \frac{2-2}{2} = 0 .
$$

In the turned coordinates the covariance is diagonal. The new axes are the
eigenvectors $\mathbf v_1=(1,1)/\sqrt2$ and $\mathbf v_2=(1,-1)/\sqrt2$ of
$\boldsymbol\Sigma$, with eigenvalues $\gamma_1=3$ and $\gamma_2=1$; indeed
$\boldsymbol\Sigma\mathbf v_1=(3,3)/\sqrt2=3\mathbf v_1$.

**The general case.** The spectral theorem says that a symmetric matrix has an
orthonormal basis of eigenvectors,
$\boldsymbol\Sigma=\mathbf V\operatorname{diag}(\gamma_1,\dots,\gamma_d)\mathbf V^\top$
with $\mathbf V^\top\mathbf V=\mathbf I$, and positive definiteness means that
every $\gamma_i>0$. In the coordinates
$\mathbf u=\mathbf V^\top(\mathbf x-\boldsymbol\mu)$ the quadratic form becomes
a weighted sum of squares,

$$
(\mathbf x-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)
= \mathbf u^\top\operatorname{diag}(\gamma_1^{-1},\dots,\gamma_d^{-1})\,\mathbf u
= \sum_{i=1}^d \frac{u_i^2}{\gamma_i}.
$$

The set where this sum equals $r^2$ is an ellipsoid with axes along the
eigenvectors $\mathbf v_i$ and half-lengths $r\sqrt{\gamma_i}$. The contours of
a Gaussian are therefore ellipsoids whose axes are the eigenvectors of the
covariance, stretched by the standard deviations $\sqrt{\gamma_i}$ along them.
For the running example they are ellipses at $45^\circ$ with half-lengths in
the ratio $\sqrt3:1$ (:numref:`fig_mdl-prob-mvn-contours`, left). Because
$|\boldsymbol\Sigma|=\prod_i\gamma_i$, the density also factorizes in
$\mathbf u$ into $d$ one-dimensional Gaussians: the coordinates along the
eigenvectors are independent with variances $\gamma_i$. When
$\boldsymbol\Sigma=\sigma^2\mathbf I$, the contours are spheres and the original
coordinates are already independent.

![The running Gaussian with covariance $\boldsymbol{\Sigma}$ with rows $(2, 1)$ and $(1, 2)$. (a) 400 samples and three contours of the density, on which the quadratic form in the exponent equals $1$, $4$ and $9$. The axes of the ellipses lie along the eigenvectors of $\boldsymbol{\Sigma}$, with half-lengths proportional to $\sqrt{\gamma_1} = \sqrt{3}$ and $\sqrt{\gamma_2} = 1$; the arrows $\mathbf{v}_1$ and $\mathbf{v}_2$ reach the middle contour. (b) Observing $x_2 = 2$ slices the density along the dashed line; the slice is the conditional density $\mathcal{N}(1, \frac{3}{2})$ of $x_1$. The conditional means of all slices lie on the line $x_1 = x_2/2$, and the conditional variance is $3/2$ for every slice.](../img/mdl-prob-mvn-contours.svg)
:label:`fig_mdl-prob-mvn-contours`

**High dimension.** For the standard Gaussian $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$
the coordinates are independent with variance $1$, so
$\mathbb{E}\|\mathbf x\|^2=d$, and the norm concentrates near $\sqrt d$. Two
independent draws have a cosine of mean $0$ and standard deviation exactly
$1/\sqrt d$, as :numref:`sec_mdl-geometry-linear-algebraic-ops` showed for a
uniformly random direction, and the direction $\mathbf x/\|\mathbf x\|$ of a
standard Gaussian is uniform on the sphere.

### Linear Maps and Sampling

In practice we build Gaussian vectors from other Gaussian vectors: we add
independent noise, mix coordinates, or project onto a few directions. All of
these are linear maps, and a linear map keeps a Gaussian Gaussian.

**Proposition.** Let $\mathbf x\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$
in $\mathbb R^d$, and let $\mathbf A$ be an $m\times d$ matrix of full row rank
$m\le d$ and $\mathbf b\in\mathbb R^m$. Then

$$
\mathbf A\mathbf x+\mathbf b\sim\mathcal N\bigl(\mathbf A\boldsymbol\mu+\mathbf b,\;\mathbf A\boldsymbol\Sigma\mathbf A^\top\bigr).
$$

**Proof.** The mean and covariance hold for any distribution with these
moments, by linearity:
$\mathbb{E}[(\mathbf A\mathbf x-\mathbf A\boldsymbol\mu)(\mathbf A\mathbf x-\mathbf A\boldsymbol\mu)^\top]
=\mathbf A\,\mathbb{E}[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]\,\mathbf A^\top
=\mathbf A\boldsymbol\Sigma\mathbf A^\top$. It remains to show that the law is
Gaussian. For a square invertible $\mathbf A$, the change-of-variables formula
of :numref:`sec_mdl-random_variables` gives the density of
$\mathbf y=\mathbf A\mathbf x+\mathbf b$ as
$p_{\mathbf y}(\mathbf y)=p_{\mathbf x}(\mathbf A^{-1}(\mathbf y-\mathbf b))/|\det\mathbf A|$.
With $\mathbf c=\mathbf y-\mathbf A\boldsymbol\mu-\mathbf b$, the exponent
becomes
$-\tfrac12\mathbf c^\top\mathbf A^{-\top}\boldsymbol\Sigma^{-1}\mathbf A^{-1}\mathbf c
=-\tfrac12\mathbf c^\top(\mathbf A\boldsymbol\Sigma\mathbf A^\top)^{-1}\mathbf c$,
and $|\det\mathbf A|\,|\boldsymbol\Sigma|^{1/2}=|\mathbf A\boldsymbol\Sigma\mathbf A^\top|^{1/2}$,
so $p_{\mathbf y}$ is the Gaussian density with the stated parameters. A wide
$\mathbf A$ with independent rows can be completed by $d-m$ further rows to an
invertible square matrix. Then $\mathbf A\mathbf x+\mathbf b$ is the first
block of a Gaussian vector, and the conditioning proposition below shows
(after reordering the coordinates) that every block of a Gaussian vector is
Gaussian. $\blacksquare$

The rank condition matters. With $\mathbf A=\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$
the two outputs are always equal, and
$\mathbf A\boldsymbol\Sigma\mathbf A^\top=\left(\begin{smallmatrix}6&6\\6&6\end{smallmatrix}\right)$
is singular. The image is a Gaussian spread along the line $y_1=y_2$, and it
has no density in the plane.

**Sampling by a matrix factor.** How do we draw from
$\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ when the generator produces only
independent standard normals? Stack $d$ of them into
$\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$, whose density is the product
of $d$ copies of $\phi$. If a matrix $\mathbf L$ satisfies
$\mathbf L\mathbf L^\top=\boldsymbol\Sigma$, then by the proposition
$\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$ is Gaussian with mean
$\boldsymbol\mu$ and covariance

$$
\mathbb{E}\bigl[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top\bigr]
= \mathbf L\,\mathbb{E}[\mathbf z\mathbf z^\top]\,\mathbf L^\top
= \mathbf L\,\mathbf I\,\mathbf L^\top
= \boldsymbol\Sigma .
$$

A matrix factor turns independent noise into correlated noise. The cheapest
factor is the *Cholesky factor*: every symmetric positive definite matrix
factors as $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ with $\mathbf L$ lower
triangular with a positive diagonal. For the running example, matching entries
of $\mathbf L\mathbf L^\top$ to $\boldsymbol\Sigma$ one at a time gives
$L_{11}=\sqrt2$, then $L_{21}=\Sigma_{21}/L_{11}=1/\sqrt2$, then
$L_{22}=\sqrt{\Sigma_{22}-L_{21}^2}=\sqrt{3/2}$, and indeed
$L_{21}^2+L_{22}^2=\tfrac12+\tfrac32=2$.

Any other factor works as well, for
instance the eigen-factor $\mathbf V\operatorname{diag}(\sqrt{\gamma_1},\dots,\sqrt{\gamma_d})$;
two factors of the same $\boldsymbol\Sigma$ differ by an orthogonal matrix,
which exercise :exref:`ex_mdl-distributions-two-factors` shows. PyTorch's
`MultivariateNormal` samples with the Cholesky factor, and NumPy's
`multivariate_normal` uses a factor from the singular value decomposition by
default.

Does the Cholesky recipe reproduce $\boldsymbol\Sigma$? The cell samples by the
recipe itself, so that the mechanism is visible, and checks the covariance of
the samples and its eigenvalues.

```{.python .input #distributions-mvn}
Sigma = np.array([[2.0, 1.0], [1.0, 2.0]])
L = np.linalg.cholesky(Sigma)                   # Sigma = L L^T
x = rng.standard_normal((400000, 2)) @ L.T      # each row is x = L z
C = np.cov(x.T)
print('L =', L.round(3).tolist())
print('sample covariance =', C.round(2).tolist())
print('its eigenvalues   =', np.linalg.eigvalsh(C).round(2))
```

The factor is $\left(\begin{smallmatrix}1.414&0\\0.707&1.225\end{smallmatrix}\right)$,
the sample covariance agrees with $\boldsymbol\Sigma$ to about $0.01$, and its
eigenvalues are close to $1$ and $3$.

### Conditioning

Often we observe some coordinates of a Gaussian vector and want to predict the
others. In the running example, if we observe $x_2=2$, where is $x_1$ likely
to be? Split the vector into blocks $\mathbf x=(\mathbf x_1,\mathbf x_2)$, and
split its parameters to match:

$$
\boldsymbol\mu = \begin{pmatrix}\boldsymbol\mu_1\\\boldsymbol\mu_2\end{pmatrix},
\qquad
\boldsymbol\Sigma = \begin{pmatrix}\boldsymbol\Sigma_{11} & \boldsymbol\Sigma_{12}\\
\boldsymbol\Sigma_{21} & \boldsymbol\Sigma_{22}\end{pmatrix}.
$$

The idea is to remove from $\mathbf x_1$ the part that $\mathbf x_2$ predicts
linearly. What is left over carries no information about $\mathbf x_2$.

**Proposition.** Let $\mathbf B=\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}$.
The residual $\mathbf r=\mathbf x_1-\mathbf B\mathbf x_2$ is independent of
$\mathbf x_2$, and $\mathbf x_2\sim\mathcal N(\boldsymbol\mu_2,\boldsymbol\Sigma_{22})$.

**Proof.** The block $\boldsymbol\Sigma_{22}$ is positive definite, as a
diagonal block of a positive definite matrix, so it is invertible. The
residual is uncorrelated with $\mathbf x_2$:
$\textrm{Cov}(\mathbf r,\mathbf x_2)=\boldsymbol\Sigma_{12}-\mathbf B\boldsymbol\Sigma_{22}=\mathbf 0$.
The map $\mathbf x\mapsto(\mathbf r,\mathbf x_2)$ is the square matrix
$\left(\begin{smallmatrix}\mathbf I&-\mathbf B\\\mathbf 0&\mathbf I\end{smallmatrix}\right)$,
which has determinant $1$, so $(\mathbf r,\mathbf x_2)$ is Gaussian by the
square case of the previous proposition. Its covariance is block diagonal,
because the off-diagonal blocks are $\textrm{Cov}(\mathbf r,\mathbf x_2)=\mathbf 0$.
Then the inverse covariance is block diagonal too, the quadratic form in the
exponent splits into a term in $\mathbf r$ plus a term in $\mathbf x_2$, and the
determinant is the product of the two blocks' determinants. The density is
therefore a Gaussian density in $\mathbf r$ times a Gaussian density in
$\mathbf x_2$: the two are independent, and $\mathbf x_2$ has mean
$\boldsymbol\mu_2$ and covariance $\boldsymbol\Sigma_{22}$. $\blacksquare$

The proof used only square invertible maps. Its second claim, that every block
of a Gaussian vector is Gaussian, therefore completes the proof of the
linear-map proposition for wide matrices without circularity. Now condition on $\mathbf x_2$. Since
$\mathbf x_1=\mathbf r+\mathbf B\mathbf x_2$ and $\mathbf r$ is independent of
$\mathbf x_2$, given $\mathbf x_2$ the block $\mathbf x_1$ is the Gaussian
$\mathbf r$ shifted by $\mathbf B\mathbf x_2$. The residual has mean
$\boldsymbol\mu_1-\mathbf B\boldsymbol\mu_2$ and covariance
$\boldsymbol\Sigma_{11}-\mathbf B\boldsymbol\Sigma_{21}-\boldsymbol\Sigma_{12}\mathbf B^\top+\mathbf B\boldsymbol\Sigma_{22}\mathbf B^\top
=\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}$,
so $\mathbf x_1\mid\mathbf x_2$ is Gaussian with

$$
\boldsymbol\mu_{1\mid 2} = \boldsymbol\mu_1 + \boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}(\mathbf x_2-\boldsymbol\mu_2),
\qquad
\boldsymbol\Sigma_{1\mid 2} = \boldsymbol\Sigma_{11} - \boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.
$$
:eqlabel:`eq_mdl-mvn_conditional`

The conditional mean is a linear function of the observed block, and the
conditional covariance does not depend on the observed value. For the running
example, $\mathbf B=\tfrac12$, so given $x_2=2$ the conditional mean is
$\tfrac12\cdot2=1$ and the conditional variance is $2-1\cdot\tfrac12\cdot1=\tfrac32$:
$x_1\mid x_2=2\sim\mathcal N(1,\tfrac32)$, the slice drawn in
:numref:`fig_mdl-prob-mvn-contours` (right).

The matrix $\mathbf B$ is the regression coefficient of $\mathbf x_1$ on
$\mathbf x_2$: the multiple of $\mathbf x_2$ whose removal leaves no
correlation. The conditional mean is linear regression read off the
covariance. The conditional covariance, called the *Schur complement* of
$\boldsymbol\Sigma_{22}$, is never larger than $\boldsymbol\Sigma_{11}$,
because the subtracted term
$\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}$ is
positive semidefinite: observing $x_2$ shrank the variance of $x_1$ from $2$ to
$\tfrac32$. With $\mathbf x_2$ the observed values of a function and
$\mathbf x_1$ its values at new inputs, this computation is Gaussian process
regression.

A reader may object that uncorrelated variables need not be independent. That
is right, and the proof used more than zero correlation: it used that
$(\mathbf r,\mathbf x_2)$ is *jointly* Gaussian. Without that, the conclusion
fails. If $Z\sim\mathcal N(0,1)$ and $S$ is an independent fair sign, then $SZ$
is also $\mathcal N(0,1)$, and $\textrm{Cov}(SZ,Z)=\mathbb{E}[S]\,\mathbb{E}[Z^2]=0$,
yet $|SZ|=|Z|$, so the two are far from independent.

Does the residual argument hold on samples? The cell reuses the Cholesky
samples, forms $r=x_1-\tfrac12x_2$, and then keeps only the samples whose $x_2$
is within $0.05$ of $2$.

```{.python .input #mdl-distributions-conditioning}
r = x[:, 0] - 0.5 * x[:, 1]                     # residual x1 - B x2, B = 1/2
print(f'Cov(r, x2) = {np.cov(r, x[:, 1])[0, 1]:.3f},  Var(r) = {r.var(ddof=1):.3f}')
near = x[np.abs(x[:, 1] - 2) < 0.05, 0]         # x1 of samples with x2 near 2
print(f'{near.size} samples with x2 near 2: mean of x1 {near.mean():.2f}, '
      f'variance {near.var(ddof=1):.2f}')
```

The residual's covariance with $x_2$ is close to $0$ and its variance close to
$\tfrac32$. About four thousand samples fall in the slice, and their $x_1$ has
mean near $1$ and variance near $\tfrac32$, the conditional law
$\mathcal N(1,\tfrac32)$.

We can now write down a likelihood for binary, categorical, count,
waiting-time and real-valued data, scalar or vector, and derive each mean and
variance from the way its law is built. Two questions are open. Nearly all of
these laws share one algebraic form. That form explains why their negative
log-likelihoods are convex in a suitable parameter and why their gradients
take the form "prediction minus observation", as
:numref:`sec_mdl-exponential-family` shows. And the probability $715/8192$
assumed that the coin is fair; how well 13 flips determine the bias of a coin
is the question of :numref:`sec_mdl-statistics`.

## Summary

The table collects the laws of this section, in the order of the text. For the
categorical and multinomial rows, a draw is coded as a one-hot vector, as the
cross-entropy loss codes it.

| Law | pmf or pdf | Mean | Variance | Model assumption |
|:--|:--|:--|:--|:--|
| $\mathrm{Bernoulli}(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | one yes/no outcome |
| $\mathrm{Binomial}(n,p)$ | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $n$ fixed, independent trials, one $p$ |
| $\mathrm{Poisson}(\lambda)$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | many rare independent chances; mean equals variance |
| $\mathrm{Categorical}(\mathbf p)$ | $p_k$ | $\mathbf p$ | $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ | one of $K$ outcomes |
| $\mathrm{Multinomial}(n,\mathbf p)$ | :eqref:`eq_mdl-multinomial_pmf` | $n\mathbf p$ | $n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$ | $n$ fixed, independent draws |
| $\mathrm{Unif}\{1,\dots,n\}$ | $1/n$ | $(n+1)/2$ | $(n^2-1)/12$ | equally likely values |
| $\mathrm{Exp}(\lambda)$ | $\lambda e^{-\lambda t}$, $t\ge0$ | $1/\lambda$ | $1/\lambda^2$ | constant event rate (no ageing) |
| $\mathrm{Gamma}(\alpha,\beta)$ | $\beta^\alpha t^{\alpha-1}e^{-\beta t}/\Gamma(\alpha)$ | $\alpha/\beta$ | $\alpha/\beta^2$ | time of the $\alpha$-th event (integer $\alpha$) |
| $U(a,b)$ | $1/(b-a)$ on $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ | equal-length intervals equally likely |
| $\mathcal N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | sum of many small independent effects, finite variance |
| $\mathrm{Laplace}(\mu,b)$ | $\frac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | symmetric, exponential tails |
| $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ | :eqref:`eq_mdl-mvn_pdf` | $\boldsymbol\mu$ | $\boldsymbol\Sigma$ | jointly Gaussian coordinates |

A small set of distributions describes the labels and noise of most models,
and each fixes a loss: the Bernoulli gives binary cross-entropy, the
categorical softmax cross-entropy, the Gaussian squared error, and the Laplace
absolute error. All of them grow out of one coin flip by a sum, a limit, a
waiting time, a random sign, or a vector of coordinates. The construction of
each law gives its mean and variance in a few lines, and the same constructions
give samplers: inverse transforms of a uniform, a random sign times an
exponential, a matrix factor times standard normals. The central limit theorem
explains why the Gaussian appears whenever many small effects add up, and the
Cauchy shows that it needs a finite variance. A Gaussian vector stays Gaussian
under linear maps and conditioning; its contours are the eigen-ellipsoids of
the covariance, and conditioning on one block is linear regression on it.

## Exercises

1. **Error counts on a test set.** A classifier is scored on $n=10{,}000$
   independent test examples, each of which it gets wrong with the same
   probability $q$, its true error rate. Its number of errors $X$ then has the
   distribution $\mathrm{Binomial}(n,q)$.
    1. Let $q=0.1$. Use the Gaussian approximation of the binomial (the
       de Moivre--Laplace theorem) to find the interval around $nq$ that
       contains $X$ with probability about $95\%$. Two classifiers with this
       error rate are scored on two independent test sets of this size. Find
       the standard deviation of the difference of their observed accuracies.
    1. Let $q=5\cdot10^{-4}$. Compute $P(X=0)$ and $P(X\ge10)$ exactly and
       under the Poisson approximation $\mathrm{Poisson}(nq)$. Show that the
       logarithm of $P(X=0)$ under the binomial differs from its Poisson
       value by about $-nq^2/2$.
    1. The classifier makes no errors on the $n$ test examples. Under the
       Poisson approximation, find the largest error rate $q$ for which this
       outcome still has probability at least $0.05$, and express it as a
       multiple of $1/n$. Use (b) to explain why the Poisson approximation
       is accurate for every error rate this small.
   :exlabel:`ex_mdl-distributions-error-counts`

1. **Conditioning on a noisy observation.** The conditional formulas
   :eqref:`eq_mdl-mvn_conditional` turn a joint Gaussian into a prediction of
   one block from another.
    1. Take the bivariate Gaussian with $\boldsymbol\mu=\mathbf 0$ and
       $\boldsymbol\Sigma=\left(\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right)$,
       where $|\rho|<1$. Identify $\mathbf B$ and the residual $r$, and check
       $\textrm{Cov}(r,x_2)=0$ directly. Show that
       $x_1\mid x_2\sim\mathcal N(\rho x_2,1-\rho^2)$. Why does observing $x_2$
       never increase the uncertainty about $x_1$?
    1. A signal $s\sim\mathcal N(0,\tau^2)$ is observed with additive noise,
       $y=s+\epsilon$, where $\epsilon\sim\mathcal N(0,\sigma^2)$ is
       independent of $s$. Write $(s,y)$ as a linear map of $(s,\epsilon)$ to
       obtain its joint Gaussian distribution, and compute the conditional
       distribution of $s$ given $y$. Show that the conditional precision
       $1/\textrm{Var}(s\mid y)$ is the sum of the prior precision
       $1/\tau^2$ and the noise precision $1/\sigma^2$. How does
       $\mathbb{E}[s\mid y]$ behave as $\sigma^2\to0$ and as
       $\sigma^2\to\infty$? This computation is Gaussian process
       regression (:numref:`chap_gp`) with a single observation.
   :exlabel:`ex_mdl-distributions-using-conditional-formula-eqref`

1. **Gamma--Poisson mixture.** Let the Poisson rate itself be random,
   $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$ with the density
   :eqref:`eq_mdl-gamma_pdf`, and let $X\mid\lambda\sim\mathrm{Poisson}(\lambda)$.
    1. Use the law of total expectation
       $\mathbb{E}[X]=\mathbb{E}\bigl[\mathbb{E}[X\mid\lambda]\bigr]$ and the law
       of total variance
       $\textrm{Var}(X)=\mathbb{E}\bigl[\textrm{Var}(X\mid\lambda)\bigr]+\textrm{Var}\bigl(\mathbb{E}[X\mid\lambda]\bigr)$
       to compute $\mathbb{E}[X]$ and $\textrm{Var}(X)$. Write the variance in
       terms of the mean $m=\alpha/\beta$ and the shape $\alpha$. In which
       limit of $\alpha$ and $\beta$ does the Poisson reappear?
    1. Show that $X$ has the *negative binomial* pmf

       $$P(X=k)=\frac{\Gamma(k+\alpha)}{k!\,\Gamma(\alpha)}\Bigl(\frac{\beta}{1+\beta}\Bigr)^{\alpha}\Bigl(\frac{1}{1+\beta}\Bigr)^{k},\qquad k=0,1,2,\dots,$$

       by integrating out $\lambda$ with the help of the Gamma normalizer.
    1. A count dataset has sample mean $2$ and sample variance $6$. Choose
       $\alpha$ and $\beta$ so that the model matches both. Which named
       distribution results?
   :exlabel:`ex_mdl-distributions-gamma-poisson-mixture`

1. **Converse of memorylessness.** ● The exponential distribution is
   memoryless. This problem proves the converse.
    1. Let $X>0$ be a continuous random variable whose survival function
       $G(t)=P(X>t)$ satisfies $G(s+t)=G(s)\,G(t)$ for all $s,t\ge0$. Show
       that $G(t)=e^{-\lambda t}$ for some $\lambda>0$. Start from $G(1)$,
       extend the formula to rational $t$, and then use that $G$ is
       nonincreasing. Why can $G(1)$ be neither $0$ nor $1$?
    1. Let $X$ take values in $\{1,2,\dots\}$ and satisfy
       $P(X>s+t\mid X>s)=P(X>t)$ for all integers $s,t\ge0$. Show that $X$ is
       geometric, $P(X>t)=(1-p)^t$ for some $p\in(0,1]$.
   :exlabel:`ex_mdl-distributions-memoryless-converse`

1. [code] **Convergence rate of the CLT.** ●● Let $F_n$ be the c.d.f. of
   the standardized sum $(S_n-n\mu)/(\sigma\sqrt n)$ of $n$ independent
   summands with mean $\mu$ and standard deviation $\sigma$. Measure the
   error of the Gaussian approximation by the Kolmogorov distance
   $D_n=\sup_x|F_n(x)-\Phi(x)|$, where $\Phi$ and $\phi$ denote the standard
   Gaussian c.d.f. and density. :numref:`subsec_mdl-gaussian-clt` judges the
   approximation by eye; this problem measures its rate.
    1. Compute $D_n$ for $n\in\{1,2,4,\dots,64\}$ and three summand
       distributions, $U(0,1)$, $\mathrm{Exp}(1)$ and
       $\mathrm{Bernoulli}(1/2)$. For uniform summands, compute the density of
       $S_n$ by repeated numerical convolution of the $U(0,1)$ density on a
       fine grid (`np.convolve`, rescaled by the grid spacing) and integrate it
       to $F_n$. For exponential summands, $S_n$ is the time of the $n$-th
       event of a Poisson process with rate $1$, so $P(S_n\le s)=P(N\ge n)$
       for $N\sim\mathrm{Poisson}(s)$. For the two continuous summands, take
       the supremum over a fine grid of $x\in[-4,4]$. For Bernoulli summands,
       use the exact binomial pmf and compare $\Phi$ with $F_n$ on both sides
       of every jump. Plot $D_n$ against $n$ on log-log axes and fit the
       slopes.
    1. For summands that are not confined to a lattice and have skewness
       $\gamma=\mathbb{E}[(X-\mu)^3]/\sigma^3$, the Edgeworth expansion
       :cite:`Hall.1992` refines the central limit theorem to

       $$F_n(x)=\Phi(x)-\frac{\gamma}{6\sqrt n}\,(x^2-1)\,\phi(x)+o\bigl(n^{-1/2}\bigr).$$

       Predict the limit of $\sqrt n\,D_n$ for exponential summands and
       compare it with your measurements. What does the expansion predict for
       uniform summands, and which rate did you measure for them?
    1. Bernoulli$(1/2)$ summands also have $\gamma=0$. Explain the slope you
       measured for them by the size of the largest jump of $F_n$, and predict
       the limit of $\sqrt n\,D_n$. Then compare $F_n$ at the lattice points
       $k$ with the continuity-corrected value
       $\Phi\bigl((k+\tfrac12-n\mu)/(\sigma\sqrt n)\bigr)$. Which rate does the
       corrected error have?
    1. The correctness indicators of a classifier with accuracy $0.99$ are
       $\mathrm{Bernoulli}(0.99)$ variables. Compute their skewness, and
       assume that the skewness term of (b) also governs the
       continuity-corrected error. Predict the number of test examples
       needed before the corrected error falls below $0.01$, and check the
       prediction by computing the corrected error for
       $n\in\{10^2,10^3,10^4\}$. Build the pmf of $S_n$ from the number of
       errors $n-S_n\sim\mathrm{Binomial}(n,0.01)$: the successive-ratio
       recursion started from $P(S_n=0)=0.01^n$ underflows to zero in
       floating point once $n$ exceeds about $160$, whereas the recursion for
       the errors starts from $0.99^n$. What does the result imply for
       Gaussian error bars on the test accuracy of a highly accurate model?
   :exlabel:`ex_mdl-distributions-clt-rate`

1. [extended] **Burstiness of word counts.** ●● :citet:`Church.Gale.1995`
   found that word counts in text vary across documents more than a Poisson
   model allows, and that Poisson mixtures such as the negative binomial fit
   them better. Reproduce the finding at small scale on
   *The Time Machine* (:numref:`sec_text-sequence`). Download the text with
   `d2l.download(d2l.DATA_URL + 'timemachine.txt')`, keep only letters as
   `d2l.TimeMachine._preprocess` does, split it into words, and cut the word
   sequence into consecutive chunks of $L=500$ words, which play the role of
   documents.
    1. For each of the words "the", "and", "of", "time", "traveller",
       "weena" and "morlocks", count its occurrences in every chunk, and
       compute the ratio of the sample variance to the sample mean of the
       counts. Which ratio does a Poisson model predict? Which words deviate
       most from it?
    1. Fit a Poisson and a negative binomial model
       (problem :exref:`ex_mdl-distributions-gamma-poisson-mixture`b) to the
       counts of each word by maximum likelihood, and compare the maximized
       log-likelihoods. Parameterize the negative binomial by its mean $m$
       and its shape $\alpha$. The maximum-likelihood $m$ is the sample mean,
       so only $\alpha$ requires a one-dimensional search, for example on a
       logarithmic grid.
    1. For each word, compare the empirical fraction of chunks that do not
       contain the word, and the empirical *adaptation*
       $P(X\ge2\mid X\ge1)$, with the predictions of both fitted models. What
       does the adaptation measure, and why does a Poisson model
       underestimate it for a name such as "traveller"?
    1. Repeat (a) for chunk lengths $L\in\{100,250,500,2000\}$. Assume that
       the rate of a word in a chunk is $L$ times a per-word rate drawn from a
       Gamma distribution with a fixed shape. Use problem
       :exref:`ex_mdl-distributions-gamma-poisson-mixture`a to predict how the
       variance-to-mean ratio grows with $L$, and compare the prediction with
       your measurements.
   :exlabel:`ex_mdl-distributions-word-burstiness`

1. **The Student-$t$ as a Gamma scale mixture.** ● A Gaussian whose precision
   (inverse variance) is itself random has heavier tails than any Gaussian.
   Let $\tau\sim\mathrm{Gamma}(\nu/2,\nu/2)$ with the density
   :eqref:`eq_mdl-gamma_pdf` (shape and rate both $\nu/2$, for some $\nu>0$),
   and let $X\mid\tau\sim\mathcal N(\mu,1/\tau)$.
    1. Integrate $\tau$ out to show that $X$ has the density

       $$p(x)=\frac{\Gamma\bigl(\frac{\nu+1}{2}\bigr)}{\sqrt{\pi\nu}\,\Gamma\bigl(\frac{\nu}{2}\bigr)}\Bigl(1+\frac{(x-\mu)^2}{\nu}\Bigr)^{-(\nu+1)/2},$$

       *Student's $t$ distribution* with $\nu$ degrees of freedom. *Hint:*
       $\int_0^\infty\tau^{a-1}e^{-c\tau}\,d\tau=\Gamma(a)/c^a$ for $a,c>0$.
    1. Show that $p(x)$ decays like $|x|^{-(\nu+1)}$, so that
       $\mathbb{E}|X-\mu|^k$ is finite only for $k<\nu$. Which distribution
       met in this section is the case $\nu=1$?
    1. Show that $\textrm{Var}(X)=\nu/(\nu-2)$ for $\nu>2$ by the law of total
       variance of exercise :exref:`ex_mdl-distributions-gamma-poisson-mixture`.
    1. [code] Draw $10^5$ samples for $\nu=3$ and $\mu=0$ by the mixture
       recipe (first $\tau$, then $X$), and compare the fraction with
       $|X|>3$ with the Gaussian's $0.0027$. Then draw $10^4$ sample means of
       20 draws each and compare their standard deviation with
       $\sqrt{3/20}$.
   :exlabel:`ex_mdl-distributions-student-t`

1. **Two factors of one covariance.** ● Let
   $\boldsymbol\Sigma=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$.
    1. Compute the Cholesky factor $\mathbf L$ by hand, and the eigen-factor
       $\mathbf M=\mathbf V\operatorname{diag}(\sqrt{\gamma_1},\sqrt{\gamma_2})$
       from the eigenvectors and eigenvalues found in the text.
       Verify $\mathbf L\mathbf L^\top=\mathbf M\mathbf M^\top=\boldsymbol\Sigma$.
    1. Show that any two invertible factors with
       $\mathbf M_1\mathbf M_1^\top=\mathbf M_2\mathbf M_2^\top$ satisfy
       $\mathbf M_2=\mathbf M_1\mathbf Q$ with $\mathbf Q$ orthogonal (*hint:*
       set $\mathbf Q=\mathbf M_1^{-1}\mathbf M_2$). What is the distribution
       of $\mathbf Q\mathbf z$ for $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$?
       Conclude that $\boldsymbol\mu+\mathbf M_1\mathbf z$ and
       $\boldsymbol\mu+\mathbf M_2\mathbf z$ have the same distribution.
    1. *Whitening.* Show that $\mathbf z=\mathbf L^{-1}(\mathbf x-\boldsymbol\mu)$
       has covariance $\mathbf I$ when
       $\mathbf x\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$. Explain why
       computing $\mathbf z$ for one $\mathbf x$ takes about $d^2$ operations
       once $\mathbf L$ is known, against about $d^3/3$ for computing
       $\mathbf L$.
    1. [code] Sample $10{,}000$ points by each factor, print both empirical
       covariances, whiten the Cholesky samples with a triangular solve, and
       print the covariance of the result.
   :exlabel:`ex_mdl-distributions-two-factors`

1. **Superposition, thinning, and the order of arrivals.** ● Requests reach a
   server from two independent sources, which are Poisson processes with
   rates $\lambda_1$ and $\lambda_2$.
    1. Show that the merged stream is a Poisson process with rate
       $\lambda_1+\lambda_2$. Use the slicing picture, or show by the binomial
       theorem that the sum of independent $\mathrm{Poisson}(\lambda_1)$ and
       $\mathrm{Poisson}(\lambda_2)$ counts is
       $\mathrm{Poisson}(\lambda_1+\lambda_2)$.
    1. Each request of a rate-$\lambda$ process is logged independently with
       probability $q$. Show that the logged requests form a Poisson process
       with rate $q\lambda$: condition on the number of requests and sum a
       binomial over a Poisson. With $\lambda=4$ per minute and $q=\tfrac14$,
       what is the probability that a minute logs nothing?
    1. Given that exactly $n$ events fell in $[0,t]$, show that their times
       have the distribution of $n$ independent $U(0,t)$ draws, sorted. Use
       the slicing picture: every slice is equally likely to hold an event.
    1. [code] Simulate a rate-$4$ process for $1{,}000$ minutes by exponential
       gaps, thin it with $q=\tfrac14$, and check that the thinned counts per
       minute have mean and variance close to $1$. Collect the positions
       within the minute of the events in minutes with exactly three events,
       and check that their histogram is flat.
   :exlabel:`ex_mdl-distributions-poisson-process`

1. **Covariance of the multinomial.** ● Write the multinomial count vector as
   $\mathbf x=\sum_{i=1}^n\mathbf e_{Y_i}$, a sum of $n$ independent one-hot
   vectors with $P(Y_i=k)=p_k$.
    1. Show that $\mathbb{E}[\mathbf e_Y]=\mathbf p$ and
       $\textrm{Cov}(\mathbf e_Y)=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$,
       and hence that $\mathbb{E}[\mathbf x]=n\mathbf p$ and
       $\textrm{Cov}(\mathbf x)=n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$.
       Read off $\textrm{Var}(x_k)=np_k(1-p_k)$ and
       $\textrm{Cov}(x_j,x_k)=-np_jp_k$ for $j\ne k$. Why is every covariance
       negative?
    1. Show that $\textrm{Cov}(\mathbf x)\mathbf 1=\mathbf 0$, and explain this
       by the constraint $\sum_kx_k=n$.
    1. For 12 rolls of a fair die, compute $\textrm{Var}(x_k)$,
       $\textrm{Cov}(x_j,x_k)$, and the correlation between two counts.
    1. Show that $x_k$ alone is $\mathrm{Binomial}(n,p_k)$ by grouping the
       other $K-1$ classes into one.
   :exlabel:`ex_mdl-distributions-multinomial-covariance`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1099)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1099)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.2]{.kicker}

Distributions<br>**from one coin flip to the Gaussian**
:::
:::

::: {.slide title="9 Heads in 13 Flips: How Surprising Is That?"}
[The problem]{.kicker}

- A coin lands heads 9 times in 13 flips. If it is fair, how likely is that?
- Every model that predicts a label has chosen a law for it, and the law fixes the loss:

| model output | law of the label | loss |
|:--|:--|:--|
| $P(y=1)$ | Bernoulli | binary cross-entropy |
| softmax over $K$ classes | categorical | cross-entropy |
| a predicted value | Gaussian | squared error |
| a robust predicted value | Laplace | absolute error |

::: {.d2l-note .warn}
A wrong law gives wrong error bars: a Poisson count must have variance equal to its mean.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute a binomial probability by a ratio, without factorials;
- derive each law's mean and variance from how it is built;
- derive the exponential wait from the Poisson count;
- state the central limit theorem and show where it fails;
- sample a Gaussian vector and condition it on an observation.

**Running examples:** the coin (9 heads in 13 flips); a server at 4 requests per minute;
$\boldsymbol\Sigma$ with rows $(2,1)$, $(1,2)$.
**Idea:** a few laws grow out of one coin flip by sums, limits and waiting times.
:::

::: {.slide title="From Densities to Named Laws"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Random Variables | densities, moments, covariance |
| **Distributions** | **the named laws and their losses** |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP, EM |
| Bayesian Computation | averaging over a posterior |
| Concentration Inequalities | exponential tails of averages |
| Uniform Convergence | guarantees after selection |
| Naive Bayes | one classifier, end to end |
:::

::: {.col .narrow}
- **Uses:** mean, variance, change of variables, inverse transform.
- **Skips:** heavy tails beyond the Laplace, event rates that change in time.
:::
:::
:::

::: {.slide title="One Flip Has Mean p and Variance p(1−p)"}
[Bernoulli]{.kicker}

$P(X=x)=p^x(1-p)^{1-x}$ for $x\in\{0,1\}$. Since $X^2=X$:

$$
\mathbb{E}[X]=p, \qquad \textrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2=p-p^2=p(1-p).
$$

- $p(1-p)$ is $0.25$ at $p=\tfrac12$, $0.09$ at $p=0.1$ or $0.9$, $0$ at $0$ or $1$.
- The running coin, $p=9/13$: variance $\approx0.21$.

::: {.d2l-note .rule}
Binary cross-entropy $-y\log\hat p-(1-y)\log(1-\hat p)$ is the Bernoulli's negative log-probability.
:::
:::

::: {.slide title="13 Flips Have Mean 6.5, Variance 3.25, and P(9) = 715/8192"}
[Binomial]{.kicker}

$X=\sum_{i=1}^n X_i$, independent flips with one $p$:
$P(X=k)=\binom nk p^k(1-p)^{n-k}$, orderings times one ordering.

- Linearity: $\mathbb{E}[X]=np$. Independence: $\textrm{Var}(X)=np(1-p)$.
- Ratio: $\dfrac{P(X=k+1)}{P(X=k)}=\dfrac{n-k}{k+1}\cdot\dfrac{p}{1-p}$, from $P(X=0)=(1-p)^n$.

@!distributions-binomial

9 heads lie $1.4$ standard deviations above the mean.
:::

::: {.slide title="Many Rare Chances Give the Poisson, with Mean = Variance = λ"}
[Poisson]{.kicker}

Slice a minute into $n$ pieces, each $\mathrm{Bernoulli}(4/n)$; let $n\to\infty$:
$P(X=k)\to\lambda^ke^{-\lambda}/k!$. Silent minute: $(1-\tfrac4n)^n=0.0115,\ 0.0169,\ 0.0182\to e^{-4}$.

@!distributions-binomial-to-poisson
:::

::: {.slide title="Check: A Silent Minute, and Exactly Four Requests"}
[Check]{.kicker}

A server gets 4 requests per minute on average, modeled as Poisson.
What is the probability of a minute with no request? With exactly 4?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $P(0)=e^{-4}\approx0.018$, about once an hour.
$P(4)=\tfrac{4^4}{4!}e^{-4}\approx0.195$: even the most likely count (4, tied with 3) happens in only one minute of five.
:::
:::
:::

::: {.slide title="K Outcomes Give the Categorical; n Draws Give the Multinomial"}
[Categorical]{.kicker}

- $P(X=k)=p_k$; softmax $p_k=e^{z_k}/\sum_je^{z_j}$ turns logits into a categorical.
- Logits $(2,1,0.1,-1)$ give $\mathbf p\approx(0.64,0.23,0.10,0.03)$.
- Cross-entropy $-\log p_y$ is its negative log-probability.
- Counts of $n$ draws: $\dfrac{n!}{x_1!\cdots x_K!}\prod_kp_k^{x_k}$; for $K=2$, $\dfrac{13!}{9!\,4!}=715$.
- Equal $p_k=\tfrac1n$: the die, mean $\tfrac{n+1}2=3.5$, variance $\tfrac{n^2-1}{12}\approx2.92$.
:::

::: {.slide title="The Wait for the Next Event Is Exponential"}
[Exponential]{.kicker}

@fig:mdl-prob-poisson-timeline

$P(T>t)=P(N(t)=0)=e^{-\lambda t}$, so $p(t)=\lambda e^{-\lambda t}$. At $\lambda=4$/min:
mean wait $1/\lambda=15$ s, median $\log2/\lambda\approx10.4$ s.
:::

::: {.slide title="The Exponential Forgets How Long You Have Waited"}
[Memorylessness]{.kicker}

$$
P(T>s+t\mid T>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}=P(T>t).
$$

- The $k$-th event's time is a sum of $k$ waits: $\mathrm{Gamma}(k,\lambda)$, mean $k/\lambda$.
- $P(T_4>1)=P(N(1)\le3)\approx0.43$.

@!distributions-exponential
:::

::: {.slide title="Every Sampler Starts from U(0,1)"}
[Uniform]{.kicker}

- $U(a,b)$: density $\tfrac1{b-a}$; every subinterval of equal length is equally likely.
- Mean $\tfrac{a+b}2$, variance $\tfrac{(b-a)^2}{12}$; $U(-\sqrt3,\sqrt3)$ has variance $1$.
- Inverse transform: $X=F^{-1}(U)$ has c.d.f. $F$.
- Exponential: $T=-\log(1-U)/\lambda$. With $-\log U$, the draw $U=0$ would give $\infty$.

@!distributions-continuous-uniform
:::

::: {.slide title="I. The Binomial's Own Ratio Turns into a Parabola"}
[Derivation]{.kicker}

Put $k=np+x$, $q=1-p$, and use $\log(1+\epsilon)\approx\epsilon$:

$$
\log\frac{P(k+1)}{P(k)}=\log\frac{1-x/(nq)}{1+(x+1)/(np)}\approx-\frac{x}{npq}.
$$

Steps linear in $x$ add up to a parabola: $\log P(np+x)\approx C-\dfrac{x^2}{2npq}$,
a bell of variance $npq$.

@!mdl-distributions-gaussian-2
:::

::: {.slide title="II. Standardized Sums with Finite Variance Become Gaussian"}
[Central limit theorem]{.kicker}

$X_i$ iid, mean $\mu$, variance $0<\sigma^2<\infty$, $S_n=\sum_{i\le n}X_i$:
$P\bigl((S_n-n\mu)/(\sigma\sqrt n)\le y\bigr)\to\Phi(y)$. Standardized sums of uniforms:

@!mdl-distributions-gaussian-1

Only two moments enter.
:::

::: {.slide title="III. Normalizer by Polar Coordinates, Variance by Parts"}
[Derivation]{.kicker}

$$
I^2=\iint e^{-(x^2+y^2)/2}\,dx\,dy=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta=2\pi .
$$

$$
\int x^2\frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx=\frac{1}{\sqrt{2\pi}}\Bigl(\bigl[-xe^{-x^2/2}\bigr]+\int e^{-x^2/2}\,dx\Bigr)=1 .
$$

::: {.d2l-note .rule}
$\mathcal N(\mu,\sigma^2)$: $p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$, mean $\mu$, variance $\sigma^2$.
:::
:::

::: {.slide title="Check: Does the Average of 1,000 Cauchy Draws Settle?"}
[Check]{.kicker}

The Cauchy density $1/(\pi(1+x^2))$ has infinite variance. Average $n$ independent
draws. Does $P(|\text{average}|>1)$ shrink as $n$ grows?

::: {.fragment .fade-up}
No. The average of Cauchy draws is again Cauchy, so the probability stays $\tfrac12$:

@!distributions-gaussian
:::
:::

::: {.slide title="The Laplace Is a Random Sign Times an Exponential"}
[Laplace]{.kicker}

::: {.cols .vc}
::: {.col}
$X=\mu+SE$: $p(x)=\frac1{2b}e^{-|x-\mu|/b}$, $\textrm{Var}=\mathbb{E}[E^2]=2b^2$.

Beyond $3\sigma$: $e^{-3\sqrt2}\approx0.014$ against $0.0027$, five times the Gaussian.

Its negative log-probability is the absolute error.
:::

::: {.col .fig .big}
@fig:mdl-prob-continuous-pdfs
:::
:::
:::

::: {.slide title="A Gaussian Vector's Contours Are the Eigen-Ellipses of Σ"}
[Multivariate Gaussian]{.kicker}

::: {.cols .vc}
::: {.col}
$p(\mathbf x)\propto\exp\bigl(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\bigr)$

- $\boldsymbol\Sigma$ with rows $(2,1)$, $(1,2)$: sum and difference
  coordinates have variances $3$ and $1$, covariance $0$.
- Axes along $\mathbf v_i$, half-lengths $\propto\sqrt{\gamma_i}$: $\sqrt3$ and $1$.
:::

::: {.col .fig}
@fig:mdl-prob-mvn-contours
:::
:::
:::

::: {.slide title="Linear Maps Keep a Gaussian Gaussian, and Sample It"}
[Linear maps]{.kicker}

- Full row rank $\mathbf A$: $\mathbf A\mathbf x+\mathbf b\sim\mathcal N(\mathbf A\boldsymbol\mu+\mathbf b,\ \mathbf A\boldsymbol\Sigma\mathbf A^\top)$.
- Sampling: $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$ with $\mathbf L\mathbf L^\top=\boldsymbol\Sigma$, $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$.
- Cholesky by hand: $L_{11}=\sqrt2$, $L_{21}=1/\sqrt2$, $L_{22}=\sqrt{3/2}$.
- Any factor works; two factors differ by an orthogonal matrix.

@!distributions-mvn
:::

::: {.slide title="Conditioning a Gaussian Is Linear Regression"}
[Conditioning]{.kicker}

$\mathbf B=\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}$; the residual
$\mathbf r=\mathbf x_1-\mathbf B\mathbf x_2$ has $\textrm{Cov}(\mathbf r,\mathbf x_2)=\mathbf 0$,
and $(\mathbf r,\mathbf x_2)$ is jointly Gaussian, so they are independent. Hence

$$
\boldsymbol\mu_{1\mid2}=\boldsymbol\mu_1+\mathbf B(\mathbf x_2-\boldsymbol\mu_2),
\qquad
\boldsymbol\Sigma_{1\mid2}=\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.
$$

Running example, $x_2=2$: $B=\tfrac12$, so $x_1\mid x_2=2\sim\mathcal N(1,\tfrac32)$.

@!mdl-distributions-conditioning
:::

::: {.slide title="Check: Condition a Correlated Pair"}
[Check]{.kicker}

$(x_1,x_2)$ is Gaussian with mean $\mathbf 0$, unit variances and correlation $\rho$,
$|\rho|<1$. What is the law of $x_1$ given $x_2$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $B=\rho$, so $x_1\mid x_2\sim\mathcal N(\rho x_2,\,1-\rho^2)$.
The variance $1-\rho^2\le1$ does not depend on $x_2$: observing never widens the uncertainty.
:::
:::
:::

::: {.slide title="One Coin Flip, a Dozen Laws, Their Moments and Their Losses"}
[Wrap-up]{.kicker}

- 9 heads in 13 fair flips: $715/8192\approx0.087$; mean $6.5$, variance $3.25$.
- Server at 4/min: a silent minute and a wait over a minute both have probability $e^{-4}\approx0.018$.
- The bell at $n=13$: $0.085$ against the exact $0.087$; the CLT needs a finite variance.
- Laplace: five times the Gaussian's mass beyond $3\sigma$; absolute error as its loss.
- $\boldsymbol\Sigma$ with rows $(2,1)$, $(1,2)$: eigenvalues $3$ and $1$; $x_1\mid x_2=2\sim\mathcal N(1,\tfrac32)$.

**Next:** the one form these laws share; then how well 13 flips determine an unknown bias.
:::
