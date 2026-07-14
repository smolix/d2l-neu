# Distributions
:label:`sec_mdl-distributions`

A *distribution* assigns a probability to every outcome a random variable can
take (:numref:`sec_mdl-random_variables`). A working deep-learning practitioner
does not need a zoo of hundreds: the fourteen distributions of this section
cover almost everything, and they form a small family with a few generators and
a handful of limiting arrows connecting them.
The Bernoulli coin flip is the seed: sum $n$ of them and you get the binomial;
let the trials become many and rare and the binomial collapses to the Poisson;
let them become many and ordinary and the *central limit theorem* sends it to the
Gaussian. The categorical generalizes the Bernoulli from two outcomes to $K$, and
its softmax parameterization is the output layer of every classifier. At the end
we will see that almost all of these (and more) are special cases of one form,
the *exponential family*, which is the reason their maximum-likelihood losses are the
convex objectives we minimize in practice and the reason each has a
*conjugate prior* that makes Bayesian updating as simple as counting.

:numref:`fig_mdl-prob-family-tree` draws the picture: the distributions are
nodes, the construction and limit relationships are arrows, and the whole
picture sits inside the exponential-family envelope.

![The distribution family. Solid arrows *construct*: Bernoulli is the seed; summing $n$ copies gives the Binomial; the many-and-rare limit ($n\to\infty$, $np\to\lambda$) gives the Poisson, whose waiting time between events is the Exponential; the many-and-ordinary limit (the central limit theorem) gives the Gaussian; the Categorical generalizes Bernoulli to $K$ outcomes and the Multinomial counts $n$ of them. Dashed arrows attach each likelihood to its *conjugate prior*: the Beta to the Bernoulli and Binomial, the Gamma to the Poisson, and the Dirichlet to the Categorical and Multinomial. Every node, the conjugate priors included, lies inside the exponential-family envelope.](../img/mdl-prob-family-tree.svg)
:label:`fig_mdl-prob-family-tree`

For each distribution we keep to a tight template: its mass or density function,
where it *arises* in machine learning, its mean and variance *derived* rather than
asserted, and a compact teaching cell that evaluates the law and draws a
sample. We treat the discrete distributions first, then the continuous ones, then
unify them, using the following imports throughout.

```{.python .input #distributions-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #distributions-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input #distributions-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #distributions-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
This section works in plain NumPy: every computation here is elementary array
arithmetic, so nothing beyond NumPy is needed.
:end_tab:

## Discrete Distributions

A discrete random variable takes values in a countable set, and its distribution
is a *probability mass function* (pmf) $p_k = P(X=k)$ with $p_k\ge0$ and
$\sum_k p_k=1$. Mean and variance are the sums
$\mathbb E[X]=\sum_k k\,p_k$ and
$\operatorname{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2$
(:numref:`sec_mdl-random_variables`). The five distributions here form a
connected chain: each is built from, or is a limit of, the one
before. :numref:`fig_mdl-prob-discrete-pmfs` shows their shapes side by side so the
shared bell-like silhouette of the count distributions is visible at a glance.

![A gallery of discrete probability mass functions in one consistent style: $\mathrm{Bernoulli}(0.3)$, a discrete uniform on $\{1,\dots,6\}$, $\mathrm{Binomial}(10,0.4)$, and $\mathrm{Poisson}(4)$. The binomial and Poisson share the same right-skewed bell silhouette, foreshadowing their limiting relationship.](../img/mdl-prob-discrete-pmfs.svg)
:label:`fig_mdl-prob-discrete-pmfs`

### Bernoulli

The Bernoulli is the simplest non-trivial random variable: a single coin flip that
comes up $1$ with probability $p$ and $0$ with probability $1-p$. We write
$X\sim\mathrm{Bernoulli}(p)$, with pmf

$$
p_1 = P(X=1) = p, \qquad p_0 = P(X=0) = 1-p .
$$
:eqlabel:`eq_mdl-bernoulli_pmf`

**Where it arises.** The Bernoulli is the output distribution of *every binary
classifier*: a model emits $\hat p$, and the negative log-likelihood
$-y\log\hat p-(1-y)\log(1-\hat p)$ is exactly the *binary cross-entropy* loss,
the maximum-likelihood view of binary classification developed in
:numref:`subsec_mdl-nll-crossentropy`.

**Mean and variance.** Because $X$ takes only the values $0$ and $1$, we have
$X^2=X$, which makes both moments immediate:

$$
\mu_X = \mathbb E[X] = 1\cdot p + 0\cdot(1-p) = p,
\qquad
\sigma_X^2 = \mathbb E[X^2]-\mu_X^2 = p - p^2 = p(1-p).
$$

The variance $p(1-p)$ is largest at $p=\tfrac12$ (the fairest, least predictable
coin) and vanishes at $p\in\{0,1\}$ (a certain outcome carries no randomness).

```{.python .input #distributions-bernoulli}
import numpy as onp
rng = onp.random.default_rng(0)
p = 0.3
sample = 1 * (rng.random((3, 3)) < p)          # 1 with prob p, else 0
big = 1 * (rng.random(10000) < p)              # large sample for the mean
print('pmf  P(0), P(1) =', (1 - p, p))
print('mean of 10,000 draws =', float(big.mean()), ' (≈ p)')
sample
```

### Categorical and Multinomial

The *categorical* distribution generalizes the Bernoulli from two outcomes to $K$.
It is parameterized by a probability vector $\mathbf p=(p_1,\dots,p_K)$ with
$p_k\ge0$ and $\sum_k p_k=1$, and a draw lands on class $k$ with probability $p_k$,

$$
P(X=k) = p_k .
$$
:eqlabel:`eq_mdl-categorical_pmf`

**Where it arises.** Every *softmax* output layer turns a vector of scores
(logits) $\mathbf z$ into a categorical over classes or vocabulary tokens via

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}},
$$
:eqlabel:`eq_mdl-softmax`

and the negative log-likelihood of the categorical,
$-\sum_k y_k\log p_k$ with $\mathbf y$ the one-hot label, is exactly the softmax
*cross-entropy* loss (:numref:`subsec_mdl-nll-crossentropy`, :numref:`sec_softmax`).
The Bernoulli is the case $K=2$. To *sample* a class rather than score one, the
*Gumbel-max trick* perturbs each logit with independent Gumbel noise (the Gumbel
distribution is the location--scale law of the maxima of many independent draws
:cite:`Gumbel.1954`) and takes the argmax: an exact categorical sampler, and the
basis of the Gumbel-softmax relaxation that lets gradients flow through discrete
choices :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017`.

The *multinomial* is its count version, the categorical analogue of how the
binomial sums Bernoullis. Drawing $n$ independent categorical samples and counting
how often each class occurs gives a vector $\mathbf x=(x_1,\dots,x_K)$ with
$\sum_k x_k=n$ and pmf

$$
P(\mathbf x) = \frac{n!}{x_1!\cdots x_K!}\,\prod_{k=1}^{K} p_k^{x_k},
$$
:eqlabel:`eq_mdl-multinomial_pmf`

the multinomial coefficient counting the orderings that yield those counts, times
the probability $\prod_k p_k^{x_k}$ of any one such ordering. For $K=2$ this is the
binomial of the next section.

The teaching cell turns a logit vector into a categorical via softmax, confirms it
normalizes, and draws one sample.

```{.python .input #distributions-categorical}
import numpy as onp
rng = onp.random.default_rng(0)
z = onp.array([2.0, 1.0, 0.1, -1.0])            # logits over K = 4 classes
p_cat = onp.exp(z) / onp.exp(z).sum()            # softmax -> categorical
draw = int(rng.choice(len(p_cat), p=p_cat))
print('categorical p =', p_cat.round(3), ' sum =', float(p_cat.sum()))
print('one sample -> class', draw)
```

### Discrete Uniform

If a categorical assigns *equal* probability to each of its outcomes it is the
*discrete uniform*. Supported on $\{1,2,\dots,n\}$, every value is equally likely,
$p_i=\tfrac1n$, written $X\sim U(n)$.

**Where it arises.** It models a fair die, a uniformly chosen index, or a token
sampled with no preference. It is also the maximum-entropy distribution on a
finite set, a fact we take as given :cite:`Cover.Thomas.1999` and place in
context when we meet the exponential family below.

**Mean and variance.** Both follow from the closed forms
$\sum_{i=1}^n i=\tfrac{n(n+1)}2$ and $\sum_{i=1}^n i^2=\tfrac{n(n+1)(2n+1)}6$:

$$
\mu_X = \frac1n\sum_{i=1}^n i = \frac{n+1}{2},
\qquad
\sigma_X^2 = \frac1n\sum_{i=1}^n i^2 - \mu_X^2 = \frac{n^2-1}{12}.
$$

The mean is the midpoint, by symmetry; the variance grows like $n^2$, since a wider
range of equally likely values is more spread out. The cell checks both formulas
against a large sample of die rolls.

```{.python .input #distributions-discrete-uniform}
rng = np.random.default_rng(0)
n = 6
sample = rng.integers(1, n + 1, 10000)           # 10,000 die rolls
print('mean (n+1)/2  =', (n + 1) / 2, '   sample mean:', float(sample.mean()))
print('var (n^2-1)/12 =', round((n**2 - 1) / 12, 3), ' sample var :',
      float(sample.var().round(3)))
```

### Binomial

Sum $n$ independent Bernoulli flips and you count *successes*: this is the
*binomial*. Writing each flip as $X_i\sim\mathrm{Bernoulli}(p)$ with $1$ for success,

$$
X = \sum_{i=1}^{n} X_i \sim \mathrm{Binomial}(n,p).
$$
:eqlabel:`eq_mdl-binomial_sum`

Exactly $k$ successes can occur in $\binom nk$ orderings, each of probability
$p^k(1-p)^{n-k}$, so

$$
P(X=k) = \binom{n}{k}\,p^k(1-p)^{n-k}, \qquad k=0,1,\dots,n.
$$
:eqlabel:`eq_mdl-binomial_pmf`

**Where it arises.** It models any "how many out of $n$" count with a fixed
per-trial probability: clicks among impressions, correct predictions on a held-out
batch, heads in a run of flips.

**Mean and variance, via the sum.** Rather than wrestle with the binomial
coefficient, use the representation :eqref:`eq_mdl-binomial_sum` as a sum of
Bernoullis. *Expectation is linear* (always, even for dependent terms), so

$$
\mu_X = \mathbb E\Bigl[\sum_i X_i\Bigr] = \sum_i \mathbb E[X_i] = np .
$$

The $X_i$ are *independent*, and variance is additive over independent summands, so

$$
\sigma_X^2 = \operatorname{Var}\Bigl(\sum_i X_i\Bigr) = \sum_i \operatorname{Var}(X_i) = np(1-p) .
$$

Two one-line sums replace a page of algebra: the payoff of seeing the binomial as
*built from* Bernoullis. Successive pmf terms differ by the ratio

$$
\frac{P(X=k+1)}{P(X=k)} = \frac{n-k}{k+1}\cdot\frac{p}{1-p},
$$

so the whole pmf builds up from $P(X=0)=(1-p)^n$ by repeated multiplication, with
never a factorial in sight.

```{.python .input #distributions-binomial}
import numpy as onp
rng = onp.random.default_rng(0)
n, p = 10, 0.4
pmf = onp.zeros(n + 1)
pmf[0] = (1 - p)**n
for k in range(n):                               # successive-ratio recursion
    pmf[k + 1] = pmf[k] * (n - k) / (k + 1) * p / (1 - p)
print('mean onp =', n * p, '  var onp(1-p) =', n * p * (1 - p))
print('P(X=k):', pmf.round(3))
rng.binomial(n, p, size=(3, 3))                  # sample counts of successes
```

### Poisson

What if the trials become *many* and each *rare*? Standing at a bus stop, the
chance of an arrival in any tiny sub-interval is small, but there are many such
intervals. Let $\lambda$ be the expected number of arrivals per minute. Split the
minute into $n$ slices, model each as $\mathrm{Bernoulli}(\lambda/n)$, and the
count is $\mathrm{Binomial}(n,\lambda/n)$. Its mean is $n\cdot\lambda/n=\lambda$
for every $n$, and its variance
$n\cdot\tfrac{\lambda}{n}(1-\tfrac{\lambda}{n})\to\lambda$.
The moments stabilize, which signals that a limiting distribution exists.

**The limit, derived.** Take $\mathrm{Binomial}(n,p_n)$ with $n\to\infty$ and
$p_n\to0$ such that $np_n\to\lambda$, the *law of rare events*. Writing
$p_n=\lambda/n$, the pmf at a fixed $k$ factors into four pieces:

$$
\binom{n}{k}\Bigl(\frac\lambda n\Bigr)^k\Bigl(1-\frac\lambda n\Bigr)^{n-k}
= \underbrace{\frac{n!}{(n-k)!\,n^k}}_{\to\,1}
  \cdot \frac{\lambda^k}{k!}
  \cdot \underbrace{\Bigl(1-\frac\lambda n\Bigr)^{n}}_{\to\,e^{-\lambda}}
  \cdot \underbrace{\Bigl(1-\frac\lambda n\Bigr)^{-k}}_{\to\,1} .
$$

The first factor is $\tfrac{n(n-1)\cdots(n-k+1)}{n^k}\to1$, the third is the
classic limit $(1-\lambda/n)^n\to e^{-\lambda}$, and the last $\to1$. Collecting
the survivors gives the *Poisson* pmf:

$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k=0,1,2,\dots
$$
:eqlabel:`eq_mdl-poisson_pmf`

with $\lambda>0$ the *rate*, the average count per unit time; we write
$X\sim\mathrm{Poisson}(\lambda)$.

**Where it arises.** Counts of rare events with no fixed ceiling: click counts,
photons on a sensor, mutations in a genome, requests hitting a server. Run over
the whole timeline, the bus-stop slicing above is the *Poisson process*: the
event counts in disjoint time windows are independent Poisson variables.

**Mean and variance.** The binomial moments suggest the answer, but
convergence in distribution alone does not automatically pass unbounded
moments to the limit. The Poisson pmf verifies them directly:

$$
E[X]=\sum_{k\ge1}k e^{-\lambda}\frac{\lambda^k}{k!}
=\lambda,
\qquad
E[X(X-1)]=\lambda^2.
$$

Therefore

$$
\mu_X=\lambda,
\qquad
\sigma_X^2=E[X(X-1)]+E[X]-E[X]^2=\lambda.
$$

Mean equals variance is the Poisson fingerprint: over-dispersed count data (where
the empirical variance exceeds the mean) is the standard sign that a plain Poisson
model is too simple. The standard fix, a Poisson whose rate is *itself random*,
reappears later in this section as the Gamma--Poisson mixture, better known as the
*negative binomial*.

```{.python .input #distributions-poisson}
import numpy as onp
rng = onp.random.default_rng(0)
lam = 4.0
k = onp.arange(15)
pmf = onp.empty_like(k, dtype=float)
pmf[0] = onp.exp(-lam)
for j in range(len(k) - 1):                    # P(j+1) / P(j) = lambda / (j+1)
    pmf[j + 1] = pmf[j] * lam / (j + 1)
print('mean = var = lambda =', lam)
print('P(X=k):', pmf.round(3))
rng.poisson(lam, size=(3, 3))                   # sample event counts
```

**Watching the limit happen.** The limit is visible in a plot. The cell below
overlays the pmf of $\mathrm{Binomial}(n,\lambda/n)$ for
growing $n$ on its $\mathrm{Poisson}(\lambda)$ limit, building every pmf by the
same successive-ratio trick we used for the binomial (for the Poisson the ratio is
simply $P(X=k+1)/P(X=k)=\lambda/(k+1)$). Already at $n=20$ the curves are visibly
closing in on each other, and at $n=100$ they coincide to plotting accuracy.

```{.python .input #distributions-binomial-to-poisson}
lam, ks = 4.0, np.arange(15)

def binomial_pmf(n, p):                # successive-ratio recursion, no factorials
    pmf = [(1 - p)**n]
    for k in range(len(ks) - 1):
        pmf.append(pmf[-1] * max(n - k, 0) / (k + 1) * p / (1 - p))
    return np.array(pmf)

poisson_pmf = [np.exp(-lam)]
for k in range(len(ks) - 1):
    poisson_pmf.append(poisson_pmf[-1] * lam / (k + 1))
d2l.plot(ks, [binomial_pmf(n, lam / n) for n in (5, 20, 100)]
         + [np.array(poisson_pmf)], xlabel='k', ylabel='P(X=k)',
         legend=[f'Binomial({n}, 4/{n})' for n in (5, 20, 100)] + ['Poisson(4)'])
```

## Continuous Distributions

A continuous random variable is described by a *probability density* (pdf)
$p(x)\ge0$ with $\int p(x)\,dx=1$; probabilities are areas under it and
expectations are integral averages (:numref:`sec_mdl-integral_calculus`). The five
laws below run from the structureless (uniform) to the universal (Gaussian) to the
vector-valued (multivariate Gaussian), with the exponential covering waiting times
on $[0,\infty)$ and the Laplace the symmetric case with heavier-than-Gaussian
tails.
:numref:`fig_mdl-prob-continuous-pdfs` overlays the three densities on the whole
line so their tail behavior is directly comparable: the Gaussian's thin
$e^{-x^2}$ tail, the Laplace's heavier $e^{-|x|}$ tail and sharp peak, and the
uniform's flat plateau.

![A gallery of continuous probability densities in one consistent style: a continuous uniform on the interval from $-2$ to $2$, a standard Gaussian $\mathcal N(0,1)$, and a Laplace with matched variance. The Laplace has a sharper peak and heavier tails than the Gaussian; the uniform is a flat plateau with hard edges.](../img/mdl-prob-continuous-pdfs.svg)
:label:`fig_mdl-prob-continuous-pdfs`

### Continuous Uniform

Refine the discrete uniform (let the $n$ equally likely points fill an interval
$[a,b]$) and you reach the *continuous uniform*, which picks a value in $[a,b]$
with every value equally likely. The density is constant on the interval and zero
outside, with the constant fixed by $\int p=1$:

$$
p(x) = \begin{cases}\frac{1}{b-a} & x\in[a,b],\\ 0 & \text{otherwise,}\end{cases}
\qquad X\sim U(a,b).
$$
:eqlabel:`eq_mdl-cont_uniform_pdf`

**Where it arises.** It is the raw material of randomness: the $U(0,1)$ generator
underlies inverse-transform sampling (:numref:`sec_mdl-random_variables`), Monte
Carlo, dropout masks, and random weight initialization.

**Mean and variance.** Direct integration, with the substitution $x\mapsto x-\tfrac{a+b}2$ centering the interval,

$$
\mu_X = \int_a^b \frac{x}{b-a}\,dx = \frac{a+b}{2},
\qquad
\sigma_X^2 = \int_a^b \frac{(x-\mu_X)^2}{b-a}\,dx = \frac{(b-a)^2}{12}.
$$

These are the continuous mirrors of the discrete-uniform formulas (with the range
$b-a$ playing the role of $n$). The default generator samples $U(0,1)$; shifting
and scaling gives any $U(a,b)$, and the cell checks the formulas against a large
sample.

```{.python .input #distributions-continuous-uniform}
rng = np.random.default_rng(0)
a, b = 1.0, 3.0
sample = (b - a) * rng.random(10000) + a         # shift-and-scale U(0,1)
print('mean (a+b)/2   =', (a + b) / 2, '  sample mean:',
      float(sample.mean().round(3)))
print('var (b-a)^2/12 =', round((b - a)**2 / 12, 3), ' sample var :',
      float(sample.var().round(3)))
```

### Exponential

If the Poisson counts how *many* rare events land in a window, the *exponential*
models the *waiting time between* them. It is the canonical density on $[0,\infty)$,

$$
p(x) = \lambda e^{-\lambda x}\ (x\ge0),
\qquad
F(x) = 1 - e^{-\lambda x},
\qquad X\sim\mathrm{Exp}(\lambda),
$$
:eqlabel:`eq_mdl-exponential_pdf`

with rate $\lambda$ matching the Poisson rate: more frequent events mean shorter
waits.

**Where it arises.** Inter-arrival times, time-to-failure in reliability, and
survival analysis: anywhere a constant *hazard* (the instantaneous failure rate
per unit time, here independent of the age already reached) is plausible.

**Memorylessness.** The exponential is the *unique* continuous distribution with no
memory: having already waited $s$, the chance of waiting $t$ longer is the same as
starting fresh. Using $P(X>x)=e^{-\lambda x}$,

$$
P(X>s+t \mid X>s)
= \frac{P(X>s+t)}{P(X>s)}
= \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}
= e^{-\lambda t}
= P(X>t).
$$

The waiting clock never "ages", exactly the property that makes the exponential
the continuous-time partner of the Poisson. The converse, that memorylessness
*forces* the exponential form, is Exercise 4.

**Mean and variance.** With the standard integrals $\int_0^\infty u\,e^{-u}\,du=1$
and $\int_0^\infty u^2 e^{-u}\,du=2$ (integration by parts,
:numref:`sec_mdl-integral_calculus`), substitute $u=\lambda x$:

$$
\mu_X = \int_0^\infty x\,\lambda e^{-\lambda x}\,dx = \frac1\lambda,
\qquad
\mathbb E[X^2] = \frac{2}{\lambda^2}
\;\Rightarrow\;
\sigma_X^2 = \frac{2}{\lambda^2}-\frac1{\lambda^2} = \frac1{\lambda^2}.
$$

The teaching cell samples by the *inverse-transform* proposition of
:numref:`sec_mdl-random_variables`; this is the closed-form case promised
there. Inverting $F(x)=1-e^{-\lambda x}$ gives
$F^{-1}(q)=-\tfrac1\lambda\log(1-q)$, and since $1-U$ is again uniform when
$U\sim U(0,1)$, the sampler simplifies to
$X=-\tfrac1\lambda\log U\sim\mathrm{Exp}(\lambda)$.

```{.python .input #distributions-exponential}
import numpy as onp
rng = onp.random.default_rng(0)
lam = 0.5
U = rng.random(100000)
sample = -onp.log(U) / lam                        # inverse-transform sampler
print('mean 1/lambda =', 1 / lam, '  sample mean =', float(sample.mean().round(3)))
print('var 1/lambda^2 =', 1 / lam**2, ' sample var =', float(sample.var().round(3)))
```

### Gaussian

The *Gaussian* (or normal) distribution is what sums of many small independent
effects look like, and such sums are everywhere.
Return to the binomial $X^{(n)}\sim\mathrm{Binomial}(n,p)$, but now hold $p$ fixed
and send $n\to\infty$. Both mean $np$ and variance $np(1-p)$ blow up, so we
standardize,

$$
Y^{(n)} = \frac{X^{(n)} - np}{\sqrt{np(1-p)}},
$$

which has mean $0$ and variance $1$ for every $n$. The *central limit theorem*
(CLT) states that $Y^{(n)}$ converges in distribution to the *standard Gaussian*:
for any interval $[a,b]$, $P(Y^{(n)}\in[a,b])\to P(\mathcal N(0,1)\in[a,b])$. This
binomial case is the *de Moivre--Laplace theorem*, historically the first central
limit theorem. The limiting density is

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr),
\qquad X\sim\mathcal N(\mu,\sigma^2).
$$
:eqlabel:`eq_mdl-gaussian_pdf`

Nothing about coin flips was special: the *Lindeberg--Lévy* form of the CLT holds
for any independent, identically distributed summands with finite variance (no
higher moments needed), which is why the conditions are so easy to meet and the
Gaussian is everywhere. We take the theorem as given; :cite:`Wasserman.2013`
gives the proof.

**Watching universality happen.** Like the Poisson limit, this claim is
checkable in a few lines. The cell below starts from about the least Gaussian
summand available (a $U(0,1)$ draw, flat with hard edges and no tails), forms
sums of $n$ of them, standardizes each sum by its mean $n/2$ and standard
deviation $\sqrt{n/12}$, and overlays the resulting histograms on the standard
normal density. At $n=1$ we see the uniform's own plateau; at $n=2$ the
triangular hat; by $n=8$ the curve is already hard to distinguish from the
Gaussian, and at $n=32$ the two agree to the resolution of the plot. Nothing in
the code knows about the Gaussian: the bell is manufactured by the summation
itself, which is exactly the universality the CLT asserts.

```{.python .input #mdl-distributions-gaussian-1}
rng = np.random.default_rng(0)
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

**The normalizer and the moments.** Why the constant $1/\sqrt{2\pi\sigma^2}$?
The identity it rests on, $\int_{-\infty}^\infty e^{-x^2/2}\,dx=\sqrt{2\pi}$, is
the Gaussian integral computed in :numref:`sec_mdl-integral_calculus` by squaring
the integral and passing to polar coordinates (:eqref:`eq_mdl-gauss-square`).
Substituting $x\mapsto(x-\mu)/\sigma$ then shows :eqref:`eq_mdl-gaussian_pdf`
integrates to one. The parameters are the first two moments, and one integration
by parts proves it: the mean is $\mu$ by symmetry, and for the variance take
$\mu=0$, $\sigma=1$ and integrate by parts with $u=x$, $dv=x\,e^{-x^2/2}\,dx$,

$$
\int_{-\infty}^{\infty} x^2\,\frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx
= \frac{1}{\sqrt{2\pi}}\Bigl(\bigl[-x\,e^{-x^2/2}\bigr]_{-\infty}^{\infty}
+ \int_{-\infty}^{\infty} e^{-x^2/2}\,dx\Bigr) = 0 + 1 = 1,
$$

so the standard Gaussian has variance $1$; shifting by $\mu$ and scaling by
$\sigma$ gives mean $\mu$ and variance $\sigma^2$ in general. (The exponential
family's moment property, :eqref:`eq_mdl-exp_family_moment`, will recover both by
differentiation alone.)

**Maximum entropy.** Among all distributions with a given mean and variance, the
Gaussian has the largest entropy: it is the *most noncommittal* choice consistent
with knowing only those two numbers :cite:`Cover.Thomas.1999`. We prove this in
:numref:`subsec_mdl-gaussian-max-entropy` as a short consequence of Gibbs'
inequality, and place it in context when we meet the
exponential family below.

**Where it arises.** As the CLT limit it models aggregate noise; it is the noise
model behind *regression*, where taking $y\sim\mathcal N(\hat y,\sigma^2)$ makes
the negative log-likelihood equal to mean squared error
(:numref:`subsec_mdl-gaussian-mse`); and it is the default prior
and latent distribution throughout deep generative modeling.

```{.python .input #distributions-gaussian}
import numpy as onp
rng = onp.random.default_rng(0)
mu, sigma = 0.0, 1.0
x = onp.arange(-4, 4, 0.01)
p = onp.exp(-(x - mu)**2 / (2 * sigma**2)) / onp.sqrt(2 * onp.pi * sigma**2)
print('total mass (≈1):', float((0.01 * p).sum().round(4)))
rng.normal(mu, sigma, size=(3, 3))              # sample from N(mu, sigma^2)
```

### Laplace

The *Laplace* distribution is a Gaussian's sharper sibling: symmetric about a
location $\mu$, but with *exponential* decay on each side instead of Gaussian,

$$
p(x) = \frac{1}{2b}\,e^{-|x-\mu|/b},
\qquad X\sim\mathrm{Laplace}(\mu,b).
$$
:eqlabel:`eq_mdl-laplace_pdf`

The two-sided exponential gives it a sharp peak at $\mu$ and *heavier tails* than
the Gaussian, visible in :numref:`fig_mdl-prob-continuous-pdfs`.

**Where it arises.** It is the probabilistic shadow of the $L_1$ loss, just as the
Gaussian is of MSE: the negative log-likelihood of a Laplace is $\propto|y-\hat y|$,
the *mean absolute error*. As a *prior* on weights it produces the
sparsity-inducing $L_1$ regularizer $\propto\|\mathbf w\|_1$, the engine of LASSO
:cite:`Tibshirani.1996`.

**Mean and variance.** By symmetry $\mu_X=\mu$. For the variance, center at $\mu$
and use $\int_0^\infty u^2 e^{-u}\,du=2$ with $u=|x-\mu|/b$:

$$
\sigma_X^2 = \int_{-\infty}^\infty \frac{(x-\mu)^2}{2b}e^{-|x-\mu|/b}\,dx
= \frac{2}{2b}\int_0^\infty x^2 e^{-x/b}\,dx
= \frac{1}{b}\cdot b^3\!\int_0^\infty u^2 e^{-u}\,du
= 2b^2 .
$$

The *location estimator* makes the contrast concrete: the maximum-likelihood
$\hat\mu$ under a Laplace minimizes $\sum_i|x_i-\mu|$, whose minimizer is the sample
*median* (robust to outliers), whereas the Gaussian's is the mean. The cell
matches a Laplace and a Gaussian on variance and confirms the Laplace's heavier
tail by the fraction of mass beyond $3$ standard deviations.

```{.python .input #distributions-laplace}
import numpy as onp
rng = onp.random.default_rng(0)
mu, b = 0.0, 1.0
sigma = onp.sqrt(2) * b                            # matched-variance Gaussian sd
U = (rng.random(200000) - 0.5) * (1 - 1e-7)       # open interval avoids log(0)
lap = mu - b * onp.sign(U) * onp.log(1 - 2 * onp.abs(U))  # inverse transform
gau = rng.normal(mu, sigma, 200000)
print('Laplace var (2b^2):', float(lap.var().round(3)), ' Gaussian var:', float(gau.var().round(3)))
print('P(|x| > 3sd): Laplace', float((onp.abs(lap) > 3 * sigma).mean()),
      ' Gaussian', float((onp.abs(gau) > 3 * sigma).mean()))
```

### Multivariate Gaussian

Almost every latent-variable and noise model in deep learning uses a Gaussian over
a *vector*. The *multivariate Gaussian* $\mathcal N(\boldsymbol\mu,
\boldsymbol\Sigma)$ on $\mathbb R^d$ has a mean vector $\boldsymbol\mu$ and a
symmetric positive-definite *covariance matrix* $\boldsymbol\Sigma$, with density

$$
p(\mathbf x) = \frac{1}{(2\pi)^{d/2}|\boldsymbol\Sigma|^{1/2}}
\exp\!\Bigl(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Bigr).
$$
:eqlabel:`eq_mdl-mvn_pdf`

**The geometry is the eigendecomposition.** The exponent is a quadratic form in
$\boldsymbol\Sigma^{-1}$, so its level sets (the contours of equal density) are
*ellipsoids*. The covariance $\boldsymbol\Sigma$ is symmetric positive definite, so
it has an orthonormal eigenbasis
$\boldsymbol\Sigma=\mathbf W\boldsymbol\Lambda\mathbf W^\top$ with all $\lambda_i>0$
by the spectral theorem (:numref:`subsec_mdl-spectral-theorem`), and the contour
ellipsoid's *axes point along the eigenvectors $\mathbf w_i$ with half-lengths
proportional to $\sqrt{\lambda_i}$*: the principal directions of spread are the
eigenvectors, stretched by the standard deviations along them. The isotropic case
$\boldsymbol\Sigma=\sigma^2\mathbf I$ gives spherical contours and factorizes into
$d$ independent one-dimensional Gaussians. :numref:`fig_mdl-prob-mvn-contours` draws
this for a $2\times2$ covariance.

![Samples from a $2$-D Gaussian (blue points) with the elliptical density contours overlaid. The ellipse axes lie along the eigenvectors of the covariance $\boldsymbol\Sigma$ with half-lengths proportional to $\sqrt{\lambda_i}$ (the standard deviation along each principal direction), connecting the multivariate Gaussian directly to the spectral theorem.](../img/mdl-prob-mvn-contours.svg)
:label:`fig_mdl-prob-mvn-contours`

It is the default vector noise and prior because it is *closed under linear maps and
conditioning*. A linear transform $\mathbf A\mathbf x+\mathbf b$ of a Gaussian is
again Gaussian, with covariance $\mathbf A\boldsymbol\Sigma\mathbf A^\top$; marginals
(a special case, projecting onto a coordinate block) are therefore Gaussian too.
*Conditioning* is closed in the same way. Partition the vector and its parameters
into blocks $\mathbf x=(\mathbf x_1,\mathbf x_2)$,

$$
\boldsymbol\mu = \begin{pmatrix}\boldsymbol\mu_1\\\boldsymbol\mu_2\end{pmatrix},
\qquad
\boldsymbol\Sigma = \begin{pmatrix}\boldsymbol\Sigma_{11} & \boldsymbol\Sigma_{12}\\
\boldsymbol\Sigma_{21} & \boldsymbol\Sigma_{22}\end{pmatrix},
$$

and the conditional $\mathbf x_1\mid\mathbf x_2$ is again Gaussian. We state its
mean and covariance as a given fact (the derivation is a block-matrix computation,
carried out in :cite:`Bishop.2006`, §2.3; Exercise 10 puts the formulas to work):

$$
\boldsymbol\mu_{1\mid 2} = \boldsymbol\mu_1 + \boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}(\mathbf x_2-\boldsymbol\mu_2),
\qquad
\boldsymbol\Sigma_{1\mid 2} = \boldsymbol\Sigma_{11} - \boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.
$$
:eqlabel:`eq_mdl-mvn_conditional`

The conditional mean is a *linear* function of the observed block (a linear
regression of $\mathbf x_1$ on $\mathbf x_2$ read straight off the covariance),
and the conditional covariance
$\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}$
is the *Schur complement* of $\boldsymbol\Sigma_{22}$, the standard name for the
block that remains when the variables in $\mathbf x_2$ are eliminated; note that
it does not depend on the observed value $\mathbf x_2$. This pair is the engine
of Gaussian process regression (:numref:`chap_gp`): condition the joint Gaussian
prior over function values on the observed points and read off the predictive
mean and variance.

**Sampling via Cholesky.** How do we draw from
$\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ given only a standard-normal
generator? This is the application promised when the Cholesky factorization was
introduced (:numref:`subsec_mdl-psd`): factor
$\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ with $\mathbf L$ lower triangular,
draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ ($d$ independent standard
normals), and set $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$. Then $\mathbf x$
is Gaussian (a linear map of a Gaussian), its mean is $\boldsymbol\mu$ because
$\mathbb E[\mathbf z]=\mathbf 0$, and one line verifies the covariance:

$$
\mathbb E\bigl[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top\bigr]
= \mathbf L\,\mathbb E[\mathbf z\mathbf z^\top]\,\mathbf L^\top
= \mathbf L\,\mathbf I\,\mathbf L^\top
= \boldsymbol\Sigma .
$$

This recipe is what the library samplers run: $d$ standard normals plus one
triangular matrix multiply. The cell draws samples through the library sampler,
estimates the covariance, checks that its eigenvalues recover the $\{1,3\}$
built into $\boldsymbol\Sigma$, then re-samples by the Cholesky recipe and
confirms that it reproduces $\boldsymbol\Sigma$ as well.

```{.python .input #distributions-mvn}
import numpy as onp
rng = onp.random.default_rng(0)
mu_v = onp.array([0., 0.])
Sigma = onp.array([[2., 1.], [1., 2.]])
sample = rng.multivariate_normal(mu_v, Sigma, size=5000)
emp = onp.cov(sample.T)
vals, vecs = onp.linalg.eigh(emp)
print('empirical covariance:\n', emp.round(2))
print('eigenvalues (≈ 1, 3):', vals.round(2))   # Sigma has eigenvalues 1 and 3
L = onp.linalg.cholesky(Sigma)                   # Sigma = L L^T
chol = mu_v + rng.standard_normal((5000, 2)) @ L.T  # x = mu + L z
print('Cholesky-recipe covariance:\n', onp.cov(chol.T).round(2))
```

**Probability in high dimension.** One more Gaussian fact matters enormously in
deep learning and defies low-dimensional intuition. For
$\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$ the squared norm
$\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ is a sum of $d$ independent terms with mean
$1$, so $\mathbb E\,\|\mathbf x\|^2=d$, and, by the same averaging that makes
sample means reliable, the sum concentrates near $d$: virtually all the mass
of a high-dimensional Gaussian lives in a *thin shell* of radius
$\approx\sqrt d$, far from the origin where the density is pointwise largest.
Likewise two independent draws $\mathbf x,\mathbf y$ are *nearly orthogonal*:
their cosine has mean $0$ and standard deviation $\approx1/\sqrt d$, a
probabilistic version of the concentration-of-angles proposition of
:numref:`sec_mdl-geometry-linear-algebraic-ops`
:cite:`Vershynin.2018`; both facts get their full quantitative treatment, with
exponential tail bounds, in :numref:`sec_mdl-concentration-generalization`.
Both matter in practice:
weight-initialization schemes scale by $1/\sqrt{d}$ precisely so that
$\|\mathbf W\mathbf x\|$ stays comparable to $\|\mathbf x\|$ layer after layer;
cosine similarity between embeddings is informative because unrelated
high-dimensional vectors sit near cosine $0$ by default; and nearest-neighbor
search loses contrast because the distances between random points all
concentrate around one common value. The check below watches both
concentrations set in as $d$ grows.

```{.python .input #mdl-distributions-multivariate-gaussian-1}
rng = np.random.default_rng(0)
for d in (3, 30, 300):
    x, y = rng.standard_normal((2, 1000, d))
    nx = np.linalg.norm(x, axis=1)
    cos = (x * y).sum(1) / (nx * np.linalg.norm(y, axis=1))
    print(f'd = {d:3d}:  ||x||/sqrt(d) = {(nx / np.sqrt(d)).mean():.3f}'
          f' ± {(nx / np.sqrt(d)).std():.3f},   mean |cos(x,y)| = {np.abs(cos).mean():.3f}')
```

The norm ratio locks onto $1$ with a spread shrinking like $1/\sqrt{2d}$, and
the typical cosine collapses toward $0$ at the promised $1/\sqrt d$ rate: in
high dimension, Gaussian geometry is a thin shell of mutually near-orthogonal
directions.

## The Exponential Family

Nearly every distribution above (Bernoulli, categorical, binomial, Poisson,
exponential, Gaussian, Laplace with fixed location) is a special case of one
form, the *exponential family*. A distribution
belongs to it if its density (or mass) can be written

$$
p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).
$$
:eqlabel:`eq_mdl-exp_family`

The four ingredients each play a clear role. The *base measure* $h(\mathbf x)$ is a
fixed background that does not depend on the parameters. The *natural parameters*
$\boldsymbol\eta$ are the knobs. The *sufficient statistics* $T(\mathbf x)$ are the
only features of the data that matter: given $T(\mathbf x)$, no other detail of
$\mathbf x$ affects the probability, which is exactly what "sufficient" means. The
*log-partition function* (or cumulant function)

$$
A(\boldsymbol\eta) = \log\!\int h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x)\bigr)\,d\mathbf x
$$
:eqlabel:`eq_mdl-log_partition`

is whatever it takes to make :eqref:`eq_mdl-exp_family` integrate to one.

### Recognizing Old Friends

The form looks abstract until we match the pieces. The **Bernoulli**
$p^x(1-p)^{1-x}$ rewrites as

$$
\exp\!\Bigl(x\log\tfrac{p}{1-p} + \log(1-p)\Bigr),
$$

so $T(x)=x$, the natural parameter $\eta=\log\tfrac{p}{1-p}$ is the *logit*, and
$A(\eta)=\log(1+e^\eta)$ is the *softplus*; the link between probabilities and
logits is built into the family. The **Poisson** $\tfrac{\lambda^k e^{-\lambda}}{k!}$
has base measure $h(k)=1/k!$, statistic $T(k)=k$, parameter $\eta=\log\lambda$, and
$A(\eta)=e^\eta=\lambda$. The **Gaussian** with unknown mean and variance is a
two-parameter member: completing the exponent,

$$
\frac{1}{\sqrt{2\pi}}\exp\!\Bigl(\underbrace{\tfrac{\mu}{\sigma^2}}_{\eta_1}\,x
+ \underbrace{\bigl(-\tfrac{1}{2\sigma^2}\bigr)}_{\eta_2}\,x^2
- \underbrace{\bigl(\tfrac{\mu^2}{2\sigma^2}+\log\sigma\bigr)}_{A(\boldsymbol\eta)}\Bigr),
$$

so $T(x)=(x,x^2)^\top$ with base measure $h(x)=1/\sqrt{2\pi}$. The second natural
parameter is negative, $\eta_2=-1/(2\sigma^2)<0$, which is what forces the
tail-suppressing $e^{-x^2}$ decay. The exact split between $h$, $\eta$, and $T$ is
not unique; what matters is that the distribution *fits the form at all*.

Membership has edges. The conspicuous absentees from the list above
are the two *uniforms*: when the endpoints are unknown parameters, the support of
$U(a,b)$ (and likewise of $U(n)$) moves with those parameters, whereas every member
of :eqref:`eq_mdl-exp_family` has the fixed, parameter-free support of its base
measure $h$. This is the classic counterexample. Heavy-tailed laws live outside
the family for a different reason. The *Student-$t$* takes two lines to define:
draw a random precision $1/\sigma^2$ from a Gamma distribution (a two-parameter
density on the positive half-line, defined in full among the conjugate priors
below), then draw a Gaussian with that precision. Averaging the Gaussian over its
random scale fattens the $e^{-x^2}$ tails into polynomial ones, and the Cauchy of
:numref:`sec_mdl-random_variables` is its one-degree-of-freedom extreme. The
support here is all of $\mathbb R$, fixed, yet the mixing destroys the exponential
form: no fixed finite-dimensional sufficient statistic exists, the situation
described by the *Pitman--Koopman--Darmois* theorem :cite:`Bishop.2006`. The
exclusion costs two conveniences: these laws admit no finite-dimensional
conjugate prior, and their negative log-likelihoods are not convex in the
parameters. Both conveniences are exponential-family privileges, as the rest of
this section shows.

### Where the Form Comes From: Maximum Entropy

The exponential form is exactly what *maximizing
entropy* produces. The two maximum-entropy facts cited earlier
:cite:`Cover.Thomas.1999` are the same
statement seen twice: the discrete uniform is the maximum-entropy distribution on a
finite set when we fix *nothing* but the support, and the Gaussian is the
maximum-entropy distribution on the line when we fix the *mean and variance*. In
general, maximizing the entropy *relative to the base measure* $h$,
$H_h[p]=-\int p\log\tfrac{p}{h}$, subject to fixing the expected
sufficient statistics $\mathbb E[T(\mathbf x)]=\boldsymbol\tau$ has, by a Lagrange
multiplier argument (:numref:`subsec_mdl-lagrange-multipliers`), the solution

$$
p(\mathbf x) \propto h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x)\bigr),
$$

which is precisely :eqref:`eq_mdl-exp_family`: the multipliers $\boldsymbol\eta$
are the natural parameters and $A(\boldsymbol\eta)$ is again the normalizer
:cite:`Murphy.2022,Wainwright.Jordan.2008`. One subtlety: the base measure does
real work here. With plain Shannon entropy, i.e. $h\equiv1$, fixing the mean on
$\{0,1,2,\dots\}$ yields the *geometric* distribution; only entropy relative to
$h(k)=1/k!$ yields the Poisson. For the uniform and the Gaussian above, $h$ is
constant, so plain entropy was the right notion all along. So the exponential
family is the *least-committal* family
consistent with knowing a fixed set of averages: the uniform fixes none, the
Bernoulli/categorical fix the class probabilities, and the Gaussian fixes the first
two moments.

### The Moment Property

The single most useful fact about :eqref:`eq_mdl-exp_family` is that $A$ is a moment
generator: *differentiating the log-partition function returns the mean of the
sufficient statistics.*

**Proposition (mean of the sufficient statistics).**
*For an exponential-family distribution :eqref:`eq_mdl-exp_family`,*

$$
\nabla_{\boldsymbol\eta} A(\boldsymbol\eta) = \mathbb E[T(\mathbf x)].
$$
:eqlabel:`eq_mdl-exp_family_moment`

**Proof.** Differentiate :eqref:`eq_mdl-log_partition` under the integral sign;
we grant the interchange of derivative and integral, which is valid on the
interior of the natural-parameter domain :cite:`Folland.1999`.
Writing $Z(\boldsymbol\eta)=e^{A(\boldsymbol\eta)}=\int h(\mathbf x)e^{\boldsymbol\eta^\top T(\mathbf x)}d\mathbf x$,
the derivative of the log is $\nabla A = (\nabla Z)/Z$, and

$$
\nabla_{\boldsymbol\eta} Z
= \int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x .
$$

Dividing by $Z=e^{A}$ pulls the $e^{-A}$ back inside, reconstituting the density
:eqref:`eq_mdl-exp_family`:

$$
\nabla_{\boldsymbol\eta} A
= \frac{1}{Z}\int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x
= \int T(\mathbf x)\,\underbrace{h(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)}}_{p(\mathbf x\mid\boldsymbol\eta)}\,d\mathbf x
= \mathbb E[T(\mathbf x)] . \quad\blacksquare
$$

Differentiating once more (the same interchange) turns $\nabla A=(\nabla Z)/Z$
into

$$
\nabla^2 A
= \frac{\nabla^2 Z}{Z} - \frac{(\nabla Z)(\nabla Z)^\top}{Z^2}
= \mathbb E[T\,T^\top] - \mathbb E[T]\,\mathbb E[T]^\top
= \operatorname{Cov}(T),
$$

so $A$ is *convex* (a covariance matrix is PSD, :numref:`subsec_mdl-psd`).
Convexity pays off in fitting. For a sample $x_1,\dots,x_n$ the negative
log-likelihood is

$$
-\sum_{i=1}^{n} \log p(x_i\mid\boldsymbol\eta)
= n\,A(\boldsymbol\eta) - \boldsymbol\eta^\top \sum_{i=1}^{n} T(x_i)
- \sum_{i=1}^{n} \log h(x_i),
$$

convex in $\boldsymbol\eta$ because $A$ is. Setting its gradient to zero gives
$n\nabla A(\boldsymbol\eta)=\sum_i T(x_i)$: by the moment property, maximum
likelihood *matches the model's expected sufficient statistics to their empirical
averages*, $\mathbb E[T]=\bar T$. Uniqueness of the optimum needs one more
hypothesis, strict convexity: $\operatorname{Cov}(T)\succ0$, which holds when the
representation is *minimal* (no linear combination of the sufficient statistics
is constant). The section's own softmax parameterization :eqref:`eq_mdl-softmax`
is *not* minimal: adding the same constant to every logit leaves every $p_k$
unchanged, so the categorical negative log-likelihood is flat along the direction
$\mathbf 1$ and its optima form a line rather than a point, which is why
implementations pin the shift (fixing one logit at zero, or subtracting the
maximum). We verify :eqref:`eq_mdl-exp_family_moment` for the
Bernoulli, where it should read $A'(\eta)=\sigma(\eta)=\mathbb E[x]=p$.

```{.python .input #distributions-exp-family}
import numpy as onp
eta = 0.7                                        # natural parameter (logit)
A = lambda e: onp.log1p(onp.exp(e))                # Bernoulli log-partition (softplus)
eps = 1e-6
dA = (A(eta + eps) - A(eta - eps)) / (2 * eps)   # numerical dA/deta
print('dA/deta      =', round(float(dA), 6))
print('sigmoid(eta) = E[x] = p =', round(float(1 / (1 + onp.exp(-eta))), 6))
```

The derivative of the softplus is the sigmoid, and it lands exactly on the
Bernoulli mean $p$. This is no coincidence: the sigmoid and softmax links that
deep-learning classifiers use are *precisely* the maps from natural parameters to
means in the Bernoulli and categorical families, and the convexity of $A$ is why
their negative log-likelihoods are convex in the natural parameters. If those
parameters are affine functions of fixed features, this gives a convex
optimization problem. A neural network makes the natural parameters nonlinear
functions of its weights, so the same loss is not generally convex in the
weights. The exponential family still supplies the likelihood and the link
functions used by standard deep-learning losses
:cite:`Bishop.2006,Koller.Friedman.2009`.

## Conjugate Priors

The family tree so far has one half missing. The arrows of
:numref:`fig_mdl-prob-family-tree` *build* distributions over data; but, as
:numref:`subsec_mdl-map` develops, fitting a model can also mean placing a *prior*
over its parameters, and the parameters of the distributions above are themselves
quantities to be inferred: the Bernoulli's $p$, the Poisson's $\lambda$, the
categorical's $\mathbf p$. A *prior* over those parameters is again a distribution,
and three new continuous laws, the **Beta**, the **Gamma**, and the
**Dirichlet**, are exactly the priors that pair with the discrete
distributions we have met. A prior is **conjugate** to a likelihood when the
posterior belongs to the *same family* as the prior, so Bayesian updating just moves
the parameters rather than changing the shape of the distribution. This closes the
tree: a tier of prior nodes flanks the data distributions in
:numref:`fig_mdl-prob-family-tree`, joined to them by "conjugate prior" links.

### Beta--Bernoulli: Counting with Pseudo-Counts

The **Beta** distribution is a density on the unit interval $p\in[0,1]$ (exactly
the range of a probability) with two shape parameters $\alpha,\beta>0$,

$$
\mathrm{Beta}(p\mid\alpha,\beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)},
$$
:eqlabel:`eq_mdl-beta_pdf`

where the normalizing constant is
$B(\alpha,\beta)=\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$, with $\Gamma$
the Gamma function that the exercises of :numref:`sec_mdl-integral_calculus`
introduced (recall $\Gamma(z+1)=z\,\Gamma(z)$ and $\Gamma(n+1)=n!$). The mean
takes one line from this normalizer:

$$
\mathbb E[p]
= \frac{1}{B(\alpha,\beta)}\int_0^1 p^{\alpha}(1-p)^{\beta-1}\,dp
= \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)}
= \frac{\alpha}{\alpha+\beta},
$$

using $\Gamma(\alpha+1)=\alpha\,\Gamma(\alpha)$ to simplify the ratio; the same
move one step further gives the variance
$\operatorname{Var}(p)=\alpha\beta/\bigl((\alpha+\beta)^2(\alpha+\beta+1)\bigr)$.
The uniform prior on $[0,1]$ that :numref:`sec_mdl-maximum_likelihood` will place
on the coin's bias is the special case $\alpha=\beta=1$.

Now take a Beta prior on a coin's bias $p$ and observe a Bernoulli/binomial sample
with $x$ heads in $n$ flips. Multiplying the prior by the likelihood
:eqref:`eq_mdl-binomial_pmf`,

$$
\underbrace{p^{\alpha-1}(1-p)^{\beta-1}}_{\text{prior}}\cdot
\underbrace{p^{x}(1-p)^{n-x}}_{\text{likelihood}}
= p^{(\alpha+x)-1}(1-p)^{(\beta+n-x)-1},
$$

which, up to normalization, is again a Beta, now with parameters shifted by the
data:

$$
p \mid x \;\sim\; \mathrm{Beta}\bigl(\alpha+x,\ \beta+(n-x)\bigr).
$$
:eqlabel:`eq_mdl-beta_posterior`

This is the **pseudo-count** picture:
$\alpha$ and $\beta$ act as *phantom* heads and tails seen
before any real data, and observing $x$ real heads and $n-x$ real tails simply adds
the real counts to the phantom ones. The posterior mean

$$
\mathbb E[p\mid x] = \frac{\alpha+x}{\alpha+\beta+n}
$$

is literally the heads-frequency with the phantom flips included; it interpolates
between the prior mean $\alpha/(\alpha+\beta)$ and the maximum-likelihood frequency
$x/n$, and slides toward the data as $n$ grows, the same flattening of the prior
that turns MAP into MLE in :numref:`subsec_mdl-map`. With the uniform prior
$\alpha=\beta=1$ it becomes *Laplace's rule of succession* $(x+1)/(n+2)$
:cite:`Laplace.1814`: estimate a
probability by adding one phantom observation of each outcome. The cell verifies
the update on a small sample.

```{.python .input #distributions-conjugate}
alpha, beta = 2.0, 2.0                           # Beta prior pseudo-counts
x, n = 7, 10                                     # 7 heads in 10 flips
post_alpha, post_beta = alpha + x, beta + (n - x)  # Beta posterior update
print('prior mean      =', round(alpha / (alpha + beta), 4))
print('MLE  x/n        =', round(x / n, 4))
print('posterior Beta  =', (post_alpha, post_beta))
print('posterior mean  =', round(post_alpha / (post_alpha + post_beta), 4))
```

The posterior mean sits between the prior's $0.5$ and the data's $0.7$, exactly as
the pseudo-count sum predicts. :numref:`fig_mdl-prob-beta-posterior` shows the
same mechanism at work over a longer run of data: the flat $\mathrm{Beta}(1,1)$
prior sharpens into a posterior that piles up on the true bias, its width
shrinking like $1/\sqrt n$ as the real counts swamp the phantom ones.

![Bayesian updating as sharpening. Starting from the flat $\mathrm{Beta}(1,1)$ prior (left), observing $9$ heads in $13$ flips of a coin with true bias $\theta^\ast=0.7$ (dashed line) gives the $\mathrm{Beta}(10,5)$ posterior (middle); after $130$ flips the $\mathrm{Beta}(91,41)$ posterior has piled up tightly around $\theta^\ast$ (right). Each update just adds the observed heads and tails to the pseudo-counts, and the posterior standard deviation shrinks like $1/\sqrt{n}$.](../img/mdl-prob-beta-posterior.svg)
:label:`fig_mdl-prob-beta-posterior`

### The Rest of the Tier, and the General Fact

The other discrete laws have their own conjugate partners, built the same way.

* **Gamma--Poisson.** The **Gamma** distribution
  $\mathrm{Gamma}(\lambda\mid\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}\,\lambda^{\alpha-1}e^{-\beta\lambda}$
  is a density on the rate $\lambda>0$ with mean $\alpha/\beta$ and variance
  $\alpha/\beta^2$. It is conjugate to the Poisson by the same one-line move as
  the Beta: $n$ observed counts $x_1,\dots,x_n$ contribute the likelihood
  $\propto\lambda^{\sum_i x_i}e^{-n\lambda}$, and multiplying by the prior
  collects powers into $\lambda^{\alpha+\sum_i x_i-1}e^{-(\beta+n)\lambda}$, the
  posterior $\mathrm{Gamma}\bigl(\alpha+\sum_i x_i,\ \beta+n\bigr)$: pseudo-events
  over a pseudo-window. Averaging the Poisson over this Gamma uncertainty in its
  rate yields the *negative binomial*, the over-dispersed fix promised in the
  Poisson section.
* **Gamma as a waiting time.** The Gamma is more than a prior: with integer shape
  $\alpha=k$, $\mathrm{Gamma}(k,\beta)$ is the waiting time to the *$k$-th* event
  of the Poisson process we named at the bus stop, the sum of $k$ independent
  $\mathrm{Exp}(\beta)$ waits, extending the exponential's arrow in the family
  tree just as the binomial extends the Bernoulli's.
* **Dirichlet--Multinomial.** The **Dirichlet** distribution
  $\mathrm{Dir}(\mathbf p\mid\boldsymbol\alpha)=\frac{\Gamma(\sum_k\alpha_k)}{\prod_k\Gamma(\alpha_k)}\prod_k p_k^{\alpha_k-1}$
  is the multivariate Beta, a density over probability vectors on the simplex,
  with mean $\mathbb E[p_k]=\alpha_k/\sum_j\alpha_j$. It is conjugate to the
  categorical/multinomial by the same multiplication: the likelihood contributes
  $\prod_k p_k^{x_k}$, so observing class counts $\mathbf x$ updates the prior to
  $\mathrm{Dir}(\boldsymbol\alpha+\mathbf x)$, one pseudo-count per class.

These are three instances of one theorem. **Every exponential-family likelihood
has a conjugate prior**, provided the hyperparameters are chosen so that the
prior below is normalizable, and it is itself an exponential family in the
natural parameters $\boldsymbol\eta$: writing the prior as
$p(\boldsymbol\eta)\propto\exp\!\bigl(\boldsymbol\nu^\top\boldsymbol\eta - \kappa\,A(\boldsymbol\eta)\bigr)$,
multiplying by the likelihood :eqref:`eq_mdl-exp_family` just adds the data's
sufficient statistics into $\boldsymbol\nu$ and increments $\kappa$ by the sample
size. The pseudo-count update of :eqref:`eq_mdl-beta_posterior` is this general
mechanism specialized to the Bernoulli, and the hyperparameters
$(\boldsymbol\nu,\kappa)$ are *pseudo-data*: a prior sufficient statistic and a prior
sample size :cite:`Bishop.2006,Murphy.2022`. Conjugacy makes the exponential
family the natural home of tractable Bayesian inference as well as of the
standard losses.

## Summary

The eleven data distributions of this section in one view (their three conjugate
priors, the Beta, Gamma, and Dirichlet, close the family). For the categorical
and multinomial rows we code a draw as a one-hot vector, respectively as the sum
of $n$ one-hot vectors: the coding the cross-entropy loss already uses, and the
object whose mean and covariance the table states.

| Law | pmf or pdf | Mean | Variance | Where it shows up in DL |
|:--|:--|:--|:--|:--|
| $\mathrm{Bernoulli}(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | binary classifier output, BCE loss |
| $\mathrm{Categorical}(\mathbf p)$ | $p_k$ | $\mathbf p$ | $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ | softmax output, cross-entropy loss |
| $\mathrm{Multinomial}(n,\mathbf p)$ | :eqref:`eq_mdl-multinomial_pmf` | $n\mathbf p$ | $n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$ | class counts over $n$ draws |
| Uniform $U(n)$ | $1/n$ | $(n+1)/2$ | $(n^2-1)/12$ | index sampling, shuffling |
| $\mathrm{Binomial}(n,p)$ | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | success counts over $n$ trials |
| $\mathrm{Poisson}(\lambda)$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | rare-event counts |
| Uniform $U(a,b)$ | $1/(b-a)$ on $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ | raw randomness, initialization |
| $\mathrm{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$, $x\ge0$ | $1/\lambda$ | $1/\lambda^2$ | waiting times, survival |
| $\mathcal N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | MSE noise model, CLT, priors |
| $\mathrm{Laplace}(\mu,b)$ | $\frac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | MAE loss, $L_1$ sparsity prior |
| $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ | :eqref:`eq_mdl-mvn_pdf` | $\boldsymbol\mu$ | $\boldsymbol\Sigma$ | latent priors, Gaussian processes |

* Distributions form a *family*: Bernoulli is the seed; summing $n$
  gives the **Binomial** ($\mu=np$, $\sigma^2=np(1-p)$, derived as a sum of
  Bernoullis); the many-rare limit gives the **Poisson** ($\mu=\sigma^2=\lambda$,
  derived from the binomial); the many-ordinary limit (CLT) gives the **Gaussian**.
* The **Categorical** generalizes Bernoulli to $K$ outcomes; its softmax form is
  every classifier's output layer and its NLL is the cross-entropy loss. The
  **Multinomial** counts $n$ categorical draws.
* The **continuous** laws: the **uniform** (raw randomness, $U(0,1)$ powers
  sampling), the **exponential** (memoryless waiting times, partner of the
  Poisson), the **Gaussian** (CLT limit, MSE noise model, max-entropy default), the
  **Laplace** ($L_1$/MAE loss, sparsity prior), and the **multivariate Gaussian**,
  whose elliptical contours are the eigendecomposition of its covariance, which is
  closed under both linear maps and conditioning (the Schur-complement formulas,
  :eqref:`eq_mdl-mvn_conditional`, that drive Gaussian-process regression), and
  which is sampled by the Cholesky recipe
  $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$.
* Means and variances follow from structure (linearity of expectation,
  limits, and standard integrals) rather than memorization.
* The **exponential family** $p(\mathbf x)=h(\mathbf x)\exp(\boldsymbol\eta^\top
  T(\mathbf x)-A(\boldsymbol\eta))$ unifies almost all of them (the uniforms, whose
  support moves with their parameters, stay outside); it is exactly the
  maximum-entropy family (entropy taken relative to the base measure $h$) for a
  fixed set of expected sufficient statistics. Its
  log-partition $A$ generates moments: $\nabla A(\boldsymbol\eta)=\mathbb E[T(\mathbf x)]$,
  and $A$ is convex. Thus its negative log-likelihood is convex in the natural
  parameters; with an affine parameter map this yields a convex generalized
  linear model, whereas a neural-network parameterization remains nonconvex.
* Every exponential-family likelihood has a **conjugate prior**: the **Beta** for the
  Bernoulli/binomial, the **Gamma** for the Poisson, the **Dirichlet** for the
  categorical/multinomial. Bayesian updating then just adds the data's sufficient
  statistics to the prior's *pseudo-counts*: e.g. a $\mathrm{Beta}(\alpha,\beta)$
  prior and $x$ heads in $n$ flips give a $\mathrm{Beta}(\alpha+x,\beta+n-x)$
  posterior.

## Exercises

1. Let $X,Y\sim\mathrm{Binomial}(16,1/2)$ be independent. Find the standard
   deviation of $X-Y$. (*Hint:* variances add for independent variables, even under
   subtraction.)
2. Show that the categorical negative log-likelihood $-\sum_k y_k\log p_k$ with
   $p_k$ from the softmax :eqref:`eq_mdl-softmax` has gradient
   $\hat{\mathbf p}-\mathbf y$ with respect to the logits $\mathbf z$. Why does this
   form make softmax classification well behaved?
3. Verify that the multinomial pmf :eqref:`eq_mdl-multinomial_pmf` for $n$ trials,
   $K=2$, reduces exactly to the binomial pmf :eqref:`eq_mdl-binomial_pmf`.
4. Prove the converse of memorylessness: if a positive continuous random variable
   has survival function $G(t)=P(X>t)$ satisfying $G(s+t)=G(s)\,G(t)$ for all
   $s,t\ge0$, show that $G(t)=e^{-\lambda t}$ for some $\lambda>0$ (use
   monotonicity of $G$). Then show that the minimum of independent
   $\mathrm{Exp}(\lambda_i)$ is again exponential with rate $\sum_i\lambda_i$.
5. For $X\sim\mathrm{Poisson}(\lambda)$, argue that $(X-\lambda)/\sqrt\lambda$
   becomes approximately Gaussian as $\lambda\to\infty$. (*Hint:* for integer
   $\lambda=m$, a $\mathrm{Poisson}(m)$ variable is the sum of $m$ independent
   $\mathrm{Poisson}(1)$ variables, so the iid CLT applies as $m\to\infty$;
   general $\lambda$ follows by splitting off the fractional part.)
6. Show that the maximum-likelihood location $\mu$ under a $\mathrm{Laplace}(\mu,b)$
   model minimizes $\sum_i|x_i-\mu|$ and hence is the sample *median*, while under a
   Gaussian it is the mean.
7. Write the **categorical** distribution over $K$ classes in exponential-family
   form :eqref:`eq_mdl-exp_family`. Identify $T(\mathbf x)$ (one-hot) and the natural
   parameters, and verify that $\nabla A(\boldsymbol\eta)=\mathbb E[T]$ recovers the
   softmax :eqref:`eq_mdl-softmax`.
8. Given $\boldsymbol\Sigma=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$,
   find the eigenvalues and eigenvectors and state the axis directions and
   half-lengths of the $\mathcal N(\mathbf 0,\boldsymbol\Sigma)$ density contours.
   Show that a linear map $\mathbf A\mathbf x+\mathbf b$ of a multivariate Gaussian
   is again Gaussian with covariance $\mathbf A\boldsymbol\Sigma\mathbf A^\top$.
9. Start from a uniform $\mathrm{Beta}(1,1)$ prior on a coin's bias and observe the
   sequence "HHHTHTTHHHHHT" ($9$ heads, $4$ tails) of
   :numref:`sec_mdl-maximum_likelihood`. Give the posterior :eqref:`eq_mdl-beta_posterior`
   and its mean, and show that as the data grows the posterior mean converges to the
   maximum-likelihood frequency $n_H/(n_H+n_T)$.
10. Using the conditional formula :eqref:`eq_mdl-mvn_conditional`, take the
    bivariate Gaussian with $\boldsymbol\mu=\mathbf 0$ and
    $\boldsymbol\Sigma=\left(\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right)$
    and find the distribution of $x_1$ given $x_2$. Show its mean is $\rho x_2$ and
    its variance is $1-\rho^2$, and explain why observing $x_2$ never *increases* the
    uncertainty about $x_1$.

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

The distributions a practitioner needs<br>**from Bernoulli to the Gaussian, and the family they belong to**.
:::
:::

::: {.slide title="A family, not a list"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Fourteen distributions cover almost everything in practice, and they
**connect**: Bernoulli is the seed; construction and limit arrows grow
the rest; conjugate priors close the tree, all inside one envelope.

::: {.d2l-note}
Learn the *map*, not a flat list of formulas.
:::
:::

::: {.col .fig .big}
@fig:mdl-prob-family-tree
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Discrete distributions]{.dtitle}

[Bernoulli · Categorical · Uniform · Binomial · Poisson]{.dsub}
:::
:::

::: {.slide title="Bernoulli: the seed"}
[Discrete]{.kicker}

One coin flip: $P(X=1)=p$, $P(X=0)=1-p$. Because $X^2 = X$, both moments
collapse instantly: $\mathbb E[X]=p$, $\operatorname{Var}(X)=p(1-p)$.

@distributions-bernoulli

::: {.d2l-note .rule}
Every binary classifier outputs a Bernoulli; its negative log-likelihood
**is** binary cross-entropy.
:::
:::

::: {.slide title="Categorical: softmax in disguise"}
[Discrete]{.kicker}

$K$ outcomes with $P(X=k)=p_k$. A network produces the $p_k$ from logits
through the softmax $p_k = e^{z_k}/\sum_j e^{z_j}$, and the NLL is exactly
cross-entropy.

@distributions-categorical

::: {.d2l-note}
The **Gumbel-max** trick samples a categorical exactly; its soft version
makes discrete choices differentiable.
:::
:::

::: {.slide title="The discrete gallery"}
[Discrete]{.kicker}

Four laws in one picture, mass on the integers:

@fig:mdl-prob-discrete-pmfs

Uniform spreads mass evenly ($\mu=\tfrac{n+1}{2}$); the others concentrate
it where events are likely.
:::

::: {.slide title="Binomial: moments for free"}
[Discrete]{.kicker}

A Binomial is a **sum of $n$ Bernoullis**, $X=\sum_i X_i$, so linearity
and independence hand us the moments with no algebra:

$$\mu = \sum_i \mathbb E[X_i] = np, \qquad
\sigma^2 = \sum_i \operatorname{Var}(X_i) = np(1-p).$$

@!distributions-binomial
:::

::: {.slide title="Poisson: the many-rare limit"}
[Discrete]{.kicker}

Take $\text{Binomial}(n,\lambda/n)$ and send $n\to\infty$:

$$\binom{n}{k}\Bigl(\tfrac{\lambda}{n}\Bigr)^k\Bigl(1-\tfrac{\lambda}{n}\Bigr)^{n-k}
\longrightarrow \frac{\lambda^k e^{-\lambda}}{k!}.$$

@!distributions-binomial-to-poisson

::: {.d2l-note}
Mean $=$ variance $=\lambda$ is the Poisson fingerprint:
**over-dispersion** (variance $>$ mean) signals a too-simple model.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Continuous distributions]{.dtitle}

[Uniform · Exponential · Gaussian · Laplace · Multivariate Gaussian]{.dsub}
:::
:::

::: {.slide title="Uniform: raw randomness"}
[Continuous]{.kicker}

::: {.cols .vc}
::: {.col}
Density $\tfrac{1}{b-a}$ on $[a,b]$; mean $\tfrac{a+b}{2}$, variance
$\tfrac{(b-a)^2}{12}$.

The unit uniform powers inverse-transform sampling, Monte Carlo,
dropout masks, and initialization.

@distributions-continuous-uniform
:::

::: {.col .fig}
@fig:mdl-prob-continuous-pdfs
:::
:::
:::

::: {.slide title="Exponential: memorylessness"}
[Continuous]{.kicker}

Waiting times: $p(x)=\lambda e^{-\lambda x}$, $F(x)=1-e^{-\lambda x}$,
mean $1/\lambda$, variance $1/\lambda^2$.

$$P(X > s+t \mid X > s) = e^{-\lambda t} = P(X > t).$$

@distributions-exponential

::: {.d2l-note .rule}
The only memoryless continuous law (converse: Exercise 4); the continuous
partner of the Poisson, and the source of $X=-\log U/\lambda$ sampling.
:::
:::

::: {.slide title="Gaussian: the CLT limit" layout="tight"}
[Continuous]{.kicker}

Standardized sums of *any* iid finite-variance terms converge to the
Gaussian. Watch universality happen, starting from the least Gaussian
summand available, the flat, hard-edged uniform:

@!mdl-distributions-gaussian-1

$n=1$ is a plateau, $n=2$ a triangle; by $n=32$ the sum is
indistinguishable from $\mathcal N(0,1)$ at this resolution. Nothing in
the code knows about the Gaussian: **summation manufactures it**. (It is
also the maximum-entropy law for fixed mean and variance.)
:::

::: {.slide title="The normalizer: a polar trick"}
[Continuous]{.kicker}

Why $\sqrt{2\pi}$? Recall from the integral-calculus chapter: square the
integral and switch to polar coordinates,

$$I^2 = \!\int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy
= \!\int_0^{2\pi}\!\!\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta = 2\pi.$$

. . .

The Jacobian factor $r$ makes the radial integral elementary, so
$I=\sqrt{2\pi}$; the full derivation lives in the integral-calculus
chapter.
:::

::: {.slide title="Laplace: the L1 sibling"}
[Continuous]{.kicker}

$p(x)=\tfrac{1}{2b}e^{-|x-\mu|/b}$: a sharp peak and heavier (exponential)
tails than the Gaussian; variance $2b^2$.

@distributions-laplace

::: {.d2l-note .rule}
Its NLL is $|y-\hat y|$ (MAE); as a prior it gives the $\ell_1$ /
LASSO penalty; its ML location estimator is the **median**, not the mean.
:::
:::

::: {.slide title="Multivariate Gaussian: covariance geometry"}
[Continuous]{.kicker}

::: {.cols .vc}
::: {.col}
$$p(\mathbf x)\propto \exp\!\Bigl(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top
\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Bigr).$$

Contours are ellipsoids: axes along the **eigenvectors** of
$\boldsymbol\Sigma$, half-lengths $\propto\sqrt{\lambda_i}$. Isotropy =
spheres = independent coordinates. Sampling is the Cholesky recipe:
$\boldsymbol\Sigma=\mathbf L\mathbf L^\top$,
$\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$.

@!distributions-mvn
:::

::: {.col .fig}
@fig:mdl-prob-mvn-contours
:::
:::
:::

::: {.slide title="Closed under conditioning"}
[Continuous]{.kicker}

Partition $\mathbf x=(\mathbf x_1,\mathbf x_2)$. Then $\mathbf x_1\mid\mathbf x_2$
is Gaussian with

$$\boldsymbol\mu_{1\mid 2} = \boldsymbol\mu_1 +
\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}(\mathbf x_2-\boldsymbol\mu_2),
\quad
\boldsymbol\Sigma_{1\mid 2} = \boldsymbol\Sigma_{11} -
\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.$$

::: {.d2l-note .rule}
Conditional mean is **linear** in $\mathbf x_2$; the Schur-complement
covariance is the entire engine of Gaussian-process regression.
:::
:::

::: {.slide title="In high dimension, the Gaussian is a thin shell"}
[Continuous]{.kicker}

$\|\mathbf x\|^2$ sums $d$ independent mean-$1$ terms, so it concentrates
near $d$: the mass lives in a shell of radius $\approx\sqrt d$, *far from
the origin where the density is pointwise largest*; and two independent
draws are nearly orthogonal, cosine $\sim 1/\sqrt d$:

@!mdl-distributions-multivariate-gaussian-1

::: {.d2l-note}
Load-bearing facts: $1/\sqrt d$ **initialization** keeps
$\|\mathbf{Wx}\|\approx\|\mathbf x\|$; cosine similarity is informative
*because* unrelated vectors sit near $0$; nearest-neighbor contrast
fades. Exponential tail bounds arrive in the concentration-and-generalization
section.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The exponential family]{.dtitle}

[one form, maximum entropy, the moment property]{.dsub}
:::
:::

::: {.slide title="One shared form"}
[Unification]{.kicker}

$$p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,
\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).$$

Base measure $h$, **natural parameters** $\boldsymbol\eta$, sufficient
statistics $T$, log-partition $A$. Bernoulli ($\eta=\operatorname{logit}p$,
$A=\operatorname{softplus}$), Poisson ($\eta=\log\lambda$), and Gaussian
all fit.

::: {.d2l-note}
Two exclusions, two reasons: the **uniforms** stay outside because their
support moves with the parameters; **Cauchy and Student-$t$** do not admit a
fixed finite-dimensional sufficient statistic for their usual unknown
location-and-scale families. They therefore lack the standard
finite-dimensional conjugate update, and their negative log-likelihoods are
not generally convex.
:::
:::

::: {.slide title="Where the form comes from"}
[Unification]{.kicker}

Maximize entropy $H_h[p]$ subject to fixed averages
$\mathbb E[T(\mathbf x)] = \boldsymbol\tau$. The Lagrange multipliers
*are* the natural parameters, and the maximizer is exactly
$p\propto h\,e^{\boldsymbol\eta^\top T}$.

::: {.d2l-note .rule}
The exponential family is the **least-committal** family consistent with a
chosen set of expected statistics.
:::
:::

::: {.slide title="The moment property"}
[Unification]{.kicker}

Differentiating the log-partition recovers the moments:

$$\nabla A(\boldsymbol\eta) = \mathbb E[T(\mathbf x)], \qquad
\nabla^2 A(\boldsymbol\eta) = \operatorname{Cov}(T) \succeq 0.$$

So $A$ is **convex** and the MLE equation is moment matching,
$\mathbb E[T] = \bar T$. Autograd confirms $dA/d\eta=\sigma(\eta)$ for the
Bernoulli:

@distributions-exp-family
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Conjugate priors]{.dtitle}

[Beta · Gamma · Dirichlet: updating is counting]{.dsub}
:::
:::

::: {.slide title="Beta–Bernoulli: pseudo-counts"}
[Priors]{.kicker}

::: {.cols .vc}
::: {.col}
A $\text{Beta}(\alpha,\beta)$ prior times a Bernoulli likelihood with $x$
heads in $n$ flips gives a $\text{Beta}(\alpha+x,\beta+n-x)$ posterior;
$\alpha,\beta$ act as **phantom** heads and tails:

$$\mathbb E[\theta\mid X] = \frac{\alpha+x}{\alpha+\beta+n}.$$

@!distributions-conjugate

::: {.d2l-note}
$\alpha=\beta=1$ recovers Laplace's rule of succession $(x+1)/(n+2)$.
:::
:::

::: {.col .fig}
@fig:mdl-prob-beta-posterior
:::
:::
:::

::: {.slide title="The rest of the tier"}
[Priors]{.kicker}

::: {.cols}
::: {.col}
**Gamma → Poisson.** Pseudo-events over a pseudo-window; marginalizing
the rate gives the over-dispersed negative binomial.
:::

::: {.col}
**Dirichlet → Categorical.** A multivariate Beta: one pseudo-count per
class on the simplex.
:::
:::

::: {.d2l-note .rule}
**General fact.** Every exponential-family likelihood has a conjugate
prior of the same form; its hyperparameters are pseudo-data
$(\boldsymbol\nu,\kappa)$.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Bernoulli → Binomial → Poisson / Gaussian by sums and limits; Categorical → Multinomial.
- Moments from linearity, limits, and a few integrals.
- Gaussian is central: CLT, max entropy, closed under linear maps and conditioning (Schur complement).
:::

::: {.col}
- The exponential family unifies them; $\nabla A=\mathbb E[T]$, and its NLL is convex in natural parameters.
- The usual conjugate families (Beta / Gamma / Dirichlet) update by adding sufficient statistics.
- The named shapes are the vocabulary; maximum likelihood is the grammar (next).
:::
:::
:::
