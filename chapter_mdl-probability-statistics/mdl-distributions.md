# Distributions
:label:`sec_mdl-distributions`

A *distribution* assigns probabilities to the possible values of a random
variable (:numref:`sec_mdl-random_variables`). This section develops eleven
data distributions and three conjugate priors used frequently in machine learning and relates
them through constructions and limits. A sum of Bernoulli variables has a
binomial distribution. Appropriate many-trial limits yield the Poisson or,
under the central limit theorem, the Gaussian distribution. The categorical
distribution extends Bernoulli variables to $K$ outcomes and provides the
probabilistic model behind a softmax classifier. Many of these distributions
belong to the exponential family and admit conjugate priors, which simplifies
maximum-likelihood and Bayesian calculations.

:numref:`fig_mdl-prob-family-tree` summarizes these relationships. Its arrows
denote constructions and limiting arguments, and the enclosing region marks
the distributions that belong to the exponential family.

![The distribution family. Solid arrows *construct*: Bernoulli is the seed; summing $n$ copies gives the Binomial; the many-and-rare limit ($n\to\infty$, $np\to\lambda$) gives the Poisson, whose waiting time between events is the Exponential; the many-and-ordinary limit (the central limit theorem) gives the Gaussian; the Categorical generalizes Bernoulli to $K$ outcomes and the Multinomial counts $n$ of them. Dashed arrows attach each likelihood to its *conjugate prior*: the Beta to the Bernoulli and Binomial, the Gamma to the Poisson, and the Dirichlet to the Categorical and Multinomial. Every node, the conjugate priors included, lies inside the exponential-family envelope.](../img/mdl-prob-family-tree.svg)
:label:`fig_mdl-prob-family-tree`

The first two parts form a reference gallery; the exponential-family and
conjugacy parts explain shared structure. A modeling choice also depends on the
support and data-generating question:

| observed quantity | useful starting model | condition or omission to check |
|:--|:--|:--|
| binary or one-of-$K$ outcome | Bernoulli or categorical | probabilities may depend on covariates |
| successes in fixed trials | binomial | conditionally independent trials with a common success probability |
| event count over exposure | Poisson | mean equals variance; overdispersion may require another count law |
| positive waiting time | exponential | constant hazard; Gamma, Weibull, and log-normal laws are omitted |
| bounded continuous value | continuous uniform as a baseline | an application need not have constant density |
| real-valued residual | Gaussian or Laplace | mixtures and heavier-tailed laws may fit better |
| correlated real vector | multivariate Gaussian | multimodality and structured covariance need richer models |

For each distribution we use a common template: its mass or density function,
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

![A gallery of discrete probability mass functions in one consistent style, left to right: $\mathrm{Bernoulli}(0.3)$, a discrete uniform on $\{1,\dots,6\}$, $\mathrm{Binomial}(10,0.4)$, and $\mathrm{Poisson}(4)$. The binomial and Poisson share the same right-skewed bell silhouette, foreshadowing their limiting relationship. The last three panels share a vertical scale; the Bernoulli panel has its own.](../img/mdl-prob-discrete-pmfs.svg)
:label:`fig_mdl-prob-discrete-pmfs`

### Bernoulli

The Bernoulli is the simplest non-trivial random variable: a single coin flip that
comes up $1$ with probability $p$ and $0$ with probability $1-p$. We write
$X\sim\mathrm{Bernoulli}(p)$, with pmf

$$
p_1 = P(X=1) = p, \qquad p_0 = P(X=0) = 1-p .
$$
:eqlabel:`eq_mdl-bernoulli_pmf`

**Where it arises.** A probabilistic binary classifier commonly uses a
Bernoulli output: the model emits $\hat p$, and the negative log-likelihood
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
distribution, whose standard form has the CDF $\exp(-e^{-x})$, is the
location--scale law of the maxima of many independent draws
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

Representing a binomial variable as a sum of Bernoulli variables gives its mean
and variance without manipulating binomial coefficients. Successive pmf terms
differ by the ratio

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
*forces* the exponential form, is
exercise :exref:`ex_mdl-distributions-memoryless-converse`.

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
:label:`subsec_mdl-gaussian-clt`

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
checkable in a few lines. The cell below starts from a summand that looks
nothing like a Gaussian (a $U(0,1)$ draw, flat with hard edges and no tails), forms
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
family's moment property, :eqref:`eq_mdl-exp_family_moment`, recovers both by
differentiation alone; see
exercise :exref:`ex_mdl-distributions-log-partition-moments`.)

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

**Where it arises.** The Laplace likelihood corresponds to the $L_1$ loss, just
as the Gaussian likelihood corresponds to squared error: its negative
log-likelihood is proportional to $|y-\hat y|$, the *mean absolute error*. As a
prior on weights, it gives the sparsity-inducing regularizer
$\|\mathbf w\|_1$ used by LASSO
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
$\hat\mu$ under a Laplace minimizes $\sum_i|x_i-\mu|$, whose minimizers include
the sample *median* (robust to outliers), whereas the Gaussian's is the mean. The cell
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

![Samples from a $2$-D Gaussian (blue points) with the elliptical density contours overlaid. The ellipse axes lie along the eigenvectors of the covariance $\boldsymbol\Sigma$ with half-lengths proportional to $\sqrt{\lambda_i}$ (the standard deviation along each principal direction); the orange arrows are the eigenvectors $\mathbf{w}_1,\mathbf{w}_2$ scaled by $2\sqrt{\lambda_i}$. This connects the multivariate Gaussian directly to the spectral theorem.](../img/mdl-prob-mvn-contours.svg)
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
carried out in :cite:`Bishop.2006`, §2.3; Exercise :exref:`ex_mdl-distributions-using-conditional-formula-eqref` puts the formulas to work):

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
is the *Schur complement* of $\boldsymbol\Sigma_{22}$, the block that remains
when the variables in $\mathbf x_2$ are eliminated. It does not depend on the
observed value $\mathbf x_2$. These formulas underlie Gaussian process
regression (:numref:`chap_gp`): conditioning the joint Gaussian prior over
function values on the observations gives the predictive mean and variance.

**Sampling via Cholesky.** How do we draw from
$\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ given only a standard-normal
generator? This is the application promised when the Cholesky factorization was
introduced (:numref:`subsec_mdl-psd`): factor
$\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ with $\mathbf L$ lower triangular,
draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ ($d$ independent standard
normals), and set $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$. Then $\mathbf x$
is Gaussian (a linear map of a Gaussian), and its mean is $\boldsymbol\mu$
because $\mathbb E[\mathbf z]=\mathbf 0$. Its covariance is

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

## Summary

Here are the eleven data distributions of this section in one view; their three
conjugate priors, the Beta, Gamma, and Dirichlet, close the family. For the categorical
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
## Exercises

1. **Error counts on a test set.** A classifier is scored on $n=10{,}000$
   independent test examples, so that its number of errors $E$ has the
   distribution $\mathrm{Binomial}(n,q)$, where $q$ is its true error rate.
    1. Let $q=0.1$. Use the Gaussian approximation of the binomial (the
       de Moivre--Laplace theorem) to find the interval around $nq$ that
       contains $E$ with probability about $95\%$. Two classifiers with this
       error rate are scored on two independent test sets of this size. Find
       the standard deviation of the difference of their observed accuracies.
    1. Let $q=5\cdot10^{-4}$. Compute $P(E=0)$ and $P(E\ge10)$ exactly and
       under the Poisson approximation $\mathrm{Poisson}(nq)$. Show that the
       logarithm of $P(E=0)$ under the binomial differs from its Poisson
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
       where $|\rho|<1$. Show that $x_1\mid x_2\sim\mathcal N(\rho x_2,1-\rho^2)$.
       Why does observing $x_2$ never increase the uncertainty about $x_1$?
    1. A signal $s\sim\mathcal N(0,\tau^2)$ is observed with additive noise,
       $y=s+\epsilon$, where $\epsilon\sim\mathcal N(0,\sigma^2)$ is
       independent of $s$. Write $(s,y)$ as a linear map of $(s,\epsilon)$ to
       obtain its joint Gaussian distribution, and compute the conditional
       distribution of $s$ given $y$. Show that the conditional precision
       $1/\operatorname{Var}(s\mid y)$ is the sum of the prior precision
       $1/\tau^2$ and the noise precision $1/\sigma^2$. How does
       $\mathbb E[s\mid y]$ behave as $\sigma^2\to0$ and as
       $\sigma^2\to\infty$? This computation is Gaussian process
       regression (:numref:`chap_gp`) with a single observation.
   :exlabel:`ex_mdl-distributions-using-conditional-formula-eqref`

1. **Gamma--Poisson mixture.** Let the Poisson rate itself be random,
   $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$ with the density of
   :numref:`subsec_mdl-other-conjugate-pairs`, and let
   $X\mid\lambda\sim\mathrm{Poisson}(\lambda)$.
    1. Use the laws of total expectation and total variance
       :eqref:`eq_mdl-tower` to compute $\mathbb E[X]$ and
       $\operatorname{Var}(X)$. Write the variance in terms of the mean
       $m=\alpha/\beta$ and the shape $\alpha$. In which limit of $\alpha$
       and $\beta$ does the Poisson reappear?
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
       $\mathrm{Bernoulli}(1/2)$, from an exact formula for $F_n$ in each
       case. For uniform summands use the Irwin--Hall formula

       $$P(S_n\le s)=\frac{1}{n!}\sum_{k=0}^{\lfloor s\rfloor}(-1)^k\binom{n}{k}(s-k)^n,$$

       evaluated in exact rational arithmetic with Python's
       `fractions.Fraction`, because its terms cancel catastrophically in
       floating point once $n$ exceeds about $40$. For exponential summands,
       $S_n$ is the time of the $n$-th event of a Poisson process with rate
       $1$, so $P(S_n\le s)=P(N\ge n)$ for $N\sim\mathrm{Poisson}(s)$. For the
       two continuous summands, take the supremum over a fine grid of
       $x\in[-4,4]$. For Bernoulli summands, compare $\Phi$ with $F_n$ on
       both sides of every jump. Plot $D_n$ against $n$ on log-log axes and
       fit the slopes.
    1. For summands that are not confined to a lattice and have skewness
       $\gamma=\mathbb E[(X-\mu)^3]/\sigma^3$, the Edgeworth expansion
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

A core distribution vocabulary<br>**from Bernoulli to the Gaussian, and the family they belong to**.
:::
:::

::: {.slide title="A family, not a list"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
These fourteen distributions form a useful core vocabulary, and they
**connect** through constructions, limits, and conjugacy. Important omissions
include mixtures, heavy-tailed multivariate laws, survival families, and
structured stochastic processes.

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

~~~python
big = 1 * (rng.random(10000) < p)   # 1 with prob p, else 0
~~~

@!distributions-bernoulli

::: {.d2l-note .rule}
A probabilistic binary classifier with a Bernoulli likelihood has binary
cross-entropy as its negative log-likelihood.
:::
:::

::: {.slide title="Categorical: softmax in disguise"}
[Discrete]{.kicker}

$K$ outcomes with $P(X=k)=p_k$. A network produces the $p_k$ from logits
through the softmax $p_k = e^{z_k}/\sum_j e^{z_j}$, and the NLL is exactly
cross-entropy.

~~~python
p_cat = onp.exp(z) / onp.exp(z).sum()   # softmax -> categorical
~~~

@!distributions-categorical

::: {.d2l-note}
The **Gumbel-max** trick samples a categorical exactly; its soft version
makes discrete choices differentiable.
:::
:::

::: {.slide title="The discrete gallery"}
[Discrete]{.kicker}

The figure compares four probability laws on the integers:

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

~~~python
sample = (b - a) * rng.random(10000) + a   # shift-and-scale
~~~

@!distributions-continuous-uniform
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

~~~python
sample = -onp.log(U) / lam   # inverse-transform sampler
~~~

@!distributions-exponential

::: {.d2l-note .rule}
The only memoryless continuous law; the continuous
partner of the Poisson, and the source of $X=-\log U/\lambda$ sampling.
:::
:::

::: {.slide title="Gaussian: the CLT limit" layout="tight"}
[Continuous]{.kicker}

Standardized sums of *any* iid finite-variance terms converge to the
Gaussian. Start from the least Gaussian summand, the flat uniform:

@!mdl-distributions-gaussian-1

$n=1$ is a plateau, $n=2$ a triangle; by $n=32$ the sum is
indistinguishable from $\mathcal N(0,1)$: **summation manufactures the
Gaussian**. It is also the maximum-entropy law for fixed mean and variance.
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

~~~python
lap = mu - b * onp.sign(U) * onp.log(1 - 2 * onp.abs(U))
~~~

@!distributions-laplace

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
covariance determines the posterior uncertainty in Gaussian-process regression.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Bernoulli → Binomial → Poisson / Gaussian by sums and limits; Categorical → Multinomial.
- Moments from linearity, limits, and a few integrals.
- Gaussian is central: CLT, max entropy, closed under linear maps and conditioning (Schur complement).
:::
