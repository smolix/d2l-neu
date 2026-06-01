# Distributions
:label:`sec_mdl-distributions`

A *distribution* is the rulebook a random variable plays by: it assigns a
probability to every outcome (:numref:`sec_mdl-random_variables`). A working
deep-learning practitioner does not need a zoo of hundreds---a dozen
distributions cover almost everything, and they are not a flat list but a small
family with a few generators and a handful of limiting arrows connecting them.
The Bernoulli coin flip is the seed: sum $n$ of them and you get the binomial;
let the trials become many and rare and the binomial collapses to the Poisson;
let them become many and ordinary and the *central limit theorem* sends it to the
Gaussian. The categorical generalizes the Bernoulli from two outcomes to $K$, and
its softmax parameterization is the output layer of every classifier. At the end
we will see that *all* of these---and more---are special cases of one form, the
*exponential family*, which is the reason their maximum-likelihood losses are the
clean, convex objectives we minimize in practice.

This is the punchline worth keeping in view, and :numref:`fig_mdl-prob-family-tree`
draws it: the distributions are nodes, the construction and limit relationships
are arrows, and the whole picture sits inside the exponential-family envelope.

![The distribution family. Bernoulli is the seed: summing $n$ copies gives the Binomial; the many-and-rare limit ($n\to\infty$, $np\to\lambda$) gives the Poisson; the many-and-ordinary limit (the central limit theorem) gives the Gaussian. The Categorical generalizes Bernoulli to $K$ outcomes and the Multinomial counts $n$ of them. Every node lies inside the exponential-family envelope.](../img/mdl-prob-family-tree.svg)
:label:`fig_mdl-prob-family-tree`

For each distribution we keep to a tight template: its mass or density function,
where it *arises* in machine learning, its mean and variance *derived* rather than
asserted, and at most one compact teaching cell that evaluates the law and draws a
sample. We treat the discrete distributions first, then the continuous ones, then
unify them. The worked cells branch per framework, so we load each library once
here. The framework-agnostic formula checks use plain NumPy.

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

## Discrete Distributions

A discrete random variable takes values in a countable set, and its distribution
is a *probability mass function* (pmf) $p_k = P(X=k)$ with $p_k\ge0$ and
$\sum_k p_k=1$. Mean and variance are the sums
$\mathbb E[X]=\sum_k k\,p_k$ and
$\operatorname{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2$
(:numref:`sec_mdl-random_variables`). The five distributions here are a connected
chain, not five unrelated facts: each is built from, or is a limit of, the one
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
classifier*: a model emits $\hat p$ and the negative log-likelihood
$-y\log\hat p-(1-y)\log(1-\hat p)$ is exactly the *binary cross-entropy* loss.

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
#@tab mxnet
p = 0.3
sample = 1 * (np.random.rand(3, 3) < p)        # 1 with prob p, else 0
print('pmf  P(0), P(1) =', (1 - p, p))
print('sample mean =', float(sample.mean()), ' (≈ p)')
sample
```

```{.python .input #distributions-bernoulli}
#@tab pytorch
p = 0.3
sample = (torch.rand(3, 3) < p).float()        # 1 with prob p, else 0
print('pmf  P(0), P(1) =', (1 - p, p))
print('sample mean =', float(sample.mean()), ' (≈ p)')
sample
```

```{.python .input #distributions-bernoulli}
#@tab tensorflow
p = 0.3
sample = tf.cast(tf.random.uniform((3, 3)) < p, tf.float32)
print('pmf  P(0), P(1) =', (1 - p, p))
print('sample mean =', float(tf.reduce_mean(sample)), ' (≈ p)')
sample
```

```{.python .input #distributions-bernoulli}
#@tab jax
p = 0.3
sample = jax.random.bernoulli(jax.random.PRNGKey(0), p, (3, 3)).astype(jnp.float32)
print('pmf  P(0), P(1) =', (1 - p, p))
print('sample mean =', float(sample.mean()), ' (≈ p)')
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

**Where it arises.** This is the single most-used distribution in deep learning:
every *softmax* output layer turns a vector of scores (logits) $\mathbf z$ into a
categorical over classes or vocabulary tokens via

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}},
$$
:eqlabel:`eq_mdl-softmax`

and the negative log-likelihood of the categorical,
$-\sum_k y_k\log p_k$ with $\mathbf y$ the one-hot label, is exactly the
*cross-entropy* loss (:numref:`sec_mdl-information_theory`, `sec_softmax`). The
Bernoulli is the case $K=2$.

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
#@tab mxnet
z = np.array([2.0, 1.0, 0.1, -1.0])            # logits over K = 4 classes
p_cat = np.exp(z) / np.exp(z).sum()            # softmax -> categorical
draw = int(np.random.choice(len(p_cat), p=p_cat.tolist()))
print('categorical p =', p_cat.round(3), ' sum =', float(p_cat.sum()))
print('one sample (argmax-ish, random) -> class', draw)
```

```{.python .input #distributions-categorical}
#@tab pytorch
z = torch.tensor([2.0, 1.0, 0.1, -1.0])        # logits over K = 4 classes
p_cat = torch.softmax(z, dim=0)                # softmax -> categorical
draw = int(torch.multinomial(p_cat, 1))
print('categorical p =', p_cat.numpy().round(3), ' sum =', float(p_cat.sum()))
print('one sample (random) -> class', draw)
```

```{.python .input #distributions-categorical}
#@tab tensorflow
z = tf.constant([2.0, 1.0, 0.1, -1.0])         # logits over K = 4 classes
p_cat = tf.nn.softmax(z)                        # softmax -> categorical
draw = int(tf.random.categorical(tf.math.log(p_cat)[None], 1)[0, 0])
print('categorical p =', p_cat.numpy().round(3), ' sum =', float(tf.reduce_sum(p_cat)))
print('one sample (random) -> class', draw)
```

```{.python .input #distributions-categorical}
#@tab jax
z = jnp.array([2.0, 1.0, 0.1, -1.0])           # logits over K = 4 classes
p_cat = jax.nn.softmax(z)                        # softmax -> categorical
draw = int(jax.random.categorical(jax.random.PRNGKey(0), z))
print('categorical p =', np.asarray(p_cat).round(3), ' sum =', float(p_cat.sum()))
print('one sample (random) -> class', draw)
```

### Discrete Uniform

If a categorical assigns *equal* probability to each of its outcomes it is the
*discrete uniform*. Supported on $\{1,2,\dots,n\}$, every value is equally likely,
$p_i=\tfrac1n$, written $X\sim U(n)$.

**Where it arises.** It is the maximum-entropy distribution on a finite set with no
prior knowledge---the honest default when all we know is the list of
possibilities---and it models a fair die, a uniformly chosen index, or a token
sampled with no preference.

**Mean and variance.** Both follow from the closed forms
$\sum_{i=1}^n i=\tfrac{n(n+1)}2$ and $\sum_{i=1}^n i^2=\tfrac{n(n+1)(2n+1)}6$:

$$
\mu_X = \frac1n\sum_{i=1}^n i = \frac{n+1}{2},
\qquad
\sigma_X^2 = \frac1n\sum_{i=1}^n i^2 - \mu_X^2 = \frac{n^2-1}{12}.
$$

The mean is the midpoint, by symmetry; the variance grows like $n^2$, since a wider
range of equally likely values is more spread out. A sampling cell suffices.

```{.python .input #distributions-discrete-uniform}
#@tab mxnet
n = 6
sample = np.random.randint(1, n + 1, size=(3, 3))
print('mean (n+1)/2 =', (n + 1) / 2, '  var (n^2-1)/12 =', (n**2 - 1) / 12)
sample
```

```{.python .input #distributions-discrete-uniform}
#@tab pytorch
n = 6
sample = torch.randint(1, n + 1, size=(3, 3))
print('mean (n+1)/2 =', (n + 1) / 2, '  var (n^2-1)/12 =', (n**2 - 1) / 12)
sample
```

```{.python .input #distributions-discrete-uniform}
#@tab tensorflow
n = 6
sample = tf.random.uniform((3, 3), 1, n + 1, dtype=tf.int32)
print('mean (n+1)/2 =', (n + 1) / 2, '  var (n^2-1)/12 =', (n**2 - 1) / 12)
sample
```

```{.python .input #distributions-discrete-uniform}
#@tab jax
n = 6
sample = jax.random.randint(jax.random.PRNGKey(0), (3, 3), 1, n + 1)
print('mean (n+1)/2 =', (n + 1) / 2, '  var (n^2-1)/12 =', (n**2 - 1) / 12)
sample
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

**Mean and variance, the elegant way.** Rather than wrestle with the binomial
coefficient, use the representation :eqref:`eq_mdl-binomial_sum` as a sum of
Bernoullis. *Expectation is linear*---always, even for dependent terms---so

$$
\mu_X = \mathbb E\Bigl[\sum_i X_i\Bigr] = \sum_i \mathbb E[X_i] = np .
$$

The $X_i$ are *independent*, and variance is additive over independent summands, so

$$
\sigma_X^2 = \operatorname{Var}\Bigl(\sum_i X_i\Bigr) = \sum_i \operatorname{Var}(X_i) = np(1-p) .
$$

Two one-line sums replace a page of algebra---the payoff of seeing the binomial as
*built from* Bernoullis. We evaluate the pmf and draw samples below.

```{.python .input #distributions-binomial}
#@tab mxnet
from scipy.special import comb
n, p = 10, 0.4
k = np.arange(n + 1)
pmf = comb(n, k) * p**k * (1 - p)**(n - k)
print('mean np =', n * p, '  var np(1-p) =', n * p * (1 - p))
print('P(X=k):', pmf.round(3))
np.random.binomial(n, p, size=(3, 3))           # sample counts of successes
```

```{.python .input #distributions-binomial}
#@tab pytorch
n, p = 10, 0.4
m = torch.distributions.Binomial(n, torch.tensor(p))
k = torch.arange(n + 1.)
pmf = m.log_prob(k).exp()
print('mean np =', n * p, '  var np(1-p) =', n * p * (1 - p))
print('P(X=k):', pmf.numpy().round(3))
m.sample((3, 3))                                 # sample counts of successes
```

```{.python .input #distributions-binomial}
#@tab tensorflow
n, p = 10, 0.4
m = tf.random.stateless_binomial((3, 3), [1, 2], n, p)
from scipy.special import comb
k = np.arange(n + 1)
pmf = comb(n, k) * p**k * (1 - p)**(n - k)
print('mean np =', n * p, '  var np(1-p) =', n * p * (1 - p))
print('P(X=k):', pmf.round(3))
m                                                # sample counts of successes
```

```{.python .input #distributions-binomial}
#@tab jax
from scipy.special import comb
n, p = 10, 0.4
k = np.arange(n + 1)
pmf = comb(n, k) * p**k * (1 - p)**(n - k)
print('mean np =', n * p, '  var np(1-p) =', n * p * (1 - p))
print('P(X=k):', pmf.round(3))
# sum n Bernoulli trials -> counts of successes
jax.random.bernoulli(jax.random.PRNGKey(0), p, (3, 3, n)).sum(-1)
```

### Poisson

What if the trials become *many* and each *rare*? Standing at a bus stop, the
chance of an arrival in any tiny sub-interval is small, but there are many such
intervals. Split one minute into $n$ slices, model each as
$\mathrm{Bernoulli}(p/n)$, and the count is $\mathrm{Binomial}(n,p/n)$. Its mean is
$n\cdot p/n=p$ for every $n$, and its variance $n\cdot\tfrac pn(1-\tfrac pn)\to p$.
The moments stabilize, which signals that a limiting distribution exists.

**The limit, derived.** Take $\mathrm{Binomial}(n,p_n)$ with $n\to\infty$ and
$p_n\to0$ such that $np_n\to\lambda$---the *law of rare events*. Writing
$p_n=\lambda/n$, the pmf at a fixed $k$ factors into three pieces:

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

with $\lambda>0$ the *rate*, the average count per unit time. Writing $X\sim
\mathrm{Poisson}(\lambda)$.

**Where it arises.** Counts of rare events with no fixed ceiling: click counts,
photons on a sensor, mutations in a genome, requests hitting a server.

**Mean and variance.** The limit hands them to us for free: the binomial mean $np_n
\to\lambda$ and variance $np_n(1-p_n)\to\lambda$, so

$$
\mu_X = \lambda, \qquad \sigma_X^2 = \lambda.
$$

Mean equals variance is the Poisson fingerprint---over-dispersed count data (where
the empirical variance exceeds the mean) is the standard sign that a plain Poisson
model is too simple.

```{.python .input #distributions-poisson}
#@tab mxnet
from scipy.special import factorial
lam = 4.0
k = np.arange(15)
pmf = np.exp(-lam) * lam**k / factorial(k)
print('mean = var = lambda =', lam)
print('P(X=k):', pmf.round(3))
np.random.poisson(lam, size=(3, 3))             # sample event counts
```

```{.python .input #distributions-poisson}
#@tab pytorch
lam = 4.0
m = torch.distributions.Poisson(lam)
k = torch.arange(15.)
pmf = m.log_prob(k).exp()
print('mean = var = lambda =', lam)
print('P(X=k):', pmf.numpy().round(3))
m.sample((3, 3))                                 # sample event counts
```

```{.python .input #distributions-poisson}
#@tab tensorflow
from scipy.special import factorial
lam = 4.0
k = np.arange(15)
pmf = np.exp(-lam) * lam**k / factorial(k)
print('mean = var = lambda =', lam)
print('P(X=k):', pmf.round(3))
tf.random.poisson((3, 3), lam)                  # sample event counts
```

```{.python .input #distributions-poisson}
#@tab jax
from scipy.special import factorial
lam = 4.0
k = np.arange(15)
pmf = np.exp(-lam) * lam**k / factorial(k)
print('mean = var = lambda =', lam)
print('P(X=k):', pmf.round(3))
jax.random.poisson(jax.random.PRNGKey(0), lam, (3, 3))  # sample event counts
```

## Continuous Distributions

A continuous random variable is described by a *probability density* (pdf)
$p(x)\ge0$ with $\int p(x)\,dx=1$; probabilities are areas under it and
expectations are integral averages (:numref:`sec_mdl-integral_calculus`). The five
laws below run from the structureless (uniform) to the universal (Gaussian) to the
vector-valued (multivariate Gaussian), with the exponential and Laplace as the
$[0,\infty)$ and heavy-tailed-symmetric cases.
:numref:`fig_mdl-prob-continuous-pdfs` overlays the three densities on the whole
line so their tail behavior is directly comparable: the Gaussian's thin
$e^{-x^2}$ tail, the Laplace's heavier $e^{-|x|}$ tail and sharp peak, and the
uniform's flat plateau.

![A gallery of continuous probability densities in one consistent style: a continuous uniform on the interval from $-2$ to $2$, a standard Gaussian $\mathcal N(0,1)$, and a Laplace with matched variance. The Laplace has a sharper peak and heavier tails than the Gaussian; the uniform is a flat plateau with hard edges.](../img/mdl-prob-continuous-pdfs.svg)
:label:`fig_mdl-prob-continuous-pdfs`

### Continuous Uniform

Refine the discrete uniform---let the $n$ equally likely points fill an interval
$[a,b]$---and you reach the *continuous uniform*, which picks a value in $[a,b]$
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
$b-a$ playing the role of $n$). The default generator samples $U(0,1)$, so we
shift and scale.

```{.python .input #distributions-continuous-uniform}
#@tab mxnet
a, b = 1.0, 3.0
sample = (b - a) * np.random.rand(3, 3) + a
print('mean (a+b)/2 =', (a + b) / 2, '  var (b-a)^2/12 =', (b - a)**2 / 12)
sample
```

```{.python .input #distributions-continuous-uniform}
#@tab pytorch
a, b = 1.0, 3.0
sample = (b - a) * torch.rand(3, 3) + a
print('mean (a+b)/2 =', (a + b) / 2, '  var (b-a)^2/12 =', (b - a)**2 / 12)
sample
```

```{.python .input #distributions-continuous-uniform}
#@tab tensorflow
a, b = 1.0, 3.0
sample = (b - a) * tf.random.uniform((3, 3)) + a
print('mean (a+b)/2 =', (a + b) / 2, '  var (b-a)^2/12 =', (b - a)**2 / 12)
sample
```

```{.python .input #distributions-continuous-uniform}
#@tab jax
a, b = 1.0, 3.0
sample = jax.random.uniform(jax.random.PRNGKey(0), (3, 3), minval=a, maxval=b)
print('mean (a+b)/2 =', (a + b) / 2, '  var (b-a)^2/12 =', (b - a)**2 / 12)
sample
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
survival analysis---anywhere a "constant hazard" is plausible.

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

The waiting clock never "ages"---exactly the property that makes the exponential the
continuous-time partner of the Poisson.

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

The teaching cell samples by *inverse transform*: if $U\sim U(0,1)$ then
$X=-\tfrac1\lambda\log U\sim\mathrm{Exp}(\lambda)$, since inverting
$F(x)=1-e^{-\lambda x}$ gives exactly this map.

```{.python .input #distributions-exponential}
#@tab mxnet
lam = 0.5
U = np.random.rand(100000)
sample = -np.log(U) / lam                        # inverse-transform sampler
print('mean 1/lambda =', 1 / lam, '  sample mean =', float(sample.mean().round(3)))
print('var 1/lambda^2 =', 1 / lam**2, ' sample var =', float(sample.var().round(3)))
```

```{.python .input #distributions-exponential}
#@tab pytorch
lam = 0.5
U = torch.rand(100000)
sample = -torch.log(U) / lam                     # inverse-transform sampler
print('mean 1/lambda =', 1 / lam, '  sample mean =', float(sample.mean()))
print('var 1/lambda^2 =', 1 / lam**2, ' sample var =', float(sample.var()))
```

```{.python .input #distributions-exponential}
#@tab tensorflow
lam = 0.5
U = tf.random.uniform((100000,))
sample = -tf.math.log(U) / lam                   # inverse-transform sampler
print('mean 1/lambda =', 1 / lam, '  sample mean =', float(tf.reduce_mean(sample)))
print('var 1/lambda^2 =', 1 / lam**2, ' sample var =', float(tf.math.reduce_variance(sample)))
```

```{.python .input #distributions-exponential}
#@tab jax
lam = 0.5
U = jax.random.uniform(jax.random.PRNGKey(0), (100000,))
sample = -jnp.log(U) / lam                        # inverse-transform sampler
print('mean 1/lambda =', 1 / lam, '  sample mean =', float(sample.mean()))
print('var 1/lambda^2 =', 1 / lam**2, ' sample var =', float(sample.var()))
```

### Gaussian

The *Gaussian* (or normal) distribution is the most important in all of
probability, because it is what sums of many small independent effects look like.
Return to the binomial $X^{(n)}\sim\mathrm{Binomial}(n,p)$, but now hold $p$ fixed
and send $n\to\infty$. Both mean $np$ and variance $np(1-p)$ blow up, so we
standardize,

$$
Y^{(n)} = \frac{X^{(n)} - np}{\sqrt{np(1-p)}},
$$

which has mean $0$ and variance $1$ for every $n$. The *central limit theorem*
(CLT) states that $Y^{(n)}$ converges in distribution to the *standard Gaussian*:
for any interval $[a,b]$, $P(Y^{(n)}\in[a,b])\to P(\mathcal N(0,1)\in[a,b])$. The
limiting density is

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr),
\qquad X\sim\mathcal N(\mu,\sigma^2).
$$
:eqlabel:`eq_mdl-gaussian_pdf`

Nothing about coin flips was special: the CLT holds for *any* independent,
identically distributed summands with *finite variance* (no higher moments
needed)---which is why the conditions are so easy to meet and the Gaussian is
everywhere.

**The normalizer, via the polar trick.** Why the constant $1/\sqrt{2\pi\sigma^2}$?
Take $\mu=0,\sigma=1$; we need $\int_{-\infty}^\infty e^{-x^2/2}\,dx=\sqrt{2\pi}$.
The one-dimensional integral has no elementary antiderivative, but its *square*
becomes a two-dimensional integral that polar coordinates crack open
(:numref:`sec_mdl-integral_calculus`). With $I=\int e^{-x^2/2}\,dx$,

$$
I^2 = \int_{-\infty}^\infty\!\!\int_{-\infty}^\infty e^{-(x^2+y^2)/2}\,dx\,dy
= \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta
= 2\pi\cdot 1 = 2\pi,
$$

where the Jacobian factor $r$ (the area-stretch of the polar map) makes the radial
integral elementary, $\int_0^\infty r\,e^{-r^2/2}\,dr=1$. Hence $I=\sqrt{2\pi}$ and
the density integrates to one. The mean and variance are then $\mu$ and $\sigma^2$
by construction: the parameters *are* the first two moments.

**Maximum entropy.** Among all distributions with a fixed mean and variance, the
Gaussian has the largest entropy (:numref:`sec_mdl-information_theory`)---it is the
*most noncommittal* choice consistent with knowing only those two numbers, the most
honest default.

**Where it arises.** As the CLT limit it models aggregate noise; it is the noise
model behind *regression* (modeling $y\sim\mathcal N(\hat y,\sigma^2)$ makes the
negative log-likelihood $\tfrac{(y-\hat y)^2}{2\sigma^2}+\text{const}$, so
maximizing it *is* minimizing mean squared error); and it is the default prior and
latent distribution throughout deep generative modeling.

```{.python .input #distributions-gaussian}
#@tab mxnet
mu, sigma = 0.0, 1.0
x = np.arange(-4, 4, 0.01)
p = np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
print('total mass (≈1):', float((0.01 * p).sum().round(4)))
np.random.normal(mu, sigma, size=(3, 3))        # sample from N(mu, sigma^2)
```

```{.python .input #distributions-gaussian}
#@tab pytorch
mu, sigma = 0.0, 1.0
x = torch.arange(-4, 4, 0.01)
p = torch.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
print('total mass (≈1):', float((0.01 * p).sum()))
torch.normal(mu, sigma, size=(3, 3))            # sample from N(mu, sigma^2)
```

```{.python .input #distributions-gaussian}
#@tab tensorflow
mu, sigma = 0.0, 1.0
x = tf.range(-4, 4, 0.01)
p = tf.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
print('total mass (≈1):', float(tf.reduce_sum(0.01 * p)))
tf.random.normal((3, 3), mu, sigma)             # sample from N(mu, sigma^2)
```

```{.python .input #distributions-gaussian}
#@tab jax
mu, sigma = 0.0, 1.0
x = jnp.arange(-4, 4, 0.01)
p = jnp.exp(-(x - mu)**2 / (2 * sigma**2)) / jnp.sqrt(2 * jnp.pi * sigma**2)
print('total mass (≈1):', float((0.01 * p).sum()))
mu + sigma * jax.random.normal(jax.random.PRNGKey(0), (3, 3))  # sample N(mu, sigma^2)
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
sparsity-inducing $L_1$ regularizer $\propto\|\mathbf w\|_1$, the engine of LASSO.

**Mean and variance.** By symmetry $\mu_X=\mu$. For the variance, center at $\mu$
and use $\int_0^\infty u^2 e^{-u}\,du=2$ with $u=|x-\mu|/b$:

$$
\sigma_X^2 = \int_{-\infty}^\infty \frac{(x-\mu)^2}{2b}e^{-|x-\mu|/b}\,dx
= \frac{2}{2b}\int_0^\infty x^2 e^{-x/b}\,dx
= \frac{1}{b}\cdot b^3\!\int_0^\infty u^2 e^{-u}\,du
= 2b^2 .
$$

A clean teaching contrast is the *location estimator*: the maximum-likelihood
$\hat\mu$ under a Laplace minimizes $\sum_i|x_i-\mu|$, whose minimizer is the sample
*median*---robust to outliers---whereas the Gaussian's is the mean. The cell
matches a Laplace and a Gaussian on variance and confirms the Laplace's heavier
tail by the fraction of mass beyond $3$ standard deviations.

```{.python .input #distributions-laplace}
#@tab mxnet
mu, b = 0.0, 1.0
sigma = np.sqrt(2) * b                            # matched-variance Gaussian sd
U = np.random.rand(200000) - 0.5
lap = mu - b * np.sign(U) * np.log(1 - 2 * np.abs(U))  # inverse transform
gau = np.random.normal(mu, sigma, 200000)
print('Laplace var (2b^2):', float(lap.var().round(3)), ' Gaussian var:', float(gau.var().round(3)))
print('P(|x| > 3sd): Laplace', float((np.abs(lap) > 3 * sigma).mean()),
      ' Gaussian', float((np.abs(gau) > 3 * sigma).mean()))
```

```{.python .input #distributions-laplace}
#@tab pytorch
mu, b = 0.0, 1.0
sigma = np.sqrt(2) * b                            # matched-variance Gaussian sd
lap = torch.distributions.Laplace(mu, b).sample((200000,))
gau = torch.normal(mu, sigma, (200000,))
print('Laplace var (2b^2):', float(lap.var()), ' Gaussian var:', float(gau.var()))
print('P(|x| > 3sd): Laplace', float((lap.abs() > 3 * sigma).float().mean()),
      ' Gaussian', float((gau.abs() > 3 * sigma).float().mean()))
```

```{.python .input #distributions-laplace}
#@tab tensorflow
mu, b = 0.0, 1.0
sigma = np.sqrt(2) * b                            # matched-variance Gaussian sd
U = tf.random.uniform((200000,)) - 0.5
lap = mu - b * tf.sign(U) * tf.math.log(1 - 2 * tf.abs(U))  # inverse transform
gau = tf.random.normal((200000,), mu, sigma)
print('Laplace var (2b^2):', float(tf.math.reduce_variance(lap)),
      ' Gaussian var:', float(tf.math.reduce_variance(gau)))
print('P(|x| > 3sd): Laplace', float(tf.reduce_mean(tf.cast(tf.abs(lap) > 3 * sigma, tf.float32))),
      ' Gaussian', float(tf.reduce_mean(tf.cast(tf.abs(gau) > 3 * sigma, tf.float32))))
```

```{.python .input #distributions-laplace}
#@tab jax
mu, b = 0.0, 1.0
sigma = jnp.sqrt(2.) * b                           # matched-variance Gaussian sd
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
lap = mu + b * jax.random.laplace(key1, (200000,))
gau = mu + sigma * jax.random.normal(key2, (200000,))
print('Laplace var (2b^2):', float(lap.var()), ' Gaussian var:', float(gau.var()))
print('P(|x| > 3sd): Laplace', float((jnp.abs(lap) > 3 * sigma).mean()),
      ' Gaussian', float((jnp.abs(gau) > 3 * sigma).mean()))
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
$\boldsymbol\Sigma^{-1}$, so its level sets---the contours of equal density---are
*ellipsoids*. Because $\boldsymbol\Sigma$ is symmetric PSD it has an orthonormal
eigenbasis $\boldsymbol\Sigma=\mathbf W\boldsymbol\Lambda\mathbf W^\top$
(:numref:`subsec_mdl-spectral-theorem`), and the contour ellipsoid's *axes point
along the eigenvectors $\mathbf w_i$ with half-lengths proportional to
$\sqrt{\lambda_i}$*: the principal directions of spread are the eigenvectors,
stretched by the standard deviations along them. The isotropic case
$\boldsymbol\Sigma=\sigma^2\mathbf I$ gives spherical contours and factorizes into
$d$ independent one-dimensional Gaussians. :numref:`fig_mdl-prob-mvn-contours` draws
this for a $2\times2$ covariance.

![Samples from a $2$-D Gaussian (blue points) with the elliptical density contours overlaid. The ellipse axes lie along the eigenvectors of the covariance $\boldsymbol\Sigma$ with half-lengths proportional to $\sqrt{\lambda_i}$ (the standard deviation along each principal direction), connecting the multivariate Gaussian directly to the spectral theorem.](../img/mdl-prob-mvn-contours.svg)
:label:`fig_mdl-prob-mvn-contours`

It is the default vector noise and prior because it is *closed under linear maps and
conditioning*: a linear transform $\mathbf A\mathbf x+\mathbf b$ of a Gaussian is
again Gaussian, with covariance $\mathbf A\boldsymbol\Sigma\mathbf A^\top$. The cell
draws samples, estimates the covariance, and checks that its eigenvectors recover
the directions we built in.

```{.python .input #distributions-mvn}
#@tab mxnet
mu_v = np.array([0., 0.])
Sigma = np.array([[2., 1.], [1., 2.]])
sample = np.random.multivariate_normal(mu_v, Sigma, size=5000)
emp = np.cov(sample.T)
vals, vecs = np.linalg.eigh(emp)
print('empirical covariance:\n', emp.round(2))
print('eigenvalues (≈ 1, 3):', vals.round(2))   # Sigma has eigenvalues 1 and 3
```

```{.python .input #distributions-mvn}
#@tab pytorch
mu_v = torch.zeros(2)
Sigma = torch.tensor([[2., 1.], [1., 2.]])
sample = torch.distributions.MultivariateNormal(mu_v, Sigma).sample((5000,))
emp = torch.cov(sample.T)
vals, vecs = torch.linalg.eigh(emp)
print('empirical covariance:\n', emp.numpy().round(2))
print('eigenvalues (≈ 1, 3):', vals.numpy().round(2))  # Sigma eigenvalues 1 and 3
```

```{.python .input #distributions-mvn}
#@tab tensorflow
mu_v = np.array([0., 0.])
Sigma = np.array([[2., 1.], [1., 2.]])
sample = np.random.multivariate_normal(mu_v, Sigma, size=5000)
emp = np.cov(sample.T)
vals, vecs = np.linalg.eigh(emp)
print('empirical covariance:\n', emp.round(2))
print('eigenvalues (≈ 1, 3):', vals.round(2))   # Sigma has eigenvalues 1 and 3
```

```{.python .input #distributions-mvn}
#@tab jax
mu_v = jnp.zeros(2)
Sigma = jnp.array([[2., 1.], [1., 2.]])
sample = jax.random.multivariate_normal(jax.random.PRNGKey(0), mu_v, Sigma, (5000,))
emp = jnp.cov(sample.T)
vals, vecs = jnp.linalg.eigh(emp)
print('empirical covariance:\n', np.asarray(emp).round(2))
print('eigenvalues (≈ 1, 3):', np.asarray(vals).round(2))  # Sigma eigenvalues 1 and 3
```

## The Exponential Family

Here is the unifying punchline. Every distribution above---Bernoulli, categorical,
binomial, Poisson, exponential, Gaussian, Laplace (with fixed location)---is a
special case of one form, the *exponential family*. A distribution belongs to it if
its density (or mass) can be written

$$
p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).
$$
:eqlabel:`eq_mdl-exp_family`

The four ingredients each play a clear role. The *base measure* $h(\mathbf x)$ is a
fixed background that does not depend on the parameters. The *natural parameters*
$\boldsymbol\eta$ are the knobs. The *sufficient statistics* $T(\mathbf x)$ are the
only features of the data that matter---given $T(\mathbf x)$, no other detail of
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
$A(\eta)=\log(1+e^\eta)$ is the *softplus*---the link between probabilities and
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

**Proof.** Differentiate :eqref:`eq_mdl-log_partition` under the integral sign.
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

Differentiating once more gives $\nabla^2 A = \operatorname{Cov}(T)$, so $A$ is
*convex* (a covariance matrix is PSD, :numref:`subsec_mdl-psd`)---which is why
exponential-family maximum-likelihood is a convex problem with a unique optimum.
The moment property is the workhorse behind it: setting the model's expected
sufficient statistics equal to their empirical averages, $\mathbb E[T]=\bar T$, *is*
the maximum-likelihood equation. We verify :eqref:`eq_mdl-exp_family_moment` for the
Bernoulli, where it should read $A'(\eta)=\sigma(\eta)=\mathbb E[x]=p$.

```{.python .input #distributions-exp-family}
#@tab mxnet
eta = 0.7                                        # natural parameter (logit)
A = lambda e: np.log1p(np.exp(e))                # Bernoulli log-partition (softplus)
eps = 1e-6
dA = (A(eta + eps) - A(eta - eps)) / (2 * eps)   # numerical dA/deta
print('dA/deta      =', round(float(dA), 6))
print('sigmoid(eta) = E[x] = p =', round(float(1 / (1 + np.exp(-eta))), 6))
```

```{.python .input #distributions-exp-family}
#@tab pytorch
eta = torch.tensor(0.7, requires_grad=True)      # natural parameter (logit)
A = torch.log1p(torch.exp(eta))                  # Bernoulli log-partition (softplus)
A.backward()                                     # autograd: dA/deta
print('dA/deta      =', round(float(eta.grad), 6))
print('sigmoid(eta) = E[x] = p =', round(float(torch.sigmoid(eta)), 6))
```

```{.python .input #distributions-exp-family}
#@tab tensorflow
eta = tf.Variable(0.7)                            # natural parameter (logit)
with tf.GradientTape() as t:
    A = tf.math.log1p(tf.exp(eta))                # Bernoulli log-partition (softplus)
print('dA/deta      =', round(float(t.gradient(A, eta)), 6))
print('sigmoid(eta) = E[x] = p =', round(float(tf.sigmoid(eta)), 6))
```

```{.python .input #distributions-exp-family}
#@tab jax
eta = 0.7                                        # natural parameter (logit)
A = lambda e: jnp.log1p(jnp.exp(e))              # Bernoulli log-partition (softplus)
dA = jax.grad(A)(eta)                            # autodiff: dA/deta
print('dA/deta      =', round(float(dA), 6))
print('sigmoid(eta) = E[x] = p =', round(float(jax.nn.sigmoid(eta)), 6))
```

The derivative of the softplus is the sigmoid, and it lands exactly on the
Bernoulli mean $p$. This is no coincidence: the sigmoid and softmax links that
deep-learning classifiers use are *precisely* the maps from natural parameters to
means in the Bernoulli and categorical families, and the convexity of $A$ is why
their losses are well behaved. The exponential family is thus not an exotic
abstraction but the structural reason the standard losses of deep learning are the
clean objectives they are :cite:`Bishop.2006,Koller.Friedman.2009`.

## Summary

* Distributions form a *family*, not a list: Bernoulli is the seed; summing $n$
  gives the **Binomial** ($\mu=np$, $\sigma^2=np(1-p)$, derived as a sum of
  Bernoullis); the many-rare limit gives the **Poisson** ($\mu=\sigma^2=\lambda$,
  derived from the binomial); the many-ordinary limit (CLT) gives the **Gaussian**.
* The **Categorical** generalizes Bernoulli to $K$ outcomes; its softmax form is
  every classifier's output layer and its NLL is the cross-entropy loss. The
  **Multinomial** counts $n$ categorical draws.
* The **continuous** workhorses: the **uniform** (raw randomness, $U(0,1)$ powers
  sampling), the **exponential** (memoryless waiting times, partner of the
  Poisson), the **Gaussian** (CLT limit, MSE noise model, max-entropy default), the
  **Laplace** ($L_1$/MAE loss, sparsity prior), and the **multivariate Gaussian**,
  whose elliptical contours are the eigendecomposition of its covariance.
* Means and variances follow from elegant structure---linearity of expectation,
  limits, and standard integrals---not memorization.
* The **exponential family** $p(\mathbf x)=h(\mathbf x)\exp(\boldsymbol\eta^\top
  T(\mathbf x)-A(\boldsymbol\eta))$ unifies all of them. Its log-partition $A$
  generates moments: $\nabla A(\boldsymbol\eta)=\mathbb E[T(\mathbf x)]$, and $A$ is
  convex, which is exactly why exponential-family maximum likelihood (the basis of
  the standard deep-learning losses) is a convex problem.

## Exercises

1. Let $X,Y\sim\mathrm{Binomial}(16,1/2)$ be independent. Find the standard
   deviation of $X-Y$. (*Hint:* variances add for independent variables, even under
   subtraction.)
2. Show that the categorical negative log-likelihood $-\sum_k y_k\log p_k$ with
   $p_k$ from the softmax :eqref:`eq_mdl-softmax` has gradient
   $\hat{\mathbf p}-\mathbf y$ with respect to the logits $\mathbf z$. Why does this
   clean form make softmax classification well behaved?
3. Verify that the multinomial pmf :eqref:`eq_mdl-multinomial_pmf` for $n$ trials,
   $K=2$, reduces exactly to the binomial pmf :eqref:`eq_mdl-binomial_pmf`.
4. Prove the exponential distribution is memoryless directly from its density, and
   show that the minimum of independent $\mathrm{Exp}(\lambda_i)$ is again
   exponential with rate $\sum_i\lambda_i$.
5. For $X\sim\mathrm{Poisson}(\lambda)$, argue that $(X-\lambda)/\sqrt\lambda$
   becomes approximately Gaussian as $\lambda\to\infty$. (*Hint:* a
   $\mathrm{Poisson}(\lambda)$ is the sum of $m$ independent
   $\mathrm{Poisson}(\lambda/m)$ variables; apply the CLT.)
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

::: {.slide title="A family, not a list"}
A dozen distributions cover almost everything, and they
connect: Bernoulli is the seed, and a few construction and
limit arrows generate the rest, all inside one envelope.

@fig:mdl-prob-family-tree
:::

::: {.slide title="Setup"}
One per-framework imports cell; the worked cells below evaluate
each law and draw a sample.

@distributions-imports
:::

::: {.slide title="Discrete distributions"}
Mass functions $p_k = P(X=k)$, $\sum_k p_k = 1$:

- **Bernoulli** — one coin flip; binary-classifier output.
- **Categorical** — $K$ outcomes; softmax layer, cross-entropy.
- **Binomial** — count of successes in $n$ Bernoullis.
- **Poisson** — count of rare events.

@fig:mdl-prob-discrete-pmfs
:::

::: {.slide title="Binomial mean/variance, the elegant way"}
$X=\sum_i X_i$ is a sum of $n$ Bernoullis, so linearity and
independence give the moments for free:

$$\mu = \sum_i \mathbb{E}[X_i] = np, \quad
\sigma^2 = \sum_i \mathrm{Var}(X_i) = np(1-p).$$

@distributions-binomial
:::

::: {.slide title="Poisson = the many-rare limit of the binomial"}
Take $\mathrm{Binomial}(n, \lambda/n)$ and let $n\to\infty$:

$$\binom{n}{k}\Bigl(\tfrac{\lambda}{n}\Bigr)^k\Bigl(1-\tfrac{\lambda}{n}\Bigr)^{n-k}
\longrightarrow \frac{\lambda^k e^{-\lambda}}{k!}.$$

Mean $=$ variance $=\lambda$.

@distributions-poisson
:::

::: {.slide title="Continuous distributions"}
Densities $p(x)\ge 0$, $\int p = 1$:

- **Uniform** — raw randomness; $U(0,1)$ powers sampling.
- **Exponential** — memoryless waiting times.
- **Gaussian** — CLT limit; MSE noise model; max entropy.
- **Laplace** — $L_1$/MAE loss; sparsity prior.

@fig:mdl-prob-continuous-pdfs
:::

::: {.slide title="Gaussian: the CLT limit"}
Standardized sums converge to $\mathcal{N}(\mu,\sigma^2)$ for
*any* iid finite-variance summands. Normalizer
$\int e^{-x^2/2}dx=\sqrt{2\pi}$ via the polar trick.

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}.$$

@distributions-gaussian
:::

::: {.slide title="Multivariate Gaussian = covariance geometry"}
Contours are ellipsoids whose axes are the eigenvectors of
$\boldsymbol\Sigma$, with half-lengths $\propto\sqrt{\lambda_i}$:

@fig:mdl-prob-mvn-contours

. . .

@distributions-mvn
:::

::: {.slide title="The exponential family unifies them all"}
Every distribution above fits one form:

$$p(\mathbf{x}\mid\boldsymbol\eta) =
h(\mathbf{x})\exp\bigl(\boldsymbol\eta^\top T(\mathbf{x}) - A(\boldsymbol\eta)\bigr).$$

The log-partition $A$ generates moments:
$\nabla A(\boldsymbol\eta) = \mathbb{E}[T(\mathbf{x})]$,
and $A$ is convex — why MLE losses are clean.

@distributions-exp-family
:::

::: {.slide title="Recap"}
- Bernoulli $\to$ Binomial $\to$ Poisson / Gaussian by sums
  and limits; Categorical $\to$ Multinomial.
- Means/variances from linearity, limits, simple integrals.
- The Gaussian is central (CLT, max entropy).
- The exponential family unifies them; $\nabla A = \mathbb{E}[T]$,
  and its convex $A$ is why deep-learning losses are clean.
:::
