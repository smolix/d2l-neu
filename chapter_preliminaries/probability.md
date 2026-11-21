```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Probability and Statistics
:label:`sec_prob`

Machine learning is all about reasoning under uncertainty: predicting how likely a patient is to suffer a heart attack, judging whether a sensor reading is anomalous, or telling whether an accuracy difference from $0.90$ ($230/256$ test images) to $0.87$ on the next minibatch reflects a real change or random sampling fluctuation.

Probability theory's cleanest thought experiment is still the coin toss: if a coin lands on heads $56$ times in $100$ tosses, should we suspect that it is biased? Even a fair coin rarely lands on heads exactly $50$ times, so we must ask whether a deviation of $6$ heads is larger than chance routinely produces. *Probability* reasons forward from a known model (a fair coin) to the likelihood of outcomes, while *statistics* reasons backward from observed data ($56$ heads in $100$ tosses) to infer properties of the underlying process.

In this section, we simulate coin tosses to watch sampling fluctuations unfold, introduce sample spaces, events, random variables, joint distributions, and independence, and derive the *standard error* $\sigma/\sqrt{n}$ via expectation and variance to settle the coin question. We then develop conditional probability and Bayes' theorem on a two-step medical diagnostic test.

```{.python .input #probability-probability-and-statistics}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #probability-probability-and-statistics}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #probability-probability-and-statistics}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #probability-probability-and-statistics}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #probability-probability-and-statistics-1}
%%tab mxnet, jax
import random
```

```{.python .input #probability-probability-and-statistics-1}
%%tab pytorch
import random
from torch.distributions.multinomial import Multinomial
```

```{.python .input #probability-probability-and-statistics-1}
%%tab tensorflow
import random
from tensorflow_probability import distributions as tfd
```

## Sampling

### Simulated Coin Tosses

For a *fair* coin, symmetry between heads and tails implies $P(\textrm{heads}) = 0.5$. Drawing realizations from a random process is called *sampling*: below, we simulate $100$ tosses first by testing whether uniform draws from $[0, 1)$ exceed $0.5$, and then using our framework's *multinomial* sampler with $n = 100$ and probabilities $(0.5, 0.5)$:

```{.python .input #probability-a-simple-example-tossing-coins-1}
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

:begin_tab:`jax`
In JAX, we split an explicit PRNG key seeded from `random` so each run produces fresh counts.
:end_tab:

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab mxnet
fair_probs = [0.5, 0.5]
mnp.random.multinomial(100, fair_probs)
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab tensorflow
fair_probs = tf.constant([0.5, 0.5])
tfd.Multinomial(100, probs=fair_probs).sample()
```

```{.python .input #probability-a-simple-example-tossing-coins-2}
%%tab jax
key = jax.random.key(random.getrandbits(32))  # A new seed on each run
key, subkey = jax.random.split(key)
fair_probs = jnp.array([0.5, 0.5])
jax.random.multinomial(subkey, 100, fair_probs)
```

### Frequencies and Probabilities

Dividing the observed counts by $n$ gives the empirical *frequency* $\hat{p}$, an *estimator* of the true probability $p = P(\textrm{heads}) = 0.5$. Repeating the 100-toss experiment five times shows $\hat{p}$ fluctuating around $0.5$, whereas increasing the sample size to $10{,}000$ tosses pulls $\hat{p}$ much closer to $0.5$:

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab mxnet
mnp.random.multinomial(100, fair_probs, size=5) / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab pytorch
Multinomial(100, fair_probs).sample((5,)) / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab tensorflow
tfd.Multinomial(100, probs=fair_probs).sample(5) / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-3}
%%tab jax
key, subkey = jax.random.split(key)
jax.random.multinomial(subkey, 100, fair_probs, shape=(5,)) / 100
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab mxnet
mnp.random.multinomial(10000, fair_probs) / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab pytorch
Multinomial(10000, fair_probs).sample() / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab tensorflow
tfd.Multinomial(10000, probs=fair_probs).sample() / 10000
```

```{.python .input #probability-a-simple-example-tossing-coins-4}
%%tab jax
key, subkey = jax.random.split(key)
jax.random.multinomial(subkey, 10000, fair_probs) / 10000
```

Tracking the running frequency after every toss from $1$ to $10{,}000$ via `cumsum` illustrates the *law of large numbers* (:numref:`sec_mdl-statistics`), which guarantees that $\hat{p} \to p$ as $n \to \infty$:

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab mxnet
n = 10000
tosses = (mnp.random.uniform(size=n) < 0.5).astype(mnp.float32)  # 1 is heads
estimates = tosses.cumsum() / mnp.arange(1, n + 1)
d2l.plot(mnp.arange(1, n + 1), [estimates, mnp.full(n, 0.5)], 'tosses',
         'frequency of heads', legend=['running frequency', 'P(heads)'],
         xscale='log', figsize=(4.5, 3))
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab pytorch
n = 10000
tosses = (torch.rand(n) < 0.5).to(torch.float32)  # 1 is heads, 0 is tails
estimates = tosses.cumsum(dim=0) / torch.arange(1, n + 1)
d2l.plot(torch.arange(1, n + 1), [estimates, torch.full((n,), 0.5)],
         'tosses', 'frequency of heads',
         legend=['running frequency', 'P(heads)'], xscale='log',
         figsize=(4.5, 3))
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab tensorflow
n = 10000
tosses = tf.cast(tf.random.uniform((n,)) < 0.5, tf.float32)  # 1 is heads
estimates = tf.cumsum(tosses) / tf.range(1, n + 1, dtype=tf.float32)
d2l.plot(tf.range(1, n + 1), [estimates, tf.fill((n,), 0.5)], 'tosses',
         'frequency of heads', legend=['running frequency', 'P(heads)'],
         xscale='log', figsize=(4.5, 3))
```

```{.python .input #probability-a-simple-example-tossing-coins-5}
%%tab jax
n = 10000
key, subkey = jax.random.split(key)
tosses = jax.random.uniform(subkey, (n,)) < 0.5  # True is heads
estimates = tosses.cumsum() / jnp.arange(1, n + 1)
d2l.plot(jnp.arange(1, n + 1), [estimates, jnp.full(n, 0.5)], 'tosses',
         'frequency of heads', legend=['running frequency', 'P(heads)'],
         xscale='log', figsize=(4.5, 3))
```

To know *how fast* $\hat{p}$ settles—and whether $56/100$ is suspicious—we now formalize random variables and their spread.

## Events and Random Variables

### Sample Spaces and Events

The set of all possible outcomes of a random experiment is its *sample space* $\mathcal{S}$ (for one die roll, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$). An *event* is any subset $\mathcal{A} \subseteq \mathcal{S}$ (such as "rolling an even number," $\mathcal{A} = \{2, 4, 6\}$, or "at least $5$," $\mathcal{B} = \{5, 6\}$). A *probability* function $P$ assigns a real number to every event $\mathcal{A} \subseteq \mathcal{S}$ subject to Kolmogorov's three axioms :cite:`Kolmogorov.1933`:
* **Nonnegativity:** $P(\mathcal{A}) \geq 0$;
* **Normalization:** $P(\mathcal{S}) = 1$;
* **Countable additivity:** for mutually exclusive events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for $i \neq j$), $P(\bigcup_i \mathcal{A}_i) = \sum_i P(\mathcal{A}_i)$.

From these axioms follow the complement rule $P(\mathcal{A}') = 1 - P(\mathcal{A})$, $P(\emptyset) = 0$, and the inclusion--exclusion formula (:numref:`fig_prob_venn`):

$$P(\mathcal{A} \cup \mathcal{B}) = P(\mathcal{A}) + P(\mathcal{B}) - P(\mathcal{A} \cap \mathcal{B}).$$

![Two events $\mathcal{A}$ and $\mathcal{B}$ in a sample space $\mathcal{S}$. The sum $P(\mathcal{A}) + P(\mathcal{B})$ counts the overlap $\mathcal{A} \cap \mathcal{B}$ twice.](../img/probability-venn.svg)
:label:`fig_prob_venn`

### Random Variables and Distributions

A *random variable* $X$ maps outcomes in $\mathcal{S}$ to numerical values (for example, the number of heads in $100$ coin tosses maps $2^{100}$ sequences to $\{0, \ldots, 100\}$). A *discrete* random variable assigns probability mass $P(X = v)$ to each value $v$ (its *distribution* $P(X)$). A *continuous* random variable (such as a person's exact height) is described instead by a *probability density* $p(x)$ (:numref:`fig_prob_density`), where the probability of landing in $[a, b]$ is the area under the density curve, $P(a \leq X \leq b) = \int_a^b p(x)\,dx$ (:numref:`sec_mdl-random_variables`).

![A discrete random variable puts probability on individual values (left). A continuous random variable has a density, and the probability of an interval is the area under the density (right).](../img/probability-density.svg)
:label:`fig_prob_density`

### Joint, Marginal, and Independent Distributions

The *joint distribution* $P(X = x, Y = y)$ of two random variables gives the probability that $X = x$ and $Y = y$ occur together. For example, for rain ($X$) and carrying an umbrella ($Y$) on a random day:

| $P(X, Y)$ | umbrella | no umbrella |
|:--|--:|--:|
| rain | 0.25 | 0.05 |
| no rain | 0.10 | 0.60 |

Summing out one variable recovers the *marginal distribution* of the other (*marginalization*, :numref:`fig_prob_joint`):

$$P(X = x) = \sum_y P(X = x, Y = y),$$

so $P(\textrm{rain}) = 0.25 + 0.05 = 0.30$ and $P(\textrm{umbrella}) = 0.25 + 0.10 = 0.35$.

![A joint distribution $P(X, Y)$ as a table. Summing a row or a column gives a marginal distribution. Dividing a row by its sum gives a conditional distribution, here $P(Y \mid X = x_2)$.](../img/probability-joint-grid.svg)
:label:`fig_prob_joint`

Two random variables $X$ and $Y$ are *independent* ($X \perp Y$) if their joint distribution factorizes into the product of their marginals:

$$P(X, Y) = P(X) P(Y).$$

Successive coin tosses are independent, whereas rain and umbrellas are not ($0.25 \neq 0.30 \cdot 0.35 = 0.105$).

## Expectation and Variance

### Expectation

The *expectation* (or *mean*) of a discrete random variable $X$ is its probability-weighted average:

$$\mathbb{E}[X] = \sum_x x \, P(X = x),$$

and for any function $f$, $\mathbb{E}[f(X)] = \sum_x f(x)\,P(X = x)$. For an investment returning $0\times$ with probability $0.5$, $2\times$ with probability $0.4$, and $10\times$ with probability $0.1$, $\mathbb{E}[X] = 0.5(0) + 0.4(2) + 0.1(10) = 1.8$. For a coin-toss indicator $X = \mathbf{1}(\textrm{heads}) \in \{0, 1\}$, $\mathbb{E}[X] = 1\cdot p + 0\cdot(1-p) = p$.

Expectation is always *linear*—$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$ whether or not $X$ and $Y$ are independent—so the sample frequency $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$ satisfies $\mathbb{E}[\hat{p}] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = p$ ($\hat{p}$ is an *unbiased* estimator of $p$).

### Variance and Sums of Independent Variables

To measure spread around $\mu = \mathbb{E}[X]$, the *variance* averages the squared deviations, and the *standard deviation* $\sigma = \sqrt{\textrm{Var}(X)}$ restores the original units:

$$\textrm{Var}(X) = \mathbb{E}\left[(X - \mu)^2\right] = \mathbb{E}[X^2] - \mu^2, \qquad \textrm{Var}(aX + b) = a^2\,\textrm{Var}(X).$$

For our investment, $\mathbb{E}[X^2] = 0.5(0) + 0.4(4) + 0.1(100) = 11.6$, so $\textrm{Var}(X) = 11.6 - 1.8^2 = 8.36$ ($\sigma \approx 2.9$). For a single coin-toss indicator $X \in \{0, 1\}$, $X^2 = X$, so $\textrm{Var}(X) = p - p^2 = p(1 - p)$ (maximized at $p = 0.5$, where $\textrm{Var}(X) = 1/4$). (When estimating variance from $n$ sample points around their sample mean as in :numref:`sec_pandas`, dividing by $n - 1$—Bessel's correction—corrects for the slight underestimation of dividing by $n$.)

For a sum of two random variables, expanding $\mathbb{E}[(X - \mu_X + Y - \mu_Y)^2]$ gives

$$\textrm{Var}(X + Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\,\textrm{Cov}(X, Y), \qquad \textrm{Cov}(X, Y) = \mathbb{E}\left[(X - \mu_X)(Y - \mu_Y)\right],$$

where the *covariance* $\textrm{Cov}(X, Y)$ (and dimensionless *correlation* $\rho = \textrm{Cov}(X, Y)/(\sigma_X\sigma_Y) \in [-1, 1]$) measures linear co-movement (:numref:`fig_prob_covariance`; for a random vector $\mathbf{x}$, the *covariance matrix* $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^\top]$ satisfies $\textrm{Var}(\mathbf{v}^\top\mathbf{x}) = \mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v}$).

![Samples of three pairs of random variables, with positive, near-zero, and negative covariance.](../img/probability-covariance.svg)
:label:`fig_prob_covariance`

When $X$ and $Y$ are *independent*, $\textrm{Cov}(X, Y) = \mathbb{E}[X - \mu_X]\mathbb{E}[Y - \mu_Y] = 0$ (though zero covariance does not imply independence, e.g., $Y = X^2$ for $X \in \{-1, 0, 1\}$), so **variances of independent variables add**:

$$\textrm{Var}(X + Y) = \textrm{Var}(X) + \textrm{Var}(Y).$$

## Estimation Error

### The Standard Error of a Frequency

Because $n$ independent coin tosses each have variance $p(1 - p)$, their sum has variance $np(1 - p)$, and dividing by $n$ scales the variance by $1/n^2$:

$$\textrm{Var}(\hat{p}) = \frac{p(1-p)}{n}.$$
:eqlabel:`eq_prob_var_phat`

Taking the square root gives the *standard error* of $\hat{p}$: $\sqrt{p(1-p)/n}$, which for a fair coin ($p = 0.5$) is $0.5/\sqrt{n}$ ($0.05$ at $n = 100$; $0.005$ at $n = 10{,}000$). Because of the $1/\sqrt{n}$ rate, halving the error requires $4\times$ the data. More generally, the average of any $n$ independent draws with variance $\sigma^2$ has standard error $\sigma/\sqrt{n}$. Simulating $1000$ repetitions at each $n \in \{10, 100, 1000, 10000\}$ confirms the $0.5/\sqrt{n}$ line of slope $-1/2$:

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab mxnet
ns = [10, 100, 1000, 10000]
stds = [float((mnp.random.uniform(size=(1000, n)) < 0.5)
              .astype(mnp.float32).mean(axis=1).std()) for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'tosses n',
         'std of frequency', legend=['simulation', r'$0.5/\sqrt{n}$'],
         xscale='log', yscale='log', fmts=('o', '-'))
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab pytorch
ns = [10, 100, 1000, 10000]
stds = [float((torch.rand(1000, n) < 0.5).to(torch.float32)
              .mean(dim=1).std()) for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'tosses n',
         'std of frequency', legend=['simulation', r'$0.5/\sqrt{n}$'],
         xscale='log', yscale='log', fmts=('o', '-'))
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab tensorflow
ns = [10, 100, 1000, 10000]
stds = [float(tf.math.reduce_std(tf.reduce_mean(
    tf.cast(tf.random.uniform((1000, n)) < 0.5, tf.float32), axis=1)))
    for n in ns]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'tosses n',
         'std of frequency', legend=['simulation', r'$0.5/\sqrt{n}$'],
         xscale='log', yscale='log', fmts=('o', '-'))
```

```{.python .input #probability-a-simple-example-tossing-coins-6}
%%tab jax
ns = [10, 100, 1000, 10000]
key, *subkeys = jax.random.split(key, len(ns) + 1)
stds = [float((jax.random.uniform(k, (1000, n)) < 0.5).mean(axis=1).std())
        for k, n in zip(subkeys, ns)]
d2l.plot(ns, [stds, [0.5 / n ** 0.5 for n in ns]], 'tosses n',
         'std of frequency', legend=['simulation', r'$0.5/\sqrt{n}$'],
         xscale='log', yscale='log', fmts=('o', '-'))
```

### Testing for Bias and Uncertainty

We can now answer our coin question. In $100$ tosses, $\hat{p} = 0.56$ is $0.06 / 0.05 = 1.2$ standard errors from $0.5$; in $10{,}000$ tosses, the same frequency $\hat{p} = 0.56$ ($5{,}600$ heads) is $0.06 / 0.005 = 12$ standard errors away. Simulating $1000$ fair-coin experiments shows how often a fair coin strays at least that far (estimating the *p-value*):

```{.python .input #probability-is-the-coin-biased}
%%tab mxnet
def fraction_as_far(heads, n, num_experiments=1000):
    """Fraction of fair-coin experiments at least as far from n/2 as heads."""
    tosses = mnp.random.uniform(size=(num_experiments, n)) < 0.5
    counts = tosses.astype(mnp.float32).sum(axis=1)
    return (mnp.abs(counts - n / 2) >= abs(heads - n / 2)).mean()

fraction_as_far(56, 100), fraction_as_far(5600, 10000)
```

```{.python .input #probability-is-the-coin-biased}
%%tab pytorch
def fraction_as_far(heads, n, num_experiments=1000):
    """Fraction of fair-coin experiments at least as far from n/2 as heads."""
    counts = (torch.rand(num_experiments, n) < 0.5).sum(dim=1)
    far = (counts - n / 2).abs() >= abs(heads - n / 2)
    return far.to(torch.float32).mean()

fraction_as_far(56, 100), fraction_as_far(5600, 10000)
```

```{.python .input #probability-is-the-coin-biased}
%%tab tensorflow
def fraction_as_far(heads, n, num_experiments=1000):
    """Fraction of fair-coin experiments at least as far from n/2 as heads."""
    tosses = tf.random.uniform((num_experiments, n)) < 0.5
    counts = tf.reduce_sum(tf.cast(tosses, tf.float32), axis=1)
    far = tf.abs(counts - n / 2) >= abs(heads - n / 2)
    return tf.reduce_mean(tf.cast(far, tf.float64))  # no float32 print noise

float(fraction_as_far(56, 100)), float(fraction_as_far(5600, 10000))
```

```{.python .input #probability-is-the-coin-biased}
%%tab jax
def fraction_as_far(key, heads, n, num_experiments=1000):
    """Fraction of fair-coin experiments at least as far from n/2 as heads."""
    counts = (jax.random.uniform(key, (num_experiments, n)) < 0.5).sum(axis=1)
    far = jnp.abs(counts - n / 2) >= abs(heads - n / 2)
    return far.sum().item() / num_experiments

key, k1, k2 = jax.random.split(key, 3)
fraction_as_far(k1, 56, 100), fraction_as_far(k2, 5600, 10000)
```

Roughly a quarter of fair 100-toss runs deviate by $6$ or more heads, so $56/100$ is no evidence of bias, whereas zero out of $1000$ fair 10,000-toss runs come anywhere near $5{,}600$. By *Chebyshev's inequality* (:numref:`sec_mdl-random_variables`), $P(|X - \mu| \geq k\sigma) \leq 1/k^2$ for any distribution, and by the *central limit theorem* (:numref:`sec_mdl-distributions`), sample averages follow an approximately normal bell curve landing within $2$ standard errors roughly $95\%$ of the time. Likewise, a classifier with true accuracy $0.9$ has standard error $\sqrt{0.9 \cdot 0.1 / 256} \approx 0.019$ on a minibatch of $256$ images (so $\pm 2\%$ swings are routine noise), versus $0.003$ on a test set of $10{,}000$ images.

Notice finally that collecting more data shrinks *epistemic uncertainty* (uncertainty about the model parameter $p$, decaying as $1/\sqrt{n}$), but never reduces *aleatoric uncertainty* (the intrinsic 50/50 randomness of the next coin toss itself, or an inherently ambiguous image) :cite:`Der-Kiureghian.Ditlevsen.2009`.

## Conditional Probability

### Conditioning, Bayes' Theorem, and Conditional Independence

Restricting attention to outcomes where $X = x$ ($P(X = x) > 0$) and renormalizing defines the *conditional probability* of $Y = y$ given $X = x$:

$$P(Y = y \mid X = x) = \frac{P(X = x, Y = y)}{P(X = x)} \quad \Longrightarrow \quad P(X, Y) = P(Y \mid X) P(X).$$

In our weather table, conditioning on rain rescales the rain row $(0.25, 0.05)$ by $0.30$ to give $P(\textrm{umbrella} \mid \textrm{rain}) = 0.25/0.30 \approx 0.83$ (versus $0.35$ unconditionally). Writing the joint probability in both orders, $P(X, Y) = P(Y \mid X)P(X) = P(X \mid Y)P(Y)$, and dividing by $P(Y) = \sum_x P(Y \mid X = x)P(X = x)$ yields *Bayes' theorem* :cite:`Bayes.1763`:

$$P(X \mid Y) = \frac{P(Y \mid X) P(X)}{P(Y)}.$$

For a hypothesis $H$ and observed data $D$, Bayes' theorem updates the *prior* $P(H)$ via the *likelihood* $P(D \mid H)$ and evidence $P(D)$ to produce the *posterior* $P(H \mid D) \propto P(D \mid H)P(H)$.

Two variables $X$ and $Y$ are *conditionally independent* given $Z$ ($X \perp Y \mid Z$) if $P(X, Y \mid Z) = P(X \mid Z)P(Y \mid Z)$. Conditioning can either *remove* dependence (as when conditioning on a common cause like age removes the correlation between shoe size and reading level, or conditioning on the present state in a Markov chain) or *create* dependence (conditioning on a common effect—*explaining away*—as when knowing a hospitalized patient has a broken bone lowers the probability they were admitted for lung cancer; :numref:`fig_prob_explaining_away`).

![Conditioning changes dependence. A common cause $Z$ makes $X$ and $Y$ dependent, and observing $Z$ removes the dependence. In a chain, $X$ and $Z$ are independent given $Y$. A common effect $Z$ makes its independent causes dependent once $Z$ is observed.](../img/probability-explaining-away.svg)
:label:`fig_prob_explaining_away`

### Example: Diagnostic Testing
:label:`subsec_probability_hiv_app`

Suppose a patient takes an HIV screening test $D_1 \in \{0, 1\}$ with true HIV status $H \in \{0, 1\}$. The test has $100\%$ sensitivity and a $1\%$ false-positive rate:

| $P(D_1 \mid H)$ | $H=1$ (HIV) | $H=0$ (healthy) |
|:--|--:|--:|
| test positive, $D_1 = 1$ | 1.00 | 0.01 |
| test negative, $D_1 = 0$ | 0.00 | 0.99 |

If the population prevalence is $P(H = 1) = 0.0015$ ($P(H = 0) = 0.9985$) and the patient tests positive ($D_1 = 1$), marginalizing gives $P(D_1 = 1) = 0.01(0.9985) + 1(0.0015) = 0.011485$, and Bayes' theorem gives

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = \frac{0.0015}{0.011485} = 0.1306.$$

Out of $10{,}000$ people (:numref:`fig_prob_natural_freq`), all $15$ infected people test positive, but so do $1\%$ of the $9{,}985$ healthy people ($\approx 100$ false alarms), leaving only $15 / 115 \approx 13.06\%$ of positive tests truly infected!

![The HIV test applied to 10,000 people. The 15 infected people test positive, and so do about 100 of the 9,985 healthy people, so only 15 of about 115 positive results are correct.](../img/probability-natural-frequencies.svg)
:label:`fig_prob_natural_freq`

Now suppose a follow-up test $D_2$ (with $P(D_2 = 1 \mid H = 1) = 0.98$ and false-positive rate $P(D_2 = 1 \mid H = 0) = 0.03$) also comes back positive. Assuming $D_1$ and $D_2$ are conditionally independent given $H$, the joint false-positive rate drops to $0.01 \cdot 0.03 = 0.0003$ while sensitivity is $1 \cdot 0.98 = 0.98$. Marginalizing gives $P(D_1 = 1, D_2 = 1) = 0.0003(0.9985) + 0.98(0.0015) = 0.00176955$, and Bayes' theorem raises the posterior probability of infection to

$$P(H = 1 \mid D_1 = 1, D_2 = 1) = \frac{0.98 \cdot 0.0015}{0.00176955} = 0.8307$$

(:numref:`fig_prob_update`). Simulating two million virtual patients confirms $0.8307$ (with standard error $\sqrt{0.83 \cdot 0.17 / 3500} \approx 0.006$ over the $\approx 3{,}500$ double-positive patients):

![The probability that the patient has HIV: the prior, after one positive test, and after two conditionally independent positive tests.](../img/probability-bayes-update.svg)
:label:`fig_prob_update`

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab mxnet
n = 2000000
H = mnp.random.uniform(size=n) < 0.0015
D1 = mnp.random.uniform(size=n) < mnp.where(H, 1.00, 0.01)
D2 = mnp.random.uniform(size=n) < mnp.where(H, 0.98, 0.03)
H[D1 & D2].astype(mnp.float32).mean()  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab pytorch
n = 2000000
H = torch.rand(n) < 0.0015
D1 = torch.rand(n) < torch.where(H, 1.00, 0.01)
D2 = torch.rand(n) < torch.where(H, 0.98, 0.03)
H[D1 & D2].to(torch.float32).mean()  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab tensorflow
n = 2000000
H = tf.random.uniform((n,)) < 0.0015
D1 = tf.random.uniform((n,)) < tf.where(H, 1.00, 0.01)
D2 = tf.random.uniform((n,)) < tf.where(H, 0.98, 0.03)
fraction = tf.reduce_mean(tf.cast(H[D1 & D2], tf.float32))
round(float(fraction), 4)  # Exact value: 0.8307
```

```{.python .input #probability-worked-example-hiv-testing-1}
%%tab jax
n = 2000000
key, k1, k2, k3 = jax.random.split(key, 4)
H = jax.random.uniform(k1, (n,)) < 0.0015
D1 = jax.random.uniform(k2, (n,)) < jnp.where(H, 1.00, 0.01)
D2 = jax.random.uniform(k3, (n,)) < jnp.where(H, 0.98, 0.03)
H[D1 & D2].mean()  # Exact value: 0.8307
```

## Summary

* *Probability* reasons forward from a model to data; *statistics* reasons backward from data to infer the underlying model.
* Kolmogorov's three axioms generate all probability rules. Joint distributions $P(X, Y)$ marginalize by summing out variables, and factorize as $P(X)P(Y)$ when $X$ and $Y$ are independent.
* Expectation $\mathbb{E}[X]$ is always linear, while variance $\textrm{Var}(X) = \mathbb{E}[(X-\mu)^2]$ is additive for independent variables, giving an average of $n$ independent draws standard error $\sigma/\sqrt{n}$.
* Measuring deviations in standard errors separates signal from noise ($56/100$ heads is $1.2$ standard errors; $5600/10000$ is $12$). More data reduces *epistemic* uncertainty as $1/\sqrt{n}$ but leaves *aleatoric* uncertainty unchanged.
* Bayes' theorem $P(H \mid D) = P(D \mid H)P(H)/P(D)$ reverses conditioning: for a disease with $0.15\%$ prevalence, one positive test ($1\%$ false-positive rate) yields a $13.1\%$ posterior, while a second conditionally independent positive test ($3\%$ false-positive rate) raises it to $83.1\%$.

## Exercises

1. **Portfolio return and risk.** Assume that you manage a portfolio of
   $n$ stocks whose returns $\mathbf{s} \in \mathbb{R}^n$ are random with
   mean $\boldsymbol{\mu} = \mathbb{E}[\mathbf{s}]$ and covariance matrix
   $\boldsymbol{\Sigma}$. You invest a fraction $\alpha_i \geq 0$ of your
   capital in stock $i$, with $\sum_i \alpha_i = 1$.
    1. Compute the expected return and the variance of the return of the
       portfolio $\boldsymbol{\alpha}$.
    1. Assume that every stock has variance $\sigma^2$ and that every pair
       of distinct stocks has correlation $\rho \geq 0$. Compute the
       variance of the equally weighted portfolio $\alpha_i = 1/n$ and its
       limit as $n \to \infty$. Compare the case $\rho = 0$
       with :eqref:`eq_prob_var_phat`.
    1. (Optional.) Formulate an optimization problem that maximizes the
       expected return while constraining the variance to an upper bound.
       This is the
       [Markowitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model)
       problem :cite:`Mangram.2013`. Methods for constrained problems of
       this kind are developed
       in :numref:`sec_mdl-constrained-optimization-duality`.

1. [code] **Chebyshev for the coin-toss estimator.**
   :eqref:`eq_prob_var_phat` gives the variance of the frequency $\hat{p}$
   of heads after $n$ tosses.
    1. For a fair coin and $n = 100$, use Chebyshev's inequality to bound
       $P(|\hat{p} - p| \geq 0.1)$.
    1. Estimate the same probability by simulating 10,000 repetitions of
       100 tosses, compare it with the bound, and account for the gap.
    1. Determine how many tosses Chebyshev's inequality requires to
       guarantee $P(|\hat{p} - p| \geq 0.01) \leq 0.05$ for a fair coin,
       and compare with the $n$ at which the standard error
       $\sqrt{p(1-p)/n}$ itself equals $0.01$.

1. **Comparing two classifiers.** ● Two classifiers are evaluated on the same
   test set of $n = 10{,}000$ images. Classifier A labels 9,050 images
   correctly and classifier B labels 9,000 correctly. Treat the test images
   as independent draws.
    1. Compute the standard error of each accuracy.
    1. Assume that the two accuracies are independent and compute the
       standard error of their difference. Is the difference of $0.005$
       large compared with the noise?
    1. The two accuracies are measured on the same images, so they are not
       independent. Suppose that the classifiers disagree on 300 images,
       A being correct on 175 of them and B on 125. For image $i$, let
       $Z_i = 1$ if only A is correct, $Z_i = -1$ if only B is correct, and
       $Z_i = 0$ otherwise, so that the difference in accuracy is the average
       of the $Z_i$. Compute the standard error of this average and compare
       it with your answer to the previous part.
    1. Use $\textrm{Var}(X - Y) = \textrm{Var}(X) + \textrm{Var}(Y) - 2\,\textrm{Cov}(X, Y)$
       to explain the difference. What sign does the covariance between the
       two classifiers' correctness on an image have, and why?

1. **Factorizing a chain.** Let $X$, $Y$, and $Z$ be random variables such
   that $Z$ is conditionally independent of $X$ given $Y$, as in the chain
   of :numref:`fig_prob_explaining_away`.
    1. Write the joint probability $P(X, Y, Z)$ as a product of factors that
       each involve at most two of the variables.
    1. Generalize to a sequence $X_1, \ldots, X_T$ in which each $X_t$
       depends on the earlier variables only through $X_{t-1}$. How many
       numbers specify the joint distribution if every $X_t$ takes $k$
       values, compared with an arbitrary joint distribution of $T$ such
       variables?

1. **Conditioning on a common effect.** Toss two independent fair coins, and
   let $X, Y \in \{0, 1\}$ indicate heads on each. Let $Z = \max(X, Y)$
   indicate that at least one coin came up heads. Compute $P(X = 1)$,
   $P(X = 1 \mid Z = 1)$, and $P(X = 1 \mid Z = 1, Y = 1)$. Use these values
   to decide whether $X$ and $Y$ are conditionally independent given
   $Z = 1$, and relate the change from the second value to the third
   to :numref:`fig_prob_explaining_away`.

1. **A second, correlated test.** ● Revisit
   :numref:`subsec_probability_hiv_app` with two new tests whose outcomes
   are conditionally independent for patients with HIV but not for
   healthy patients. Either test on its own has a false positive rate of
   10% and a false negative rate of 1%:

   $$P(D_i = 1 \mid H = 0) = 0.1, \quad P(D_i = 0 \mid H = 1) = 0.01 \quad \textrm{for } i = 1, 2.$$

   For $H = 1$, the joint distribution factorizes as
   $P(D_1, D_2 \mid H = 1) = P(D_1 \mid H = 1)\, P(D_2 \mid H = 1)$. For
   $H = 0$, the outcomes are coupled through
   $P(D_1 = D_2 = 1 \mid H = 0) = 0.02$. The baseline probability is
   $P(H = 1) = 0.0015$ as before.
    1. Work out the joint probability table for $D_1$ and $D_2$ given
       $H = 0$.
    1. Derive the probability that the patient has HIV after the first
       test returns positive, before the second test is run.
    1. Derive the probability that the patient has HIV after both tests
       return positive.
    1. Repeat the previous calculation for tests that are conditionally
       independent given $H = 0$ as well, and explain why the two
       posteriors differ.

1. **A defense lawyer's statistics.** A suspect had a documented history of
   abusing his now-murdered wife. His lawyer argues that, statistically,
   only one in a thousand wife-abusers goes on to murder his wife, so the
   history of abuse is weak evidence and the jury should acquit. Write
   $M$ for "the suspect murdered his wife," $K$ for "his wife was killed,"
   and $B$ for "he had a history of abusing her."
    1. Express the lawyer's one-in-a-thousand figure and the posterior
       probability of guilt in these terms, and identify which probability
       the lawyer has confused with which.
    1. Assume that, over the time span the lawyer's statistic refers to, an
       abused wife is murdered by someone other than her husband with
       probability $10^{-5}$. Compute the posterior probability of guilt and
       compare it with the lawyer's figure.

    *Adapted from MIT 18.05,
    [problem set 7](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_pset07_sol.pdf),
    Problem 4.*

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/198)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17971)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.6]{.kicker}

Evidence and error bars for measured numbers<br>
**sampling · random variables · expectation · the standard error · Bayes' theorem**
:::
:::

::: {.slide title="Every Number Computed From Data Fluctuates"}
[The problem]{.kicker}

- A coin lands heads **56** times in 100 tosses. Is it biased?
- A classifier labels **230 of 256** images correctly (0.90). The next
  minibatch gives 0.87. Did anything change?
- Even a fair coin rarely gives exactly 50 heads.

::: {.d2l-note}
**Probability:** model → data (how likely is 56 for a fair coin?).
**Statistics:** data → model (what do 56 heads say about the coin?).
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read marginals off a joint table and test two variables for independence;
- compute expectations, variances, and the variance of a sum;
- attach a standard error $\sigma/\sqrt{n}$ to an average, and decide
  whether 56 heads in 100 tosses is evidence of bias;
- reverse a conditional with Bayes' theorem: why a positive HIV test
  means only 13%.

**Running examples:** a coin tossed $n$ times, a classifier's accuracy on
a minibatch of 256 images, and a two-test HIV diagnosis.
:::

::: {.slide title="Probability Turns Measurements Into Evidence"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | tool |
|---|---|
| Data Manipulation | tensors |
| Data Preprocessing | pandas |
| Linear Algebra | matrix products, norms, eigenvalues |
| Calculus | derivatives, gradients, chain rule |
| Automatic Differentiation | autograd |
| **Probability and Statistics** | **standard error, Bayes' theorem** |
| Documentation | `help`, small tests |
:::

::: {.col .narrow}
- **Uses:** tensors and random sampling from the earlier decks.
- **Skips:** densities in depth, named distributions, maximum likelihood,
  and the proofs, all in the mathematics appendix.
:::
:::
:::

::: {.slide title="Frequencies Fluctuate Around the Probability"}
[Sampling]{.kicker}

Frequencies of heads and tails in 100 tosses of a fair coin:

@probability-a-simple-example-tossing-coins-3

In 10,000 tosses:

@!probability-a-simple-example-tossing-coins-4

- The **probability** $p = 0.5$ belongs to the coin. The **frequency**
  $\hat{p}$ is an estimate: it changes with every repetition.
:::

::: {.slide title="The Running Frequency Settles Toward 0.5"}
[Sampling]{.kicker}

::: {.cols .vc}
::: {.col}
- Heads so far divided by tosses so far, after each toss.
- **Law of large numbers:** $\hat{p} \to p$ as $n \to \infty$.

::: {.d2l-note}
It does not say **how fast**. How far from $0.5$ may 100 tosses land?
:::
:::

::: {.col .fig .big}
@!probability-a-simple-example-tossing-coins-5
:::
:::
:::

::: {.slide title="Three Axioms Generate Every Rule"}
[The rules]{.kicker}

::: {.cols .vc}
::: {.col}
An **event** is a subset of the sample space $\mathcal{S}$.

- $P(\mathcal{A}) \ge 0$, and $P(\mathcal{S}) = 1$;
- mutually exclusive events **add**.

Consequences: $P(\mathcal{A}') = 1 - P(\mathcal{A})$ and
$P(\mathcal{A}\cup\mathcal{B}) = P(\mathcal{A}) + P(\mathcal{B}) - P(\mathcal{A}\cap\mathcal{B})$.

Die, "even or at least 5":
$$\frac{3}{6} + \frac{2}{6} - \frac{1}{6} = \frac{2}{3}.$$
:::

::: {.col .fig}
@fig:probability-venn
:::
:::
:::

::: {.slide title="A Random Variable Maps Outcomes to Values"}
[The rules]{.kicker}

::: {.cols .vc}
::: {.col}
- Heads in 100 tosses: $2^{100}$ outcomes, only **101** values.
- **Distribution** $P(X)$: the probability of each value.
- A height has a **density**: $P(a \le X \le b) = \int_a^b p(x)\,dx$;
  each exact value has probability zero.
:::
::: {.col .fig .big}
@fig:probability-density
:::
:::
:::

::: {.slide title="A Joint Table Holds Everything About Two Variables"}
[The rules]{.kicker}

::: {.cols .vc}
::: {.col}
Rain and umbrella on a random day:

| $P(X, Y)$ | umbrella | no umbrella |
|:--|:--:|:--:|
| rain | 0.25 | 0.05 |
| no rain | 0.10 | 0.60 |

- **Marginal:** sum a row or a column.
  $P(\textrm{rain}) = 0.30$, $P(\textrm{umbrella}) = 0.35$.
- **Independent** means $P(X, Y) = P(X)\,P(Y)$. Here
  $0.25 \neq 0.30 \cdot 0.35$: not independent.
- Two coin tosses are independent; the coin question needs this.
:::

::: {.col .fig}
@fig:probability-joint-grid
:::
:::
:::

::: {.slide title="Expectation Is the Probability-Weighted Average"}
[Expectation and variance]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}[X] = \sum_x x\,P(X{=}x)$$

- Investment: $0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
- Indicator of heads: $\mathbb{E}[X] = p$.
- **Linear**, always: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, so
  $\mathbb{E}[\hat{p}] = p$: the frequency is **unbiased**.
:::

::: {.col .fig}
@fig:probability-expectation
:::
:::
:::

::: {.slide title="Variance Measures the Spread Around the Mean"}
[Expectation and variance]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] = \mathbb{E}[X^2] - \mu^2$$

- Investment: $11.6 - 1.8^2 = 8.36$, so $\sigma \approx 2.9$.
- Indicator of heads: $p - p^2 = p(1-p)$, at most $\tfrac14$.
- $\textrm{Var}(aX + b) = a^2\,\textrm{Var}(X)$.
:::

::: {.col .fig}
@fig:probability-spread
:::
:::
:::

::: {.slide title="Variances of Independent Variables Add"}
[Expectation and variance · derivation]{.kicker}

**Step 1.** Expand the square:
$\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\,\textrm{Cov}(X,Y)$,
with $\textrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$.

**Step 2.** Independent: $\textrm{Cov}(X,Y) = \mathbb{E}[X-\mu_X]\,\mathbb{E}[Y-\mu_Y] = 0$.

::: {.d2l-note .rule}
$\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$: the sum of $n$
independent draws has standard deviation $\sqrt{n}\,\sigma$, not $n\sigma$.
:::
:::

::: {.slide title="The Standard Error of a Frequency Is 0.5/√n"}
[Estimation error]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Var}(\hat{p}) = \frac{1}{n^2} \cdot n\,p(1-p) = \frac{p(1-p)}{n}$$

- Fair coin: $0.05$ at $n = 100$, $0.005$ at $n = 10{,}000$.
- 1000 repeats per $n$ land on the line: **slope $-\tfrac12$**.
- Half the error costs **4×** the data.
:::

::: {.col .fig .big}
@!probability-a-simple-example-tossing-coins-6
:::
:::
:::

::: {.slide title="56 of 100 Is No Evidence of Bias; 5,600 of 10,000 Is"}
[Estimation error]{.kicker}

- $56/100$: deviation $0.06 = 1.2$ standard errors.
- $5600/10000$: deviation $0.06 = 12$ standard errors.

Fraction of 1000 fair-coin experiments at least this far from $n/2$, an
estimate of the **p-value**:

@!probability-is-the-coin-biased

::: {.d2l-note .rule}
Chebyshev, any distribution: $P(|X - \mu| \ge k\sigma) \le 1/k^2$.
Averages do better: within 2 standard errors about 95% of the time.
Under 2 is unremarkable; 3 or more is hard to blame on chance.
:::
:::

::: {.slide title="A Minibatch of 256 Images Measures Accuracy to About ±2 Points"}
[Estimation error]{.kicker}

Correct label with probability $0.9$, independently per image:

- minibatch of 256: $\sqrt{0.9 \cdot 0.1 / 256} \approx 0.019$;
- test set of 10,000: $\sqrt{0.9 \cdot 0.1 / 10{,}000} = 0.003$;
- the difference of two such accuracies: $\sqrt{2} \cdot 0.003 \approx 0.004$,
  since variances add, so one point is more than two standard errors.

::: {.d2l-note}
In general: an average of $n$ independent draws has standard error
$\sigma/\sqrt{n}$.
:::
:::

::: {.slide title="More Data Removes Only Epistemic Uncertainty"}
[Estimation error]{.kicker}

::: {.cols .vc}
::: {.col}
- **Epistemic:** uncertainty about the model, such as $p$. It shrinks
  like $1/\sqrt{n}$.
- **Aleatoric:** randomness of the outcome. Knowing $p = 0.5$ exactly,
  the next toss is still 50/50; an ambiguous image stays ambiguous.
:::

::: {.col .fig .big}
@fig:probability-uncertainty
:::
:::
:::

::: {.slide title="Check: How Large a Test Set?"}
[Check]{.kicker}

A classifier has accuracy about $0.9$. How many test images make the
standard error of its measured accuracy $0.005$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\sqrt{0.9 \cdot 0.1 / n} = 0.005$ gives
$n = 0.09 / 0.000025 = 3{,}600$. Halving the error to $0.0025$ takes four
times as many: 14,400.
:::
:::
:::

::: {.slide title="Conditioning Keeps One Row and Renormalizes It"}
[Conditional probability]{.kicker}

::: {.cols .vc}
::: {.col}
$$P(Y \mid X{=}x) = \frac{P(X{=}x,\, Y)}{P(X{=}x)}$$

- Rain row $(0.25, 0.05)$ becomes $(0.83, 0.17)$:
  $P(\textrm{umbrella} \mid \textrm{rain}) = 0.83$ against $0.35$ on an
  arbitrary day.
- In the figure, $P(Y{=}y_3 \mid X{=}x_2) = 0.25 / 0.60 \approx 0.42$.
- Multiplication rule: $P(X, Y) = P(Y \mid X)\,P(X)$.
:::

::: {.col .fig}
@fig:probability-joint-grid
:::
:::
:::

::: {.slide title="Bayes' Theorem Reverses the Conditional"}
[Conditional probability · derivation]{.kicker}

**Step 1.** Write the joint both ways: $P(X, Y) = P(Y \mid X)\,P(X) = P(X \mid Y)\,P(Y)$.

**Step 2.** Divide by $P(Y)$:
$$P(X \mid Y) = \frac{P(Y \mid X)\,P(X)}{P(Y)}, \qquad P(Y) = \sum_x P(Y \mid X{=}x)\,P(X{=}x).$$

**Step 3.** Hypothesis $H$, observed data $D$:

::: {.d2l-note .rule}
posterior $\propto$ likelihood $\times$ prior:
$\;P(H \mid D) \propto P(D \mid H)\,P(H)$.
:::
:::

::: {.slide title="Conditioning Can Remove or Create Dependence"}
[Conditional probability]{.kicker}

**Conditionally independent given $Z$:**
$P(X, Y \mid Z) = P(X \mid Z)\,P(Y \mid Z)$.

::: {.cols .vc}
::: {.col}
- **Common cause:** shoe size and reading level are related through age,
  and unrelated at a fixed age.
- **Common effect:** broken bones and lung cancer become related among
  hospital patients (*explaining away*).
:::
::: {.col .fig .big}
@fig:probability-explaining-away
:::
:::
:::

::: {.slide title="Check: Condition on a Column"}
[Check]{.kicker}

In the joint table, $P(X{=}x_1, Y{=}y_1) = 0.12$ and $P(Y{=}y_1) = 0.32$.
What is $P(X{=}x_1 \mid Y{=}y_1)$? Are $X$ and $Y$ independent?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $0.12 / 0.32 = 0.375$, while $P(X{=}x_1) = 0.40$. Learning
$Y = y_1$ changes the distribution of $X$, so $X$ and $Y$ are **not**
independent.
:::
:::
:::

::: {.slide title="A Test That Never Misses Gives Only 13%"}
[Bayes in action]{.kicker}

::: {.cols .vc}
::: {.col}
$$
\begin{aligned}
P(D{=}1 \mid H{=}1) &= 1.00 \\
P(D{=}1 \mid H{=}0) &= 0.01 \\
\text{prior } P(H{=}1) &= 0.0015
\end{aligned}
$$

$$P(H{=}1 \mid D{=}1) = \frac{0.0015}{0.011485} = 0.13$$

Rare disease: false positives outnumber true ones.
:::

::: {.col .fig .big}
@fig:probability-natural-frequencies
:::
:::
:::

::: {.slide title="A Second Positive Test Raises 13% to 83%"}
[Bayes in action]{.kicker}

::: {.cols .vc}
::: {.col}
Second test: positive with probability $0.98$ if $H{=}1$, $0.03$ if $H{=}0$.
Given $H$ the tests are independent, so likelihoods multiply:
$1 \cdot 0.98$ if $H{=}1$, $0.01 \cdot 0.03$ if $H{=}0$.

$$\frac{0.98 \cdot 0.0015}{0.98 \cdot 0.0015 + 0.01 \cdot 0.03 \cdot 0.9985} = 0.83$$
:::

::: {.col .fig}
@fig:probability-bayes-update
:::
:::
:::

::: {.slide title="Simulating Two Million Patients Confirms 0.83 to About One Point"}
[Bayes in action]{.kicker}

Draw each patient's status, then both test results:

@probability-worked-example-hiv-testing-1

::: {.d2l-note}
Only about 3,500 patients test positive twice, so the simulated fraction
is a frequency with standard error $\sqrt{0.83 \cdot 0.17 / 3500} \approx
0.006$. A deviation of one point from $0.83$ is what we should expect.
:::
:::

::: {.slide title="Check: A Less Rare Disease"}
[Check]{.kicker}

Same first test, but 1% of the population is infected. What is
$P(H{=}1 \mid D_1{=}1)$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.**
$$\frac{1 \cdot 0.01}{1 \cdot 0.01 + 0.01 \cdot 0.99} = \frac{0.01}{0.0199} \approx 0.50.$$
The prior matters as much as the test: one positive is now a coin flip.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Rules:** marginalize by summing; independent means the joint is a
  product.
- **Summaries:** $\mathbb{E}[X]$ and $\textrm{Var}(X)$; variances of independent
  variables add.
- **Standard error** $\sigma/\sqrt{n}$: a frequency of $0.56$ lies $1.2$
  standard errors from $0.5$ after 100 tosses (no evidence of bias), and
  $12$ after 10,000.
- **Conditioning:** keep a row and renormalize. **Bayes:** two positive HIV
  tests take 0.15% to 13% to 83%.

**Next:** looking up what a framework function does, and then the first
model fitted to data.
:::
