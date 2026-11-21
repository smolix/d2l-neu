# Test Functions and Transport
:label:`sec_mdl-test-functions-transport`

A model whose samples miss the data learns nothing from an f-divergence: on
disjoint supports, every member of that family reduces to a constant
(:numref:`sec_mdl-f-divergences`), and a constant has zero gradient. That
diagnosis came from an extreme example, two point masses. Real distributions
usually have spread: two Gaussians overlap everywhere because their tails
never reach zero. Does a little overlap bring the gradient back? The f-divergences also left us with a practical burden: because
they depend on the density ratio $p/q$, which raw samples do not provide, we
had to train a critic network just to estimate the divergence
(:numref:`sec_mdl-f-gan-dual`). Can we compare two distributions from samples
alone, with no critic at all?

To answer the first question with numbers, let the data distribution be
$P = \mathcal{N}(0, 1)$ and the model be $Q = \mathcal{N}(d, 1)$, the same
bell curve shifted by $d$ standard deviations, where $d$ is the model's single
parameter. Below, after importing the libraries we need, we evaluate the five
comparisons from :numref:`tab_mdl-point-mass-table` on this pair along with
their slopes in $d$, which are the gradients the model would receive during
training. The formulas used in the code are derived in the previous section
and this one; for now, focus only on the printed numbers, which we will refer
to as the opening table.

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-1}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-1}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-1}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-1}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-2}
from scipy import optimize
from scipy.stats import norm
```

```{.python .input #mdl-test-functions-transport-test-functions-and-transport-3}
def js(d, x=np.linspace(-12, 20, 6401)):
    """JS divergence of N(0, 1) and N(d, 1) by quadrature on the grid x."""
    log_p, log_q = norm.logpdf(x), norm.logpdf(x, loc=d)
    log_m = np.logaddexp(log_p, log_q) - math.log(2)    # log of the mixture
    terms = np.exp(log_p) * (log_p - log_m) + np.exp(log_q) * (log_q - log_m)
    return terms.sum() * (x[1] - x[0]) / 2

comparisons = {                       # N(0, 1) against N(d, 1), RBF ell = 1
    'KL': lambda d: d ** 2 / 2,
    'JS': js,
    'TV': lambda d: 2 * norm.cdf(d / 2) - 1,
    'MMD^2': lambda d: 2 / math.sqrt(3) * (1 - math.exp(-d ** 2 / 6)),
    'W1': lambda d: d}
h = 1e-3                              # step of the central difference
print('   d' + ''.join(f'{name:>8s} {"slope":>7s}' for name in comparisons))
for d in (0.5, 1.0, 2.0, 4.0, 8.0):
    print(f'{d:4.1f}' + ''.join(
        f'{f(d):8.3f} {(f(d + h) - f(d - h)) / (2 * h):7.1e}'
        for f in comparisons.values()))
```

When the two Gaussians are half a standard deviation apart, all five slopes
lie between $0.1$ and $1$, so any of these objectives would train the model
effectively. By $d = 8$, however, the Jensen--Shannon divergence has saturated at $0.693$
and total variation at $1.000$ (the same constants $\log 2$ and $1$ we saw for
disjoint point masses), and both slopes have fallen to roughly $10^{-4}$. Overlapping tails do not rescue them. The KL divergence still has a
healthy slope equal to $d$ itself, but only because a Gaussian model assigns
positive density to every data point; an implicit generator whose samples lie
on a low-dimensional surface has no such density. That leaves the two
distances that remained informative for point masses. $\textrm{MMD}^2$ grows
initially but stalls by $d = 8$, reaching $1.155$ with a slope below
$10^{-4}$. Only $W_1$ stays linear, equaling $d$ with a slope of $1$ at every
separation.

Neither MMD nor $W_1$ relies on the density ratio $p/q$. Instead, MMD compares
how $P$ and $Q$ *average* a class of functions: averages can be estimated
directly from samples, and the choice of function class governs how far apart
two distributions can be and still produce a useful signal. $W_1$ begins from a geometric question: how far must probability mass travel
to reshape one distribution into the other? Surprisingly, it turns out to be a
comparison of averages as well. Two results organize the rest of this section. Over a fixed
set of features, the comparison of averages admits a closed form with no
auxiliary network to train. Meanwhile, the minimum cost of moving mass equals
the largest gap in averages over functions of slope at most one, which is the
duality behind the Wasserstein GAN (:numref:`sec_gan_objectives`). Throughout,
$P$ denotes the data and $Q$ the model, and we work with concrete examples on
the real line.

## Integral Probability Metrics and MMD
:label:`sec_mdl-ipm-mmd`

### Classes of Test Functions

Total variation is a proper metric, yet it still flattened out in the opening
table. Its variational form explains both why that happened and how to fix it:
total variation is the largest disagreement between $P$ and $Q$ over the
probability of any event $A$ (:numref:`sec_mdl-tv-pinsker`), namely
$\textrm{TV}(P, Q) = \sup_A |P(A) - Q(A)|$. Since the probability of an event
is simply the expectation of its indicator function, total variation is a
supremum, taken over a class of functions, of the difference between two
averages. Nothing forces us to restrict ourselves to indicator functions.
Given any class $\mathcal{G}$ of *test functions* $g$, the **integral
probability metric** (IPM) :cite:`Muller.1997` is defined as

$$
\textrm{IPM}_{\mathcal{G}}(P, Q)
= \sup_{g \in \mathcal{G}} \,\big( \mathbb{E}_{x \sim P}[g(x)] - \mathbb{E}_{x \sim Q}[g(x)] \big).
$$
:eqlabel:`eq_mdl-ipm-def`

Think of $\mathcal{G}$ as a panel of auditors, each scoring points in the
space; the IPM reports the largest gap in average score that any auditor on
the panel can uncover. Every class we consider is symmetric, meaning $-g$
belongs to $\mathcal{G}$ whenever $g$ does, so the gap is always non-negative
and symmetric in $P$ and $Q$. Whether the IPM vanishes only when $P = Q$, as
the word "metric" promises, depends on whether $\mathcal{G}$ is rich enough to
distinguish any two distinct distributions. The critic of
:numref:`sec_mdl-f-gan-dual` played a similar auditing role, but with a crucial
difference: in the variational bound
$D_f(P\|Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$, the convex conjugate
$f^*$ automatically keeps an unconstrained critic $T$ in check. In an IPM, $g$
is averaged directly under both $P$ and $Q$, so nothing constrains $g$ except
membership in $\mathcal{G}$. Everything therefore hinges on choosing the right
class.

Three function classes take center stage here:

- Bounded functions with $|g| \leq \tfrac{1}{2}$ recover total variation. An
  indicator function minus $\tfrac{1}{2}$ satisfies this bound, the constant
  cancels in the difference of averages, and $g = \tfrac{1}{2}\,\textrm{sign}(p - q)$
  attains the supremum.
- Functions that are linear in a fixed set of features yield the **maximum
  mean discrepancy** (MMD) :cite:`Gretton.Borgwardt.Rasch.ea.2012`, which we
  construct next.
- Functions with slope at most one yield the Wasserstein distance, covered in
  detail in :numref:`sec_mdl-optimal-transport`.

Why not simply choose the largest possible class $\mathcal{G}$ every time? A
useful class must satisfy two competing requirements, and making $\mathcal{G}$
too large breaks the first. First, sample averages must track true
expectations. While samples let us estimate expectations directly without a
density ratio, an overly rich panel of auditors will always find spurious gaps
between finite samples. For example, two independent samples of $1{,}000$
points drawn from the *same* continuous distribution share no points at all.
The indicator function of the first sample averages to $1$ on the first sample
and $0$ on the second, so the empirical total variation is $1$ regardless of
whether $P$ and $Q$ are identical. To avoid this, $\mathcal{G}$ must be small
or smooth enough to exclude such spiky functions.

Second, the gap should keep growing as $P$ and $Q$ move farther apart, which
requires $\mathcal{G}$ to contain functions that grow with distance. Total
variation has none: because its test functions are bounded by $\tfrac{1}{2}$,
its gap can never exceed $1$. The largest disagreement occurs on the event
$\{p > q\}$, which for the two Gaussians in the opening table is the half-line
$x < d/2$. Computing the probability of that half-line under both Gaussians
gives $\textrm{TV} = \Phi(d/2) - \Phi(-d/2) = 2\Phi(d/2) - 1$, where $\Phi$ is
the standard normal cumulative distribution function (CDF). That value reaches
$0.954$ at $d = 4$ and lies within $10^{-4}$ of $1$ at $d = 8$. Once a step
function separates the two bell curves almost completely, pushing them farther
apart changes the score by virtually nothing. Total variation fails both
requirements, and we will judge our next two classes against both.

### Mean Features and Kernels

Let us build a better class from the ground up, starting from the smallest
possible panel of auditors and testing each step on a single pair:
$P = \mathcal{N}(0, 1)$ against $Q = \mathcal{N}(0, 4)$, which share the same
mean but differ in variance.

Start with the simplest choice, linear test functions $g(x) = w\, x$ with
$|w| \leq 1$. Here
$\mathbb{E}_P g - \mathbb{E}_Q g = w\,(\mathbb{E}_P X - \mathbb{E}_Q X)$, so
taking the supremum over $|w| \leq 1$ gives $|\mathbb{E}_P X - \mathbb{E}_Q X|$,
the distance between the two means. That panel is blind to variance, and on
our test pair it reports $0$.

To see spread as well as location, add a second feature. Define the feature
map $\boldsymbol{\varphi}(x) = (x, x^2)$ and take linear combinations
$g(x) = \mathbf{w}^\top \boldsymbol{\varphi}(x)$ with $\|\mathbf{w}\| \leq 1$.
Writing $\mathbf{m}_P = \mathbb{E}_P[\boldsymbol{\varphi}(x)]$ for the *mean
feature* of $P$ (and similarly $\mathbf{m}_Q$ for $Q$), the gap in averages is
$\mathbb{E}_P g - \mathbb{E}_Q g = \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q)$.
By the Cauchy--Schwarz inequality, the maximum of a linear function over the
unit ball is simply the Euclidean norm of its coefficient vector:

$$
\textrm{MMD}(P, Q) = \sup_{\|\mathbf{w}\| \leq 1} \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q)
= \|\mathbf{m}_P - \mathbf{m}_Q\|.
$$
:eqlabel:`eq_mdl-mmd-features`

Thus MMD is the distance between the mean feature vectors of $P$ and $Q$, and
the supremum has disappeared altogether. On our test pair, the mean features
are $\mathbf{m}_P = (0, 1)$ and $\mathbf{m}_Q = (0, 4)$, giving
$\textrm{MMD}^2 = 0^2 + 3^2 = 9$: the comparison now detects differences in
both means and second moments.

Next, expand the squared norm. Because the product of two expectations equals
the expectation of a product over independent draws, we can write
$\|\mathbf{m}_P\|^2 = \mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle$
for independent $x, x' \sim P$. Doing the same for independent $y, y' \sim Q$
and the cross term gives
$\|\mathbf{m}_P - \mathbf{m}_Q\|^2 =
\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle +
\mathbb{E}\langle\boldsymbol{\varphi}(y), \boldsymbol{\varphi}(y')\rangle -
2\,\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(y)\rangle$.
Notice that the features $\boldsymbol{\varphi}(x)$ never appear alone; only
their inner products do. Defining the **kernel**
$k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle$,
we obtain

$$
\textrm{MMD}^2(P, Q)
= \mathbb{E}_{x, x' \sim P}[k(x, x')] + \mathbb{E}_{y, y' \sim Q}[k(y, y')]
- 2\, \mathbb{E}_{x \sim P, y \sim Q}[k(x, y)].
$$
:eqlabel:`eq_mdl-mmd2`

This expression has an immediate geometric reading: the first two terms
measure average similarity *within* $P$ and *within* $Q$, while the third
measures similarity *across* them. When draws from $P$ and $Q$ intermingle,
within-sample and cross-sample similarities match and the three terms cancel;
when they separate into distinct clusters, the within terms dominate. Our
first step corresponds to the *linear kernel* $k(x, x') = x x'$, and our
second to the *quadratic kernel* $k(x, x') = x x' + x^2 x'^2$.

Even the quadratic kernel still has a blind spot: any two distributions with
the same mean and variance have distance zero. Adding $x^3$, then $x^4$, and
so on would never end. The decisive step is therefore to choose the kernel
directly as a similarity measure, without ever writing down a finite list of
features. The *RBF kernel* $k(x, x') = e^{-(x - x')^2/(2\ell^2)}$ treats two
points as similar when they lie within roughly a *length scale* $\ell$ of each
other. Although we chose it directly, it is still an inner product of
features: factoring it as
$e^{-x^2/(2\ell^2)}\, e^{-x'^2/(2\ell^2)}\, e^{x x'/\ell^2}$ and expanding the
last factor as a power series reveals an inner product over infinitely many
features $\varphi_j(x) = x^j e^{-x^2/(2\ell^2)} / (\ell^j \sqrt{j!})$ for
$j = 0, 1, 2, \ldots$, which are damped monomials of every order with rapidly
decaying weights. Our panel now contains infinitely many auditors, yet
evaluating its verdict still requires only the three averages in
:eqref:`eq_mdl-mmd2`. In $\mathbb{R}^n$, we simply replace $(x - x')^2$ with
$\|x - x'\|^2$ and nothing else changes.

With this kernel, no blind spot remains: $\textrm{MMD}(P, Q) = 0$ if and only
if $P = Q$ :cite:`Gretton.Borgwardt.Rasch.ea.2012`, making RBF-MMD a true
metric. Kernels with this property are called *characteristic*; the linear and
quadratic kernels are not.

For Gaussians, we can evaluate the RBF expectations in closed form. If
$Z \sim \mathcal{N}(\mu, s^2)$, a standard Gaussian integral gives

$$
\mathbb{E}\big[e^{-Z^2/(2\ell^2)}\big]
= \Big(1 + \frac{s^2}{\ell^2}\Big)^{-1/2} \exp\!\Big(-\frac{\mu^2}{2(\ell^2 + s^2)}\Big).
$$
:eqlabel:`eq_mdl-rbf-gaussian-integral`

Because the difference of two independent Gaussians is again Gaussian with the
sum of their variances, we can read off all three terms of :eqref:`eq_mdl-mmd2`
immediately. On our test pair $\mathcal{N}(0, 1)$ versus $\mathcal{N}(0, 4)$
at $\ell = 1$, the differences $x - x'$, $y - y'$, and $x - y$ all have mean
zero and variances $2$, $8$, and $5$, yielding
$\textrm{MMD}^2 = 3^{-1/2} + 9^{-1/2} - 2 \cdot 6^{-1/2} \approx 0.094$. For
the two shifted Gaussians of the opening table, the same integral gives
$\tfrac{2}{\sqrt{3}}\big(1 - e^{-d^2/6}\big)$ (Exercise
:exref:`ex_mdl-divergences-distances-q-mathcaln-delta`a). For the point masses
of :numref:`tab_mdl-point-mass-table`, there is no averaging to do at all, and
:eqref:`eq_mdl-mmd2` immediately gives $1 + 1 - 2k(0, d) = 2 - 2e^{-d^2/2}$,
matching the fourth row of that table. In both cases, only the cross term
depends on $d$, and it decays rapidly once $d$ exceeds a few length scales.
The gap therefore flattens out, failing our second requirement: an RBF auditor
is smooth, but nearsighted.

### Estimation from Samples

Where a kernel shines is our first requirement: reliable estimation from
samples. Given finite samples from $P$ and $Q$, each expectation in
:eqref:`eq_mdl-mmd2` becomes an average over pairs of points. To obtain an
unbiased estimator :cite:`Gretton.Borgwardt.Rasch.ea.2012`, we omit the
diagonal terms pairing a point with itself, which would otherwise bias the
within-sample averages upward. Below, we evaluate this estimator on
$N = 1{,}000$ draws from $\mathcal{N}(0, 1)$ and $1{,}000$ draws from
$\mathcal{N}(0, 4)$ under the linear, quadratic, and RBF kernels, whose
population values we just computed as $0$, $9$, and $0.094$. For the RBF
kernel, we also test two additional pairs: a second independent sample from
$\mathcal{N}(0, 1)$ (population value $0$), and a shifted sample from
$\mathcal{N}(0.5, 1)$ (population value $0.047$ in the opening table). The
goal is to see how tightly the sample estimates track these population values.

```{.python .input #divergences-distances-mmd}
def mmd2_unbiased(x, y, k):
    """Unbiased MMD^2 between samples x and y for the kernel k(a, b)."""
    kxx = k(x[:, None], x[None, :])
    kyy = k(y[:, None], y[None, :])
    n, m = len(x), len(y)
    return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
            + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2 * k(x[:, None], y[None, :]).mean())

linear = lambda a, b: a * b                         # features phi(x) = x
quadratic = lambda a, b: a * b + a ** 2 * b ** 2    # features (x, x^2)
rbf = lambda a, b: np.exp(-(a - b) ** 2 / 2)        # RBF kernel, ell = 1

rng = np.random.default_rng()
n = 1000
x_p = rng.standard_normal(n)                # P = N(0, 1)
y_var = 2 * rng.standard_normal(n)          # N(0, 4)
y_same = rng.standard_normal(n)             # N(0, 1) again
y_shift = rng.standard_normal(n) + 0.5      # N(0.5, 1)
for name, y, k, pop in [
        ('linear,    N(0,1) vs N(0,4)', y_var, linear, 0.0),
        ('quadratic, N(0,1) vs N(0,4)', y_var, quadratic, 9.0),
        ('RBF,       N(0,1) vs N(0,4)', y_var, rbf,
         3 ** -0.5 + 9 ** -0.5 - 2 * 6 ** -0.5),
        ('RBF,       N(0,1) vs N(0,1)', y_same, rbf, 0.0),
        ('RBF,       N(0,1) vs N(0.5,1)', y_shift, rbf,
         2 / np.sqrt(3) * (1 - np.exp(-1 / 24)))]:
    print(f'{name:30s} MMD^2 = {mmd2_unbiased(x_p, y, k):+.4f}  '
          f'(population {pop:.4f})')
```

The linear kernel returns a number close to zero because the two means match.
Specifically, its unbiased estimate equals the squared difference of the two
sample means minus an estimate of that difference's variance, roughly
$(1 + 4)/1{,}000 = 0.005$; as a result, it rarely dips much below $-0.005$,
while about one run in twenty exceeds $+0.015$. The quadratic kernel lands
near $9$, give or take $3$, because fourth powers amplify sampling noise.
Under the RBF kernel, the estimate is roughly $0.094 \pm 0.03$ for the
variance-shifted pair and $0.047 \pm 0.02$ for the mean-shifted pair, so the
sampling error can be a noticeable fraction of the signal. Between two
independent samples from the *same* $\mathcal{N}(0, 1)$ distribution, however,
the RBF estimate is remarkably tight, staying within a few thousandths of zero
(and occasionally dipping slightly below it) because the first-order sampling
errors in the three averages cancel out. Where total variation reported a spurious gap of
$1$ between two samples from the same distribution, the RBF kernel sees none.

In short, the RBF estimator has relatively large error when measuring a
non-zero gap, yet extraordinary precision when $P = Q$. That makes it a
powerful *two-sample test*: with $1{,}000$ points per sample, an RBF
$\textrm{MMD}^2$ of even a few hundredths is strong evidence that the two
distributions differ, obtained without training a critic or evaluating a
single density.

### The Witness

Because an IPM is defined as a supremum, it is natural to ask which test
function actually achieved the maximum gap. For MMD, the supremum in
:eqref:`eq_mdl-mmd-features` is attained by
$\mathbf{w}^\star \propto \mathbf{m}_P - \mathbf{m}_Q$. The resulting test
function, known as the **witness**, is
$g^\star(x) \propto \langle \mathbf{m}_P - \mathbf{m}_Q, \boldsymbol{\varphi}(x)\rangle = \mathbb{E}_P[k(x, X)] - \mathbb{E}_Q[k(x, Y)]$,
which again requires only kernel evaluations. For our shifted Gaussians
$\mathcal{N}(0, 1)$ and $\mathcal{N}(0.5, 1)$, the two densities are mirror
images across $x = 0.25$, so the population witness changes sign right at
$0.25$. Below, we estimate the witness from our samples on a grid and plot it
alongside the two densities. We also print $\mathbb{E}_P g - \mathbb{E}_Q g$
for the unscaled witness
$g = \langle \mathbf{m}_P - \mathbf{m}_Q, \boldsymbol{\varphi}\rangle$, which
equals $\|\mathbf{m}_P - \mathbf{m}_Q\|^2 = \textrm{MMD}^2$, estimated here
with the self-pairs included.

```{.python .input #mdl-divergences-distances-integral-probability-metrics-and-mmd-1}
g = lambda s: (rbf(s[:, None], x_p[None, :]).mean(axis=1)    # the unscaled
               - rbf(s[:, None], y_shift[None, :]).mean(axis=1))  # witness
grid = np.linspace(-4.0, 4.5, 400)
witness = g(grid)
print(f'witness changes sign at x = {grid[np.argmax(witness < 0)]:.2f}')
print(f'E_P g - E_Q g = {g(x_p).mean() - g(y_shift).mean():.4f}')
d2l.plot(grid, [witness / np.abs(witness).max(), norm.pdf(grid),
                norm.pdf(grid, loc=0.5)], 'x', '',
         legend=['witness (scaled)', 'p = N(0, 1)', 'q = N(0.5, 1)'],
         figsize=(5, 3))
```

As expected, the witness is positive where $P$ has higher density and negative
where $Q$ has higher density, crossing zero near $x = 0.25$ (with $1{,}000$
points per sample, the crossing fluctuates by a tenth or two across runs). The
printed gap matches the unbiased estimate from the previous cell to within
about $0.001$, with the small discrepancy coming from the self-pairs.

The tails of the witness plot tell an even more important story than the
crossing. To shrink the gap $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$, a generative
model must push each of its samples uphill along $g$, so the slope of the
witness at a model sample is the gradient signal that sample receives. Between
the two bell curves, the witness is steep and provides a strong pull toward
the data. A few length scales out in the tails, however, the witness flattens
to zero, leaving any distant model sample with almost no gradient. We are
seeing the $\textrm{MMD}^2$ column of the opening table from the vantage point
of a single sample.

Fixed-kernel MMD therefore plays two distinct roles in practice: as a
non-parametric two-sample test, and as a training objective for generative
models that avoids adversarial optimization :cite:`Li.Swersky.Zemel.2015`. For
instance, the kernel inception distance used to evaluate image generators
(:numref:`sec_dcgan`) is simply the unbiased MMD estimator applied to deep
image features with a cubic kernel. Choosing the length scale $\ell$ carefully
is essential: at $\ell = 1$, the slope at $d = 8$ collapsed to
$7 \times 10^{-5}$ (Exercise
:exref:`ex_mdl-divergences-distances-q-mathcaln-delta`b explores larger scales).
In practice, one often sets $\ell$ to the median pairwise distance between
samples or sums RBF kernels across multiple length scales, the strategy used
when training a GAN *generator* with MMD in :numref:`sec_gan_objectives`.

## Optimal Transport and the Wasserstein Distance
:label:`sec_mdl-optimal-transport`

One column of the opening table never flattened out. $W_1$ kept a constant
slope of $1$ because it incorporates the geometry of the underlying space,
which f-divergences ignore and a fixed kernel sees only locally. Under
$W_1$, the point mass $\delta_{0.01}$ is close to $\delta_0$ simply because
its mass need only travel a distance of $0.01$. The idea goes back to 1781,
when Gaspard Monge asked how to move a pile of earth into a pit of equal
volume with the least total work. That minimum work is a natural measure of
how far apart two distributions are.

### Transport Plans

Consider two discrete distributions supported on the six points
$0, 1, \ldots, 5$ of the real line, with $P$ concentrated on the left and $Q$
on the right, as shown in the top left panel of
:numref:`fig_mdl-ot-transport-plan`. To reshape $P$ into $Q$, we must specify
how much mass moves from each source $x$ to each destination $y$. Such a
*transport plan* is a joint distribution $\gamma$ over pairs $(x, y)$ whose
marginals are $P$ and $Q$: the total mass leaving each $x$ equals $P(x)$, and
the total mass arriving at each $y$ equals $Q(y)$. In probability theory, a
transport plan is also called a *coupling* of $P$ and $Q$. The
**Wasserstein-1 distance** (also known as the earth mover's distance) is the
minimum expected distance traveled over all valid plans. Writing $\Pi(P, Q)$
for the set of all couplings of $P$ and $Q$ on $\mathbb{R}^n$ with finite
first moments,

$$
W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}\big[ \|x - y\| \big].
$$
:eqlabel:`eq_mdl-w1-primal`

The subscript $1$ refers to the first power of the distance; we will meet the
squared-distance version $W_2$ shortly. Between two point masses $\delta_0$
and $\delta_d$, there is only one possible plan, moving all mass from $0$ to
$d$, so $W_1(\delta_0, \delta_d) = |d|$, with gradient $\pm 1$ for $d \neq 0$.
For the two shifted Gaussians of the opening table, translating every point by
$d$ is a valid plan of cost $d$, and no plan can do better because
$\mathbb{E}_\gamma|x - y| \geq |\mathbb{E}_\gamma[x - y]| = d$ for every
coupling $\gamma$. In both cases, $W_1$ recovers the separation $d$ itself,
providing the clean gradient signal that f-divergences lack.

![Transport on the line. Top left: a transport plan between two distributions on six points; each ribbon carries mass from a point of $P$ (top) to a point of $Q$ (bottom), with width proportional to the mass moved. Bottom left: the CDFs of $P$ and $Q$; the shaded area between them is $1.7$. Right: for two point masses separated by $d$, the Jensen--Shannon divergence (in nats) stays flat at $\log 2$, whereas $W_1$ (in units of $x$) grows linearly with $d$.](../img/mdl-it-ot-transport-plan.svg)
:label:`fig_mdl-ot-transport-plan`

### Kantorovich--Rubinstein Duality

As a computational recipe, :eqref:`eq_mdl-w1-primal` looks intimidating: it
minimizes over joint distributions $\gamma$, which are awkward even to
parameterize from samples. On finite point sets, however, it is a familiar
optimization problem: a linear program. The unknowns are the entries of the
plan, and both the transport cost and the marginal constraints are linear in
those entries. Every linear program has a *dual* problem that maximizes over
multipliers for its constraints (:numref:`subsec_mdl-dual-problem`). For
optimal transport, the dual replaces the search over joint distributions with
a maximization over a single function, and that reformulation is what makes
$W_1$ practical in deep learning.

**Proposition (Kantorovich--Rubinstein duality).** *For distributions $P$ and
$Q$ on $\mathbb{R}^n$ with finite first moments,*

$$
W_1(P, Q) = \sup_{\|g\|_{\textrm{Lip}} \leq 1}
\big( \mathbb{E}_{x \sim P}[g(x)] - \mathbb{E}_{y \sim Q}[g(y)] \big),
$$
:eqlabel:`eq_mdl-kr-dual`

*where the supremum runs over all 1-Lipschitz functions $g$, meaning
$|g(x) - g(y)| \leq \|x - y\|$ for all $x, y$.*

An infimum over couplings has turned into a supremum over a single test
function: $W_1$ is precisely the IPM :eqref:`eq_mdl-ipm-def` over functions
with slope at most $1$, the third class we promised earlier. A 1-Lipschitz
auditor cannot make sudden jumps, so it cannot memorize individual sample
points the way an indicator function does. Yet unlike bounded functions or RBF
kernels, it can grow without limit, satisfying our second requirement. On the
two shifted Gaussians of the opening table, for example, $g(x) = -x$ is
1-Lipschitz and has expectations $0$ under $P$ and $-d$ under $Q$, achieving a
gap of $d = W_1(P, Q)$. Better still, its gradient points toward the data with
magnitude $1$ everywhere, no matter how far a model sample lies from the data.

Because a neural network can approximate such a test function, the **WGAN**
:cite:`Arjovsky.Chintala.Bottou.2017` trains the dual objective
:eqref:`eq_mdl-kr-dual` directly, using a neural critic constrained to be
approximately 1-Lipschitz in the role of $g$. :numref:`sec_gan_objectives`
compares this objective with other adversarial losses and trains such a
critic, whose slope a gradient penalty pulls toward one.
:numref:`sec_gan_convergence` shows why later recipes center their penalties
at zero instead: a penalty that prefers unit slope keeps pushing a generator
that already matches the data, while one centered at zero only keeps the
critic's gradient small at the samples and enforces no Lipschitz constraint.
Keep in mind that when a critic is only
approximately Lipschitz and restricted to a neural architecture, the score it
reports is no longer exact $W_1$.

### Transport on the Line

We prove Kantorovich--Rubinstein duality in one dimension, where the argument
also delivers a closed-form formula requiring no optimization at all;
:citet:`Peyre.Cuturi.2019` give the proof in higher dimensions. The intuition
is simply to count how much mass crosses each point $t \in \mathbb{R}$. To the
left of $t$ (including $t$), $P$ holds mass $F_P(t)$ and $Q$ holds mass
$F_Q(t)$, so any valid transport plan must move at least $|F_P(t) - F_Q(t)|$
of mass across $t$. Moreover, a parcel moved from $x$ to $y$ crosses every $t$
between $x$ and $y$, so its distance $|x - y|$ equals the length of the
interval of points it crosses. Summing over all parcels, the total transport
cost is the integral over $t$ of the total mass crossing $t$.

**Proposition ($W_1$ on the line).** *For distributions $P$ and $Q$ on
$\mathbb{R}$ with CDFs $F_P$, $F_Q$ and finite first moments,*

$$
W_1(P, Q) = \int_{-\infty}^{\infty} \lvert F_P(t) - F_Q(t) \rvert \, dt
= \sup_{\|g\|_{\textrm{Lip}} \leq 1}
\big( \mathbb{E}_P[g] - \mathbb{E}_Q[g] \big),
$$
:eqlabel:`eq_mdl-w1-cdf`

*so Kantorovich--Rubinstein duality holds on the real line.*

**Proof.** *Step 1: Every plan costs at least the area between the CDFs.* For
$(X, Y) \sim \gamma$, write
$|X - Y| = \int \mathbf{1}(t \textrm{ lies between } X \textrm{ and } Y)\, dt$.
Because the integrand is non-negative, we may exchange expectation and
integration:

$$
\mathbb{E}_\gamma|X - Y|
= \int \Big( \gamma(X \leq t < Y) + \gamma(Y \leq t < X) \Big)\, dt
\geq \int \big| \gamma(X \leq t < Y) - \gamma(Y \leq t < X) \big|\, dt.
$$

The key step is simplifying the difference inside the last integral. Adding
and subtracting $\gamma(X \leq t, Y \leq t)$ gives
$\gamma(X \leq t < Y) - \gamma(Y \leq t < X) = \gamma(X \leq t) - \gamma(Y \leq t) = F_P(t) - F_Q(t)$,
since $\gamma$ has marginals $P$ and $Q$. Thus every transport plan costs at
least $\int |F_P - F_Q|$.

*Step 2: One plan achieves this lower bound.* Match quantiles directly: send
the lowest tenth of $P$ to the lowest tenth of $Q$, and so on up the line.
Formally, let $U$ be uniform on $(0, 1)$ and set $X = F_P^{-1}(U)$ and
$Y = F_Q^{-1}(U)$, where the quantile function
$F^{-1}(\lambda) = \inf\{x : F(x) \geq \lambda\}$ for $\lambda \in (0, 1)$
satisfies $F^{-1}(\lambda) \leq t$ if and only if $\lambda \leq F(t)$. This
defines a valid coupling because $\Pr(X \leq t) = \Pr(U \leq F_P(t)) = F_P(t)$
and similarly for $Y$. We call it the *monotone plan*, since $X$ and $Y$
increase together. Under this coupling,
$\gamma(X \leq t < Y) = \Pr\big(F_Q(t) < U \leq F_P(t)\big)$ and
$\gamma(Y \leq t < X) = \Pr\big(F_P(t) < U \leq F_Q(t)\big)$. At every $t$, at
least one of these two probabilities is zero, meaning the monotone plan never
sends mass in both directions across the same point. The inequality in Step 1
therefore holds with equality, so the minimum transport cost equals the area
between the CDFs.

*Step 3: No 1-Lipschitz test function beats any plan.* This *weak duality*
argument holds in every dimension. For any 1-Lipschitz function $g$ and any
coupling $\gamma$, we have $g(x) - g(y) \leq \|x - y\|$ pointwise; taking the
expectation under $\gamma$ immediately gives
$\mathbb{E}_P[g] - \mathbb{E}_Q[g] \leq \mathbb{E}_\gamma\|x - y\|$ (compare
the weak duality of :numref:`subsec_mdl-dual-problem`).

*Step 4: One 1-Lipschitz test function achieves the area.* It suffices to
consider functions of the form $g(x) = g(0) + \int_0^x g'(t)\, dt$ with
$|g'| \leq 1$: every such $g$ is 1-Lipschitz, and Step 3 already bounds all
others. The constant $g(0)$ cancels in $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$.
Rewriting
$\int_0^X g'(t)\,dt = \int_0^{\infty} g'(t)\, \mathbf{1}(X > t)\, dt - \int_{-\infty}^0 g'(t)\, \mathbf{1}(X \leq t)\, dt$
and exchanging expectation and integration (justified by Fubini's theorem from
:numref:`sec_mdl-integral_calculus`, given the finite first moments) yields

$$
\mathbb{E}_P[g] - \mathbb{E}_Q[g] = \int_{-\infty}^{\infty} g'(t)\, \big( F_Q(t) - F_P(t) \big)\, dt.
$$

Choosing $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ turns the right-hand
side into $\int |F_P(t) - F_Q(t)|\, dt$. Combining all four steps, the
supremum over 1-Lipschitz functions is at least the area (Step 4) and at most
the minimum plan cost (Step 3), which equals the area (Steps 1 and 2).
$\blacksquare$

On the real line, then, $W_1$ is simply the area between the two CDFs, and for
two equal-sized samples it reduces to the mean absolute difference of the
sorted values (Exercise
:exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`a). It is remarkable that
a full linear program collapses so cleanly. Let us verify this collapse on the
six-point example from :numref:`fig_mdl-ot-transport-plan`. Across the five
intervals between adjacent atoms, the CDF gaps are $0.25$, $0.35$, $0.5$,
$0.35$, and $0.25$, summing to $1.7$, the shaded area in the figure. The code
below computes $W_1$ in three ways: by summing the five CDF gaps, by solving
the $36$-variable linear program over $6 \times 6$ transport matrices with row
sums $p$ and column sums $q$, and by constructing the monotone plan of Step 2
(where each route carries the overlap of the corresponding quantile intervals)
and evaluating its cost. Finally, it checks whether the linear program
returned the monotone plan.

```{.python .input #divergences-distances-w1}
atoms = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
p_w = np.array([0.30, 0.20, 0.25, 0.10, 0.10, 0.05])
q_w = np.array([0.05, 0.10, 0.10, 0.25, 0.20, 0.30])

F_p, F_q = np.cumsum(p_w), np.cumsum(q_w)
w1_cdf = np.sum(np.abs(F_p - F_q)[:-1] * np.diff(atoms))

C = np.abs(atoms[:, None] - atoms[None, :])      # cost matrix |x_i - y_j|
n_atoms = len(atoms)
A_eq = np.zeros((2 * n_atoms, n_atoms * n_atoms))
for i in range(n_atoms):
    A_eq[i, i * n_atoms:(i + 1) * n_atoms] = 1   # row sums    = p
    A_eq[n_atoms + i, i::n_atoms] = 1            # column sums = q
res = optimize.linprog(C.ravel(), A_eq=A_eq,
                       b_eq=np.concatenate([p_w, q_w]), method='highs')
plan_lp = res.x.reshape(n_atoms, n_atoms)
# Monotone plan (Step 2): source i holds the quantiles (F_p[i-1], F_p[i]],
# destination j holds (F_q[j-1], F_q[j]]; route i -> j carries their overlap
lo_p, lo_q = F_p - p_w, F_q - q_w
mono = np.clip(np.minimum(F_p[:, None], F_q[None, :])
               - np.maximum(lo_p[:, None], lo_q[None, :]), 0, None)
print(f'W1 from the CDF gaps     : {w1_cdf:.4f}')
print(f'W1 from the LP           : {res.fun:.4f}')
print(f'cost of the monotone plan: {np.sum(mono * C):.4f}')
print(f'LP plan = monotone plan  : {np.allclose(plan_lp, mono)}, '
      f'mass the LP plan moves leftward: {np.tril(plan_lp, -1).sum():.2f}')
```

All three methods agree on $W_1 = 1.7$: summing five CDF gaps gives the exact
same answer as solving a $36$-variable linear program. The last line, however,
holds a small surprise: the linear program did not return the monotone plan.
Instead, it found a different plan with the exact same cost, which likewise
moves zero mass to the left.

The optimal $W_1$ plan is therefore not unique here, and the reason is easy to
spot. Because $F_P \geq F_Q$ everywhere, no mass ever needs to move leftward.
For *any* plan that moves mass strictly rightward ($y \geq x$), we have
$|x - y| = y - x$, so its total cost equals the difference of the two means,
$3.35 - 1.65 = 1.7$, regardless of how the routes cross. The monotone plan
drawn in the figure is simply one of many equally good plans.

Two tools that appear elsewhere in the book replace the absolute distance with
its square: the Fréchet inception distance for evaluating image generators
(:numref:`sec_dcgan`), and minibatch optimal transport in flow matching, which
pairs noise with data by minimizing squared distance
(:numref:`sec_mdl-flow-matching-transport`). The **Wasserstein-2 distance**
averages the squared distance and takes the square root afterwards:
$W_2(P, Q) = \big(\inf_{\gamma} \mathbb{E}_\gamma \|x - y\|^2\big)^{1/2}$.
Under a squared cost, crossing two transport ribbons always costs strictly
more than uncrossing them, so on the line the monotone plan becomes the
*unique* optimum (Exercise
:exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`d). Between two
one-dimensional Gaussians with means $\mu_1, \mu_2$ and standard deviations
$\sigma_1, \sigma_2$, it evaluates cleanly to
$W_2^2 = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$ (Exercise
:exref:`ex_mdl-test-functions-transport-w2-gaussians`a). The Fréchet inception
distance is the multivariate counterpart of this formula applied to Gaussians
fitted to image features, with covariance matrices replacing the variances.

### Entropic Regularization and Sinkhorn Iterations
:label:`subsec_mdl-sinkhorn`

This subsection may be skipped on a first reading. Beyond one dimension, there
is in general no closed-form CDF formula, and the linear program over $N$
points on each side has $N^2$ variables. The standard way to make optimal
transport fast and differentiable is **entropic regularization**
:cite:`Cuturi.2013`: given a cost matrix $\mathbf{C}$ and a regularization
strength $\varepsilon > 0$, we add
$\varepsilon \sum_{ij} \gamma_{ij} (\log \gamma_{ij} - 1)$ to the transport
cost $\sum_{ij} \gamma_{ij} C_{ij}$. Because any valid plan satisfies
$\sum_{ij}\gamma_{ij} = 1$, this extra term equals $-\varepsilon$ times the
entropy of $\gamma$ plus a constant; it discourages plans that concentrate all
their mass on a few sharp routes. Entropic regularization is what turned
optimal transport into a practical loss function for deep learning, enabling
generative models to be trained directly on regularized transport costs
:cite:`Genevay.Peyre.Cuturi.2018`.

Remarkably, the regularized problem solves almost in closed form. Forming the
Lagrangian (:numref:`subsec_mdl-dual-problem`) with a multiplier $\xi_i$ for
each row-sum constraint $\sum_j \gamma_{ij} = p_i$ and a multiplier $\zeta_j$
for each column-sum constraint $\sum_i \gamma_{ij} = q_j$, and setting the
derivative with respect to $\gamma_{ij}$ to zero, we obtain
$C_{ij} + \varepsilon \log \gamma_{ij} - \xi_i - \zeta_j = 0$, or
$\gamma_{ij} = e^{-(C_{ij} - \xi_i - \zeta_j)/\varepsilon}$. Each route
carries mass decaying exponentially in how far its cost $C_{ij}$ exceeds
$\xi_i + \zeta_j$, measured in units of $\varepsilon$. In matrix form,

$$
\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\, \mathbf{K}\, \textrm{diag}(\mathbf{b}),
\qquad \mathbf{K} = e^{-\mathbf{C}/\varepsilon} \textrm{ (entrywise)},
$$
:eqlabel:`eq_mdl-sinkhorn-form`

where $\mathbf{a} = e^{\boldsymbol{\xi}/\varepsilon}$ and
$\mathbf{b} = e^{\boldsymbol{\zeta}/\varepsilon}$ are positive scaling
vectors. The regularized plan is simply the *Gibbs kernel* $\mathbf{K}$ (a
matrix of positive weights, unrelated to the kernels of MMD), rescaled row by
row and column by column. Only $\mathbf{a}$ and $\mathbf{b}$ remain to be
found, and each set of marginal constraints determines one of them: with
$\mathbf{b}$ fixed, setting $\mathbf{a} = \mathbf{p} / (\mathbf{K}\mathbf{b})$
(entrywise) enforces the row sums $\mathbf{p}$; with $\mathbf{a}$ fixed,
setting $\mathbf{b} = \mathbf{q} / (\mathbf{K}^\top \mathbf{a})$ enforces the
column sums $\mathbf{q}$.

The **Sinkhorn iterations** alternate these two updates, updating $\mathbf{b}$
and then $\mathbf{a}$ until both marginals match. After the $\mathbf{a}$
update the row sums are exact, so we stop once the column sums match
$\mathbf{q}$ within $10^{-9}$. Every step requires only two matrix-vector
products, making the algorithm fast on GPUs and trivial to differentiate
through. Below we report the *entropic cost*, meaning the transport cost
$\sum_{ij} (\gamma_\varepsilon)_{ij} C_{ij}$ of the regularized plan (without
the entropy penalty). As $\varepsilon \to 0$, this cost approaches the exact
unregularized optimum, but convergence slows dramatically
:cite:`Peyre.Cuturi.2019` and the entries of $\mathbf{K}$ can underflow;
production implementations therefore work in log-space on $\log \mathbf{a}$
and $\log \mathbf{b}$ (Exercise
:exref:`ex_mdl-divergences-distances-sinkhorn-divergences`b).

To see how small $\varepsilon$ must be, and how expensive a small
$\varepsilon$ becomes, we apply Sinkhorn to the six-point example with the
squared cost $C_{ij} = (x_i - x_j)^2$, where the monotone plan of
:numref:`fig_mdl-ot-transport-plan` is the unique optimum and its ten active
routes cost $W_2^2 = 3.5$. The cell below verifies this optimum with a linear
program and runs Sinkhorn at three values of $\varepsilon$, reporting the
iteration count, entropic cost, and number of routes carrying mass above
$10^{-3}$.

```{.python .input #divergences-distances-sinkhorn}
def sinkhorn(p, q, C, eps, tol=1e-9, max_iters=10 ** 6):
    """Entropic OT by Sinkhorn iterations, stopped once the column sums are
    within tol of q (L1); returns the plan, its cost and the iterations."""
    K = np.exp(-C / eps)                         # the Gibbs kernel
    a = np.ones_like(p)
    for it in range(1, max_iters + 1):
        b = q / (K.T @ a)                        # match the column sums
        a = p / (K @ b)                          # match the row sums ...
        if np.abs(b * (K.T @ a) - q).sum() < tol:  # ... columns still right?
            break
    plan = a[:, None] * K * b[None, :]
    return plan, np.sum(plan * C), it

C2 = (atoms[:, None] - atoms[None, :]) ** 2      # squared cost (x_i - y_j)^2
res2 = optimize.linprog(C2.ravel(), A_eq=A_eq,
                        b_eq=np.concatenate([p_w, q_w]), method='highs')
print(f'squared cost, LP optimum : {res2.fun:.5f}')
plans, eps_values = [], (2.0, 0.5, 0.1)
for eps in eps_values:
    plan, cost, iters = sinkhorn(p_w, q_w, C2, eps)
    plans.append(plan)
    print(f'epsilon = {eps}: {iters:6d} iterations, entropic cost = '
          f'{cost:.5f}, routes above 1e-3 = {(plan > 1e-3).sum()}')
d2l.show_heatmaps(d2l.tensor(np.stack(plans)[None]), 'destination j',
                  'source i', titles=[f'epsilon = {e}' for e in eps_values],
                  figsize=(9, 2.8), cmap='Blues')
```

The linear program confirms $W_2^2 = 3.5$. At $\varepsilon = 2$, the
regularized plan smears each source's mass across many destinations ($25$ of
the $36$ routes carry more than $10^{-3}$), pushing the cost up to $4.06$. At
$\varepsilon = 0.5$, the plan tightens to $14$ active routes with cost $3.57$.
By $\varepsilon = 0.1$, only the ten routes of the monotone plan survive, and
the cost reaches $3.50002$, within $2 \times 10^{-5}$ of the exact optimum. In
the heatmaps, a blurry band sharpens into a clean staircase. The natural scale
for $\varepsilon$ is the cost gap between competing routes, which is of order
$1$ on this grid; setting $\varepsilon$ to a tenth of that scale recovers the
optimum accurately.

The iteration counts reveal the catch. Sinkhorn needed about $50$ iterations
at $\varepsilon = 2$ and about $190$ at $\varepsilon = 0.5$, but roughly
$150{,}000$ at $\varepsilon = 0.1$. Cutting $\varepsilon$ by a factor of five
increased the work by a factor of nearly $800$. Sinkhorn iterations are fast
only when you can afford a blurred transport plan (Exercise
:exref:`ex_mdl-test-functions-transport-sinkhorn-iterations`b tracks this
growth quantitatively).

## Fixed Kernels and Trained Critics

MMD and $W_1$ split our two requirements right down the middle. A fixed kernel
defines a small, smooth function class: MMD reduces to three sample averages,
and its estimation error from $N$ points decays as $O(N^{-1/2})$ in *any*
dimension :cite:`Gretton.Borgwardt.Rasch.ea.2012`, yet it loses sight of the
data beyond a few length scales. By contrast, 1-Lipschitz functions provide a
steady gradient at every separation, which is exactly what a model far from
its data needs. However, even though they cannot jump like indicators, the
class of all 1-Lipschitz functions is still enormous in high dimensions: for
two samples in dimension $n > 2$, the empirical $W_1$ error decays only as
$O(N^{-1/n})$ :cite:`Weed.Bach.2019`. In $100$ dimensions, halving that
estimation error would require $2^{100}$ times as much data. Nobody estimates
$W_1$ directly from high-dimensional samples that way. Instead, a WGAN critic
searches over a much smaller class of neural networks, while Sinkhorn
iterations handle smaller discrete matching problems.

Both kernels and adversarial critics start by drawing samples from the model.
For some generative models, sampling itself is the bottleneck, and
:numref:`sec_mdl-scores` asks how we can compare two distributions without
drawing model samples at all.

## Summary

Overlapping tails do not rescue f-divergences: when two unit Gaussians sit
eight standard deviations apart, both the Jensen--Shannon divergence and total
variation have slopes near $10^{-4}$. Integral probability metrics sidestep
density ratios by comparing how $P$ and $Q$ average a class of test functions,
and the choice of class determines both sample efficiency and how far the
metric can see. Test functions linear in fixed features yield MMD, the
distance between mean feature vectors: it evaluates in closed form as three
kernel averages with no critic to train, and estimates at rate $O(N^{-1/2})$
regardless of dimension, though an RBF kernel goes blind beyond a few length
scales. Functions of slope at most one yield $W_1$, the minimum mass-times-distance
cost of transporting $P$ into $Q$, which maintains slope one at every
separation and underpins the Wasserstein GAN critic. In one dimension, $W_1$
collapses to the area between the two CDFs; in higher dimensions, it requires
either a trained critic or, on small discrete problems, Sinkhorn iterations
with entropic regularization.

## Exercises

1. [code] **MMD in closed form.** Use the Gaussian integral
   :eqref:`eq_mdl-rbf-gaussian-integral` for the RBF kernel with length scale
   $\ell$, with $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(d, 1)$.
    1. Show that
       $\textrm{MMD}^2 = \frac{2\ell}{\sqrt{\ell^2 + 2}}
       \big(1 - e^{-d^2/(2(\ell^2 + 2))}\big)$, and check that $\ell = 1$
       gives the $\textrm{MMD}^2$ column of the opening table. Which length
       scale maximizes $\textrm{MMD}^2$ for small shifts $d$? What happens
       as $\ell \to 0$ and as $\ell \to \infty$?
    1. Evaluate the slope in $d$ at $d = 8$ for $\ell = 1$, $4$ and $16$.
       What do the three values say about the length scale for a model that
       starts far from its data?
   :exlabel:`ex_mdl-divergences-distances-q-mathcaln-delta`

1. [code] **One-dimensional transport.** ● On the real line, $W_1$ is the
   area between the CDFs :eqref:`eq_mdl-w1-cdf`.
    1. For two empirical distributions on samples $x_1, \ldots, x_N$ and
       $y_1, \ldots, y_N$ of equal size, use the monotone plan of the proof
       to show that $W_1 = \tfrac{1}{N} \sum_i |x_{(i)} - y_{(i)}|$, the mean
       absolute difference of the sorted values. Check this against the
       linear program of the six-point example on two random samples of
       size $7$.
    1. Compute the sorted-sample $W_1$ between `x_p` and `y_shift` and
       between `x_p` and `y_same` of the MMD experiment. Then average it
       over $20$ pairs of independent samples of size $N$ from
       $\mathcal{N}(0, 1)$, for $N \in \{10^2, 10^3, 10^4\}$, and find the
       rate at which it decreases. The estimate is never negative. Why does
       this rule out unbiasedness when $P = Q$?
    1. Change `q_w` to $(0.40, 0.05, 0.05, 0.10, 0.10, 0.30)$, so that the
       two CDFs cross, and rerun the cell that computes $W_1$. Build the test
       function $g$ with $g(0) = 0$ and slope
       $\textrm{sign}\big(F_Q(t) - F_P(t)\big)$ between neighboring atoms, as
       in Step 4 of the proof, and check that
       $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$ equals the LP value. For every
       route $(i, j)$ that carries mass in the LP solution `plan_lp`, compare
       $g(x_i) - g(x_j)$ with $|x_i - x_j|$. Why must the two agree on every
       such route for any optimal plan and any optimal test function?
    1. *Uniqueness under the squared cost.* Suppose a plan moves mass from
       $x$ to $y'$ and from $x'$ to $y$, with $x < x'$ and $y < y'$, so that
       the two ribbons cross. Show that under the squared cost, sending the
       same mass from $x$ to $y$ and from $x'$ to $y'$ saves
       $2(x' - x)(y' - y)$ per unit of mass, and deduce that the monotone
       plan is the only optimal one. What does the same exchange save under
       the cost $|x - y|$ when all four moves go rightward? Find two ribbons
       that cross in `plan_lp`, the plan that the linear program returned for
       the six-point example, and check your answer on them.
   :exlabel:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`

1. **Optimal critics on the line.** Step 4 of the proof of
   :eqref:`eq_mdl-w1-cdf` shows that a test function with slope
   $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ attains $W_1$.
    1. For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(d, 1)$ with $d > 0$,
       show that $F_P > F_Q$ everywhere, so that Step 4 returns
       $g(x) = -x$, the critic that the text found by inspection.
    1. For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(0, 4)$, show that
       the two CDFs cross only at $0$, that $g(x) = -|x|$ is optimal, and
       that $W_1 = \sqrt{2/\pi} \approx 0.80$. (For
       $X \sim \mathcal{N}(0, \sigma^2)$, $\mathbb{E}|X| = \sigma\sqrt{2/\pi}$.)
       The linear kernel reported $0$ on this pair. Which way does the
       critic move the samples of $Q$?
    1. Let $P = \tfrac12 \delta_0 + \tfrac12 \delta_3$ and
       $Q = \tfrac12 \delta_1 + \tfrac12 \delta_3$. Find $W_1(P, Q)$ and
       describe every optimal critic of the form used in Step 4. Then show
       in general that such a critic is optimal if and only if
       $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ at almost every $t$
       with $F_P(t) \neq F_Q(t)$, the points that mass has to cross, while
       its slope elsewhere is free.
    1. A WGAN critic is often trained with a penalty on
       $(|g'(x)| - 1)^2$ at points $x$ between the data and the generated
       samples :cite:`Gulrajani.Ahmed.Arjovsky.ea.2017`. Explain this choice
       by (c). Which critics are optimal when $P = Q$, and which of them does
       the penalty prefer? :numref:`sec_gan_convergence` shows what follows
       for training.
   :exlabel:`ex_mdl-test-functions-transport-optimal-critics`

1. [code] **$W_2$ between Gaussians and the Fréchet distance.** On the
   line the monotone plan $X = F_P^{-1}(U)$, $Y = F_Q^{-1}(U)$ is optimal
   for the squared cost.
    1. Show that for $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and
       $Q = \mathcal{N}(\mu_2, \sigma_2^2)$ the monotone plan is
       $Y = \mu_2 + (\sigma_2/\sigma_1)(X - \mu_1)$, and deduce that
       $W_2^2(P, Q) = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$. With the
       same plan, recover $W_1 = |\mu_1 - \mu_2|$ for equal variances, the
       $W_1$ column of the opening table.
    1. Let both standard deviations be $\sigma$ and the means $d$ apart. Then
       $W_2^2 = d^2$ for every $\sigma$, while the KL divergence between the
       two Gaussians is $d^2/(2\sigma^2)$. What do the two report as
       $\sigma \to 0$, when the Gaussians become the point masses of
       :numref:`tab_mdl-point-mass-table`?
    1. In $\mathbb{R}^n$ the squared $W_2$ between
       $\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and
       $\mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ is the
       squared *Fréchet distance*
       $\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2 + \textrm{tr}\big(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2\,(\boldsymbol{\Sigma}_1^{1/2} \boldsymbol{\Sigma}_2 \boldsymbol{\Sigma}_1^{1/2})^{1/2}\big)$
       :cite:`Dowson.Landau.1982`. Check that it reduces to (a) for $n = 1$
       and, for diagonal covariance matrices, to the sum of (a) over the
       coordinates. Prove the diagonal case directly: the squared cost is a
       sum over coordinates, so no plan costs less than the sum of the
       coordinatewise optima, and the monotone plans of the coordinates
       together attain it.
    1. The Fréchet inception distance fits a Gaussian to each of two feature
       distributions and reports the squared Fréchet distance between the
       fits. Let $P$ be the mixture
       $0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$ and
       $Q = \mathcal{N}(-0.8, 3.72)$, which has the same mean and variance.
       What is the Fréchet distance between the two fits? Estimate
       $W_2^2(P, Q)$ from $10^5$ sorted draws of each (you should find about
       $0.63$). What does a Fréchet distance of zero certify, and what does
       it not?
   :exlabel:`ex_mdl-test-functions-transport-w2-gaussians`

1. [code] **Entropic transport and Sinkhorn divergences.** ●● Regularize the
   transport LP of the six-point example by the divergence from the
   independent plan,

   $$
   \textrm{OT}_\varepsilon(P, Q) = \min_{\gamma}\, \sum_{i, j} \gamma_{ij} C_{ij}
   + \varepsilon\, D_{\textrm{KL}}\big(\gamma \,\|\, p\, q^\top\big),
   $$

   where $\gamma$ ranges over the plans with row sums $p$ and column sums
   $q$.
    1. Show that
       $D_{\textrm{KL}}(\gamma \,\|\, p\, q^\top) = -H(\gamma) + H(P) + H(Q)$
       for every such plan, where $H$ denotes the entropy, so that `sinkhorn`
       returns the minimizer. Show that
       $\textrm{OT}_\varepsilon(P, Q) \to \sum_{i, j} p_i q_j C_{ij}$
       as $\varepsilon \to \infty$, and that
       $\textrm{OT}_\varepsilon(P, P) > 0$ for every $\varepsilon > 0$
       unless $P$ is a point mass. With $C_{ij} = |x_i - x_j|$, show that
       the distributions $P$ on the atoms that minimize the limit
       $\sum_{i, j} p_i q_j C_{ij}$ put all their mass on medians of $Q$,
       for `q_w` the atoms $3$ and $4$. What would a model trained on
       $\textrm{OT}_\varepsilon$ with a large $\varepsilon$ learn?
    1. The Sinkhorn divergence
       $S_\varepsilon(P, Q) = \textrm{OT}_\varepsilon(P, Q)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(P, P)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(Q, Q)$ removes this bias
       :cite:`Genevay.Peyre.Cuturi.2018,Feydy.Sejourne.Vialard.ea.2019`.
       Compute $S_\varepsilon(P, Q)$ for `p_w` and `q_w` with the absolute
       cost `C` at $\varepsilon \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$. At
       $\varepsilon = 10^{-3}$ the entries of $\mathbf{K}$ underflow, so
       carry out the Sinkhorn updates on $\log \mathbf{a}$ and
       $\log \mathbf{b}$ there, with `scipy.special.logsumexp`. Check that
       $S_\varepsilon$ approaches $W_1$ as $\varepsilon \to 0$ and
       $\tfrac{1}{2} \mathcal{E}(P, Q)$ as $\varepsilon \to \infty$, where
       $\mathcal{E}(P, Q) = 2\mathbb{E}|X - Y| - \mathbb{E}|X - X'| - \mathbb{E}|Y - Y'|$
       is the energy distance :cite:`Szekely.Rizzo.2013`, the expression
       :eqref:`eq_mdl-mmd2` with $-|x - y|$ in place of the kernel. Derive
       the second limit from (a).
   :exlabel:`ex_mdl-divergences-distances-sinkhorn-divergences`

1. [code] **Sinkhorn iterations at small $\varepsilon$.** ● Use `sinkhorn`,
   `p_w` and `q_w` of the six-point example.
    1. *Stopping early.* Call `sinkhorn` with the squared cost `C2` at
       $\varepsilon = 0.1$ and `max_iters=5000`, so that the loop ends before
       the column sums have settled. The cost that it returns is below the
       optimum $3.5$. Print the column sums of the returned matrix minus
       `q_w`. Which destinations receive too little, and why does that lower
       the cost? Why can no matrix with row sums `p_w` and column sums `q_w`
       cost less than $3.5$?
    1. *The number of iterations.* Count the iterations that `sinkhorn`
       needs with `C2` at
       $\varepsilon \in \{2, 1, 0.5, 0.3, 0.2, 0.15, 0.1\}$, and plot the
       logarithm of the count against $1/\varepsilon$. What kind of growth do
       the points at small $\varepsilon$ show, and what count does it predict
       for $\varepsilon = 0.05$?
    1. *Ties.* Under the absolute cost `C` every plan that moves mass only
       rightward is optimal for `p_w` and `q_w`. Run `sinkhorn` with `C` at
       $\varepsilon = 0.1$ and $0.02$. How many routes carry mass? Show that
       the limit plan has the form $\gamma_{ij} = a_i b_j$ for $j \geq i$
       and $0$ otherwise. On plans the regularizer is $-\varepsilon$ times
       the entropy plus a constant. Use this to explain why the limit is the
       plan of largest entropy among the optimal plans.
   :exlabel:`ex_mdl-test-functions-transport-sinkhorn-iterations`

<!-- Discussions: forum topic to be created at deploy -->

[Discussions](https://d2l.discourse.group/t/36895)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.4]{.kicker}

Test Functions and Transport<br>**comparing averages, and moving mass**
:::
:::

::: {.slide title="Eight Standard Deviations Apart, JS and TV Have Slopes Near 10⁻⁴"}
[The problem]{.kicker}

Data $P=\mathcal N(0,1)$, model $Q=\mathcal N(d,1)$. Slope of each comparison in $d$, the gradient that the model receives:

| slope in $d$ | $d=0.5$ | $d=2$ | $d=8$ |
|:---|---:|---:|---:|
| KL divergence | $0.5$ | $2$ | $8$ |
| Jensen–Shannon divergence (JS) | $0.12$ | $0.22$ | $0.0002$ |
| total variation (TV) | $0.39$ | $0.24$ | $0.0001$ |
| kernel distance $\textrm{MMD}^2$, length scale $1$ | $0.18$ | $0.40$ | $0.00007$ |
| transport distance $W_1$ | $1$ | $1$ | $1$ |

::: {.d2l-note .warn}
Overlapping tails do not help. At $d=8$ only KL and $W_1$ say which way to move, and KL needs the model's density.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- say why total variation cannot be estimated from two samples;
- compute MMD as a distance between mean features, and as three kernel averages;
- read a witness function;
- compute $W_1$ on the line as the area between two CDFs, and state its dual;
- choose between a fixed kernel and a trained critic.

**Running examples:** $\mathcal N(0,1)$ against $\mathcal N(0,4)$; two distributions on the points $0,\dots,5$.

**The idea:** compare how $P$ and $Q$ average a class of functions. The class decides how far the comparison sees.
:::

::: {.slide title="Averages and Transport Take Over Where the Ratio Fails"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Entropy, Cross-Entropy, and KL | surprise, KL, the loss floor |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| f-Divergences | one convex function of the ratio |
| **Test Functions and Transport** | **averages and moving mass** |
| Score-Based Comparisons | slopes, no normalizer |
| Mutual Information | dependence and what it limits |
| Estimating Mutual Information | what samples can show |
:::

::: {.col .narrow}
- **Uses:** total variation, the critic bound, Cauchy–Schwarz.
- **Skips:** duality off the line; how a critic is kept Lipschitz; proofs of the estimation rates.
:::
:::
:::

::: {.slide title="An IPM Is the Largest Gap in Averages That a Class of Functions Finds"}
[Test functions]{.kicker}

Total variation is $\sup_A\lvert P(A)-Q(A)\rvert$, a supremum over indicator functions. Replace them by a class $\mathcal G$ of test functions $g$:

$$\textrm{IPM}_{\mathcal G}(P,Q)=\sup_{g\in\mathcal G}\big(\mathbb{E}_P[g]-\mathbb{E}_Q[g]\big)$$

| class $\mathcal G$ | integral probability metric (IPM) |
|:--|:--|
| $\lvert g\rvert\le\tfrac12$ (an indicator minus $\tfrac12$) | total variation |
| linear in fixed features, weights in the unit ball | maximum mean discrepancy (MMD) |
| slope at most $1$ | Wasserstein distance $W_1$ |

::: {.d2l-note}
The class is a panel of auditors. It decides what the comparison can see.
:::
:::

::: {.slide title="Check: Total Variation Between Two Samples"}
[Check]{.kicker}

Draw $1{,}000$ points from $\mathcal N(0,1)$, twice. What is the total variation between the two empirical distributions?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1$. The samples share no point, so the indicator of the first sample averages $1$ on it and $0$ on the other.
A class with functions this sharp finds gaps that are not there.
:::
:::
:::

::: {.slide title="A Class Must Be Estimable, and Its Functions Must Grow"}
[Two requirements]{.kicker}

- **Estimable:** sample averages must track true averages. Indicators fail: empirical TV is $1$.
- **Far-sighted:** the gap must keep growing with distance. Bounded $g$ fail: gap $\le1$.

$\mathcal N(0,1)$ against $\mathcal N(d,1)$: the best event is $\{x<d/2\}$, so $\textrm{TV}=2\Phi(d/2)-1$, with $\Phi$ the standard normal CDF.

| $d$ | $1$ | $2$ | $4$ | $8$ |
|:--|--:|--:|--:|--:|
| TV | $0.383$ | $0.683$ | $0.954$ | $0.99994$ |

::: {.d2l-note .rule}
Total variation fails both. A fixed kernel will meet the first, functions of slope one the second.
:::
:::

::: {.slide title="MMD Is the Distance Between Mean Features"}
[Mean features]{.kicker}

Test pair: $P=\mathcal N(0,1)$, $Q=\mathcal N(0,4)$.

**I. Linear functions** $g(x)=wx$, $\lvert w\rvert\le1$: the gap is $\lvert\mathbb{E}_PX-\mathbb{E}_QX\rvert=0$. Blind to spread.

. . .

**II. Features** $\varphi(x)=(x,x^2)$, $g=\mathbf w^\top\varphi$, $\lVert\mathbf w\rVert\le1$, mean feature $\mathbf m_P=\mathbb{E}_P[\varphi]$:

$$\textrm{MMD}(P,Q)=\sup_{\lVert\mathbf w\rVert\le1}\mathbf w^\top(\mathbf m_P-\mathbf m_Q)=\lVert\mathbf m_P-\mathbf m_Q\rVert$$

. . .

$\mathbf m_P=(0,1)$ and $\mathbf m_Q=(0,4)$, so $\;\boxed{\textrm{MMD}^2=0^2+3^2=9}$

[Cauchy–Schwarz: a linear function on the unit ball is largest along its coefficient vector.]{.small}
:::

::: {.slide title="Three Kernel Averages Replace the Supremum"}
[Kernels]{.kicker}

**III. Expand the square.** Kernel $k(x,x')=\langle\varphi(x),\varphi(x')\rangle$; $x,x'\sim P$ and $y,y'\sim Q$ independent:

$$\textrm{MMD}^2=\mathbb{E}\,k(x,x')+\mathbb{E}\,k(y,y')-2\,\mathbb{E}\,k(x,y)$$

. . .

**IV. Choose the kernel,** not the features. RBF: $k(x,x')=e^{-(x-x')^2/(2\ell^2)}$, similar within a length scale $\ell$; infinitely many features.

. . .

Test pair, $\ell=1$: $\mathbb{E}\,e^{-Z^2/2}=(1+s^2)^{-1/2}$ for $Z\sim\mathcal N(0,s^2)$; $x-x'$, $y-y'$, $x-y$ have $s^2=2,\,8,\,5$:

$$\boxed{\textrm{MMD}^2=3^{-1/2}+9^{-1/2}-2\cdot6^{-1/2}=0.094}$$
:::

::: {.slide title="Between Equal Distributions the RBF Estimate Stays Within Thousandths of Zero"}
[Estimation]{.kicker}

Unbiased estimate: average $k$ over pairs, leaving out the pairs of a point with itself. $1{,}000$ points per sample:

@!divergences-distances-mmd

::: {.d2l-note}
No critic, no density. The estimate detects a gap far better than it measures one: a two-sample test.
:::
:::

::: {.slide title="The Witness Is Flat in the Tails, So Distant Samples Get No Push"}
[The witness]{.kicker}

::: {.cols .vc}
::: {.col}
- The auditor that finds the gap: $g^\star(x)\propto\mathbb{E}_P\,k(x,X)-\mathbb{E}_Q\,k(x,Y)$.
- $\mathcal N(0,1)$ against $\mathcal N(0.5,1)$: by symmetry the sign change is at $0.25$; a sample's crossing is off by a tenth or two.
- A model's sample moves uphill on $g^\star$. The slope there is its push.
- A few length scales out the slope is gone.
:::

::: {.col}
@!mdl-divergences-distances-integral-probability-metrics-and-mmd-1
:::
:::
:::

::: {.slide title="W₁ Is the Least Mass × Distance over All Transport Plans"}
[Optimal transport]{.kicker}

- Plan $\gamma$: how much mass goes from each $x$ to each $y$; a joint distribution of $(x,y)$ with marginals $P$, $Q$.
- $W_1(P,Q)=\inf_\gamma\mathbb{E}_\gamma\lvert x-y\rvert$
- Point masses $\delta_0$, $\delta_d$: one plan, $W_1=d$.
- $\mathcal N(0,1)$, $\mathcal N(d,1)$: shifting every point by $d$ costs $d$.
- No plan is cheaper: $\mathbb{E}_\gamma\lvert x-y\rvert\ge\lvert\mathbb{E}x-\mathbb{E}y\rvert=d$.

::: {.d2l-note}
$W_1$ is the distance itself: slope $1$ at every $d$.
:::
:::

::: {.slide title="The Dual of W₁ Is a Critic of Slope at Most One"}
[Kantorovich–Rubinstein duality]{.kicker}

$$W_1(P,Q)=\sup_{g}\big(\mathbb{E}_P[g]-\mathbb{E}_Q[g]\big)\quad\textrm{over all } g \textrm{ with } \lvert g(x)-g(y)\rvert\le\lVert x-y\rVert$$

- Easy half: $g(x)-g(y)\le\lVert x-y\rVert$; average over any plan $\gamma$.
- Such a $g$ cannot jump, but it can climb: the gap grows with distance.
- $\mathcal N(0,1)$, $\mathcal N(d,1)$: $g(x)=-x$ has gap $d$, and slope $1$ at every sample.
- A Wasserstein GAN trains a network as $g$, kept approximately 1-Lipschitz.

::: {.d2l-note .rule}
An infimum over joint distributions becomes a supremum over one function.
:::
:::

::: {.slide title="On the Line, W₁ Is the Area Between the CDFs"}
[Transport on the line]{.kicker}

- **I.** $F_P(t)$: the mass of $P$ at or left of $t$. Any plan carries at least $\lvert F_P(t)-F_Q(t)\rvert$ across $t$.
- **II.** A parcel moved from $x$ to $y$ crosses every $t$ between: cost $=\int(\textrm{mass crossing }t)\,dt$.
- **III.** The *monotone plan* sends quantile to quantile, $F_P^{-1}(u)\to F_Q^{-1}(u)$, and never carries mass both ways.

$$\boxed{W_1(P,Q)=\int\lvert F_P(t)-F_Q(t)\rvert\,dt}$$

::: {.d2l-note}
No optimization: sort, and add up the gaps.
:::
:::

::: {.slide title="On Six Points the Area Between the CDFs Is 1.7"}
[Worked example]{.kicker}

Left: the monotone plan and the CDFs of $P$, $Q$. Right: point masses $d$ apart.

@fig:mdl-it-ot-transport-plan
:::

::: {.slide title="A 36-Route Linear Program Returns 1.7 by a Different Plan"}
[Worked example]{.kicker}

| point | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ |
|:--|--:|--:|--:|--:|--:|--:|
| $p$ | $0.30$ | $0.20$ | $0.25$ | $0.10$ | $0.10$ | $0.05$ |
| $q$ | $0.05$ | $0.10$ | $0.10$ | $0.25$ | $0.20$ | $0.30$ |
| $F_P-F_Q$ | $0.25$ | $0.35$ | $0.50$ | $0.35$ | $0.25$ | $0$ |

@!divergences-distances-w1

::: {.d2l-note}
$F_P\ge F_Q$, so the best critic is $g(x)=-x$: gap $\mathbb{E}_QX-\mathbb{E}_PX=3.35-1.65=1.7$.
Every plan that moves mass only rightward costs the same $1.7$. The monotone plan is one of many.
:::
:::

::: {.slide title="Check: W₁ on Three Points"}
[Check]{.kicker}

$P=(0.6,\,0.3,\,0.1)$ and $Q=(0.2,\,0.5,\,0.3)$ on the points $0,1,2$. Total variation is $0.4$. What is $W_1$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $F_P=(0.6,\,0.9,\,1)$ and $F_Q=(0.2,\,0.7,\,1)$. The gaps are $0.4$ and $0.2$, so $W_1=0.6$.
Move the points twice as far apart: $W_1$ doubles, total variation stays $0.4$.
:::
:::
:::

::: {.slide title="Entropy Turns Transport into Row and Column Scalings"}
[Entropic transport]{.kicker}

- Plan $\boldsymbol\gamma$: a matrix with row sums $\mathbf p$, column sums $\mathbf q$. Cost $\sum_{ij}\gamma_{ij}C_{ij}$, with $C_{ij}$ the cost of route $i\to j$.
- A linear program in $N^2$ unknowns for $N$ points.
- Subtract $\varepsilon$ times the entropy of $\boldsymbol\gamma$. The minimizer is

$$\boldsymbol\gamma_\varepsilon=\mathrm{diag}(\mathbf a)\,\mathbf K\,\mathrm{diag}(\mathbf b),\qquad K_{ij}=e^{-C_{ij}/\varepsilon}$$

- **Sinkhorn:** $\mathbf a\leftarrow\mathbf p/(\mathbf K\mathbf b)$, $\mathbf b\leftarrow\mathbf q/(\mathbf K^\top\mathbf a)$, repeat.

::: {.d2l-note}
Two matrix–vector products per step. Small $\varepsilon$: closer to the optimum, slower to converge.
:::
:::

::: {.slide title="Ten Sharp Routes at ε = 0.1 Take Sinkhorn 150,000 Iterations" layout="tight"}
[Entropic transport]{.kicker}

Six points, squared cost: the ten routes of the monotone plan cost $3.5$. Stop when the column sums match $\mathbf q$ to $10^{-9}$.

@!divergences-distances-sinkhorn
:::

::: {.slide title="Kernels Are Cheap and Nearsighted; Critics See Far and Need Training"}
[Kernel or critic]{.kicker}

| | MMD, fixed kernel | $W_1$, critic of slope $\le1$ |
|:--|:--|:--|
| cost | three averages over pairs | train a network, or solve transport |
| error from $N$ points in dimension $n$ | $N^{-1/2}$ for every $n$ | $N^{-1/n}$ for $n>2$ |
| slope at $d=8$ | $0.00007$ | $1$ |
| use | two-sample tests, scoring models | a model far from its data |

::: {.d2l-note .warn}
In $100$ dimensions, halving the error of $W_1$ between samples takes $2^{100}$ times the data. A critic network searches a much smaller class.
:::

[Rates: Gretton et al. (2012); Weed and Bach (2019).]{.small}
:::

::: {.slide title="Check: W₁ Between N(0, 1) and N(0, 4)"}
[Check]{.kicker}

The linear kernel reported $0$ on this pair. The monotone plan sends $x$ to $2x$. What is $W_1$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $W_1=\mathbb{E}\lvert 2X-X\rvert=\mathbb{E}\lvert X\rvert=\sqrt{2/\pi}\approx0.80$.
The best critic is $g(x)=-\lvert x\rvert$: it tells every sample of the wider model to move inward.
:::
:::
:::

::: {.slide title="Recap: Averages Can Be Estimated, and Transport Sees Every Distance"}
[Recap]{.kicker}

| comparison | what it is | on the running examples | how far it sees |
|:--|:--|:--|:--|
| JS, TV | functions of the ratio $p/q$ | slopes near $10^{-4}$ at $d=8$ | overlap only |
| MMD | distance between mean features: three kernel averages | $0$, $9$, $0.094$ for linear, quadratic, RBF | a few length scales |
| $W_1$ | least mass × distance; best critic of slope $\le1$ | $1.7$: five CDF gaps, or a linear program | every distance |
| Sinkhorn | transport with an entropy term | $\varepsilon=0.1$: ten routes, cost $3.50002$ against $3.5$, after $150{,}000$ iterations | as transport, blurred by $\varepsilon$ |

**Next:** a model known only up to its normalizing constant offers neither a density nor cheap samples. Its slopes can still be compared.
:::
