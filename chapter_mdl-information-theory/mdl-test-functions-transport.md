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
\begin{aligned}
\mathbb{E}_\gamma|X - Y|
&= \int \Big( \gamma(X \leq t < Y) + \gamma(Y \leq t < X) \Big)\, dt \\
&\geq \int \big| \gamma(X \leq t < Y) - \gamma(Y \leq t < X) \big|\, dt.
\end{aligned}
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
increased the work by a factor of more than $800$. Sinkhorn iterations are fast
only when you can afford a blurred transport plan (Exercise
:exref:`ex_mdl-test-functions-transport-sinkhorn-iterations`b tracks this
growth quantitatively).

## Fixed Kernels and Trained Critics

MMD and $W_1$ split our two requirements right down the middle. A fixed kernel
defines a small, smooth function class: MMD reduces to three sample averages,
and its estimation error from $N$ points decays as $O(N^{-1/2})$ in *any*
dimension :cite:`Gretton.Borgwardt.Rasch.ea.2012`. Kernels do not escape the
curse of dimension that way, unfortunately. The error rate is dimension-free,
but the size of the signal is not: with a fixed kernel, the population MMD
between two given distributions typically shrinks as the dimension grows
:cite:`Ramdas.Reddi.Poczos.ea.2015`, which is why the kernel inception
distance and MMD losses are computed on learned features, not on raw pixels.
A fixed kernel also loses sight of the data beyond a few length scales. By
contrast, 1-Lipschitz functions provide a
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
regardless of dimension, though the MMD itself tends to shrink as the
dimension grows and an RBF kernel goes blind beyond a few length scales. Functions of slope at most one yield $W_1$, the minimum mass-times-distance
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

[Discussions](https://d2l.discourse.group/t/36895)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.4]{.kicker}

Test Functions and Optimal Transport<br>**integral probability metrics · kernel MMD · witness functions · Wasserstein-1 · Kantorovich–Rubinstein · Sinkhorn**
:::
:::

::: {.slide title="Does Tail Overlap Rescue Density-Ratio Losses?"}
[Motivation: The Shifted-Gaussian Diagnostic]{.kicker}

::: {.cols .vc}
::: {.col}
Compare overlapping shifted Gaussians $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(d, 1)$ separated by $d = 8$ standard deviations:

- **Smooth tails do not rescue JS, TV, or local MMD:** By $d = 8$, $\textrm{JS} \approx \log 2$ and $\textrm{TV} \approx 1$ have saturated, collapsing $\partial\textrm{JS}/\partial d$, $\partial\textrm{TV}/\partial d$, and $\partial\textrm{MMD}^2/\partial d$ ($\ell = 1$) to $\sim 10^{-4}$!
- **Wasserstein-1 ($W_1$) stays steady:** Maintains unit slope $\partial W_1/\partial d = 1.00$ at every distance $d$ without needing an explicit model density $q(x)$.
:::

::: {.col .fig .big}
@fig:ch37-info-shifted-gaussian-slopes
:::
:::
:::

::: {.slide title="Evaluating Values and Gradients at Separation d"}
[Five Comparisons on Shifted Gaussians]{.kicker}

Evaluating all five comparisons and their central-difference derivatives in $d$ reveals how quickly bounded ratios and local kernels lose sight of distant data:

@!mdl-test-functions-transport-test-functions-and-transport-3
:::

::: {.slide title="Integral Probability Metrics: A Panel of Auditors"}
[Unifying Test-Function Comparisons]{.kicker}

::: {.cols .vc}
::: {.col}
An **Integral Probability Metric (IPM)** compares $P$ and $Q$ across a symmetric class $\mathcal{G}$ of test functions ("auditors") $g$:

$$\textrm{IPM}_{\mathcal{G}}(P, Q) = \sup_{g \in \mathcal{G}}\Big(\mathbb{E}_{x \sim P}[g(x)] - \mathbb{E}_{y \sim Q}[g(y)]\Big)$$

- **Three choices of $\mathcal{G}$:** Bounded $|g| \le \tfrac12$ gives **TV**; RKHS unit ball $\|\mathbf{w}\|_2 \le 1$ gives **MMD** (closed-form); 1-Lipschitz $\textrm{Lip}_1$ gives **Wasserstein $W_1$** (WGAN).
:::

::: {.col .fig}
@fig:ch37-info-ipm-auditor-classes
:::
:::
:::

::: {.slide title="Two Competing Requirements on Test Class G"}
[Sample Efficiency vs. Far-Sighted Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
- **Failure 1 ($\mathcal{G}$ too rich / spiky):** For $|g(x)| \le \tfrac12$, two independent $N$-samples from $\mathcal{N}(0, 1)$ share no points almost surely, so $g(x) = \mathbf{1}_{x \in \{X_i\}} - \tfrac12$ gives empirical $\textrm{TV}(\hat{P}_N, \hat{Q}_N) = 1.00$ even when $P = Q$!
- **Failure 2 ($\mathcal{G}$ bounded in height):** Any class with $|g(x)| \le M$ caps the gap at $2M$, saturating with zero gradient on distant modes.
- **1-Lipschitz resolution:** Bounded slope $|\nabla g| \le 1$ prevents sample memorization *and* grows linearly with distance $d$.
:::

::: {.col .fig .big}
@fig:ch37-info-test-class-goldilocks
:::
:::
:::

::: {.slide title="From Linear and Quadratic Features to Mean Maps"}
[Building Maximum Mean Discrepancy]{.kicker}

::: {.cols .vc}
::: {.col}
With feature map $\boldsymbol{\varphi}(x)$ and mean embedding $\mathbf{m}_P = \mathbb{E}_{x \sim P}[\boldsymbol{\varphi}(x)]$, Cauchy–Schwarz solves the supremum over $\|\mathbf{w}\|_2 \le 1$ analytically:

$$\textrm{MMD}(P, Q) = \sup_{\|\mathbf{w}\|_2 \le 1} \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q) = \|\mathbf{m}_P - \mathbf{m}_Q\|_2$$

- **Feature richness:** Linear $\varphi(x) = x$ is blind to variance on $\mathcal{N}(0,1)$ vs. $\mathcal{N}(0,4)$, whereas $\boldsymbol{\varphi}(x) = (x, x^2)$ separates them ($\|\mathbf{m}_P - \mathbf{m}_Q\|_2 = 3$), and the characteristic RBF kernel embeds all moments $j = 0, \ldots, \infty$.
:::

::: {.col .fig}
@fig:ch37-info-mmd-mean-embedding
:::
:::
:::

::: {.slide title="The Kernel Trick Eliminates the Supremum"}
[Closed-Form Pairwise Expectations]{.kicker}

::: {.cols .vc}
::: {.col}
Expanding $\|\mathbf{m}_P - \mathbf{m}_Q\|_2^2$ via $k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x') \rangle$ yields three pairwise kernel expectations:

$$\textrm{MMD}^2(P, Q) = \mathbb{E}_{P}[k(x, x')] + \mathbb{E}_{Q}[k(y, y')] - 2\mathbb{E}_{P, Q}[k(x, y)]$$

- **No critic to train:** Within-group similarities minus twice cross-group similarity cancel to $0$ iff $P = Q$ (for characteristic kernels).
:::

::: {.col .fig}
@fig:ch37-info-mmd-kernel-expansion
:::
:::
:::

::: {.slide title="Characteristic Kernels and Closed-Form RBF MMD"}
[Infinite-Dimensional Feature Spaces]{.kicker}

Any finite polynomial kernel (such as $k(x, x') = xx' + x^2 x'^2$) is blind to higher moments. Choosing the **RBF (Gaussian) kernel** $k(x, x') = \exp\!\big(-\frac{\|x - x'\|^2}{2\ell^2}\big)$ with length scale $\ell > 0$ compares all moments simultaneously:

- **Infinite feature expansion:** Factoring $e^{-(x-x')^2/(2\ell^2)} = e^{-x^2/(2\ell^2)}e^{-x'^2/(2\ell^2)}\sum_{j=0}^\infty \frac{(xx'/\ell^2)^j}{j!}$ reveals an inner product over infinitely many damped monomial features $\varphi_j(x) = \frac{x^j}{\ell^j\sqrt{j!}}e^{-x^2/(2\ell^2)}$.
- **Characteristic property:** Because these features separate all distributions, $\textrm{MMD}(P, Q) = 0 \iff P = Q$, making RBF-MMD a true metric on probability distributions (Gretton et al., 2012).
- **Closed form on Gaussians ($\ell = 1$):** Using $\mathbb{E}[e^{-Z^2/(2\ell^2)}] = (1 + s^2/\ell^2)^{-1/2}e^{-\mu^2/(2(\ell^2+s^2))}$ for $Z \sim \mathcal{N}(\mu, s^2)$:
  - On $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(0, 4)$, $x - x' \sim \mathcal{N}(0, 2)$, $y - y' \sim \mathcal{N}(0, 8)$, and $x - y \sim \mathcal{N}(0, 5)$, giving $\textrm{MMD}^2 = \frac{1}{\sqrt{3}} + \frac{1}{\sqrt{9}} - \frac{2}{\sqrt{6}} \approx 0.0942$.
  - On shifted Gaussians $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(d, 1)$, $\textrm{MMD}^2 = \frac{2}{\sqrt{3}}(1 - e^{-d^2/6})$, which saturates as $d \gg \ell$.
:::

::: {.slide title="Unbiased U-Statistic Estimation from Samples"}
[Two-Sample Testing without a Critic]{.kicker}

Given finite samples $\{x_i\}_{i=1}^n \sim P$ and $\{y_j\}_{j=1}^m \sim Q$, omitting diagonal self-pairs ($i = j$, which would bias within-sample similarities upward by $k(x_i, x_i)/n = 1/n$) yields an unbiased U-statistic estimator:

@!divergences-distances-mmd
:::

::: {.slide title="The Witness Function and Tail Gradient Vanishing"}
[Why Fixed-Bandwidth MMD Is Nearsighted]{.kicker}

Which test function $g^\star \in \mathcal{G}$ attains the maximum gap in $\textrm{MMD}(P, Q)$?

::: {.cols .vc}
::: {.col}
- **Closed-form witness function:** By Cauchy–Schwarz, $\mathbf{w}^\star \propto \mathbf{m}_P - \mathbf{m}_Q$, so the optimal auditor (the **witness**) is:
  $$g^\star(x) \propto \mathbb{E}_{X \sim P}[k(x, X)] - \mathbb{E}_{Y \sim Q}[k(x, Y)]$$
- **Sign crossing:** Positive where $P$ dominates and negative where $Q$ dominates (crossing $0$ at $x = 0.25$ for $\mathcal{N}(0, 1)$ vs. $\mathcal{N}(0.5, 1)$).
- **Vanishing tail gradients:** A generator pushes samples uphill along $\nabla g^\star(y)$, but beyond a few length scales $\ell$ in the tails, $g^\star(x)$ flattens to $0$!
:::

::: {.col .fig}
@!mdl-divergences-distances-integral-probability-metrics-and-mmd-1
:::
:::
:::

::: {.slide title="Optimal Transport and Wasserstein-1 Distance"}
[Moving Probability Mass]{.kicker}

To keep a steady gradient at *every* distance $d$, we must incorporate the ground distance $\|x - y\|$ of the space. A **transport plan** (or *coupling*) $\gamma \in \Pi(P, Q)$ is a joint distribution over $(x, y)$ whose marginals are $P$ and $Q$:

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}\big[\|x - y\|\big]$$

- **Physical meaning (Monge, 1781):** $\gamma(x, y)$ specifies how much earth moves from source $x$ to pit $y$; $W_1(P, Q)$ is the minimum total **mass $\times$ distance** work required to reshape $P$ into $Q$.
- **Exact distance recovery at every separation $d$:**
  - **Disjoint point masses ($\delta_0$ vs. $\delta_d$):** The unique coupling puts all mass on $(0, d)$, giving $W_1(\delta_0, \delta_d) = |d|$ with gradient $\pm 1$.
  - **Shifted Gaussians ($\mathcal{N}(0, 1)$ vs. $\mathcal{N}(d, 1)$):** Translating $y = x + d$ costs $d$, and by Jensen's inequality $\mathbb{E}_\gamma|x - y| \ge |\mathbb{E}_P[X] - \mathbb{E}_Q[Y]| = |d|$, so $W_1 = |d|$ with constant slope $1$ for all $d > 0$.
:::

::: {.slide title="Visualizing Transport Plans and 1D CDF Areas"}
[Monge–Kantorovich Geometry]{.kicker}

On the real line, reshaping a left-heavy distribution $P$ into a right-heavy distribution $Q$ on atoms $\{0, 1, 2, 3, 4, 5\}$ moves mass across intervals whose vertical gaps $|F_P(t) - F_Q(t)|$ sum to $W_1 = 1.7$:

@fig:mdl-it-ot-transport-plan
:::

::: {.slide title="Kantorovich–Rubinstein Duality: 1-Lipschitz Critics"}
[Primal Linear Program to Dual IPM]{.kicker}

::: {.cols .vc}
::: {.col}
LP strong duality converts the primal optimal-transport infimum over couplings $\gamma \in \Pi(P, Q)$ into a dual IPM supremum over **1-Lipschitz critics**:

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P,Q)}\mathbb{E}_\gamma[\|x - y\|] = \sup_{\|g\|_{\textrm{Lip}} \le 1}\big(\mathbb{E}_P[g(x)] - \mathbb{E}_Q[g(y)]\big)$$

- **Weak duality in one line:** Since $g(x) - g(y) \le \|x - y\|$ pointwise for $\|g\|_{\textrm{Lip}} \le 1$, taking $\mathbb{E}_\gamma$ gives $\mathbb{E}_P[g] - \mathbb{E}_Q[g] \le \mathbb{E}_\gamma[\|x - y\|]$.
:::

::: {.col .fig}
@fig:ch37-info-kr-duality-bridge
:::
:::
:::

::: {.slide title="Exact 1D Optimal Transport: Area Between CDFs"}
[Four-Step Proof on the Real Line]{.kicker}

In one dimension, Kantorovich–Rubinstein duality and primal optimal transport both collapse to the area between the cumulative distribution functions $F_P(t) = \Pr(X \le t)$ and $F_Q(t) = \Pr(Y \le t)$:

$$W_1(P, Q) = \int_{-\infty}^\infty \big|F_P(t) - F_Q(t)\big|\,dt$$

- **Step 1 (Lower bound on any plan):** Writing $|X - Y| = \int \mathbf{1}(t \text{ between } X \text{ and } Y)\,dt$ and taking $\mathbb{E}_\gamma$ shows that net mass crossing $t$ is at least $|\gamma(X \le t < Y) - \gamma(Y \le t < X)| = |F_P(t) - F_Q(t)|$.
- **Step 2 (Monotone quantile coupling attains the area):** Let $U \sim \mathrm{Unif}(0, 1)$ and pair quantiles $X = F_P^{-1}(U)$, $Y = F_Q^{-1}(U)$. Because $X$ and $Y$ never send mass in opposite directions across any $t$, its cost equals $\int |F_P(t) - F_Q(t)|\,dt$.
- **Step 3 & 4 (Optimal 1-Lipschitz dual critic):** By Fubini's theorem, $\mathbb{E}_P[g] - \mathbb{E}_Q[g] = \int_{-\infty}^\infty g'(t)(F_Q(t) - F_P(t))\,dt$, which attains $\int |F_P(t) - F_Q(t)|\,dt$ at slope $g'(t) = \textrm{sign}(F_Q(t) - F_P(t)) \in \{\pm 1\}$. $\blacksquare$
:::

::: {.slide title="Verifying 1D Transport: CDF Area vs. 36-Var LP"}
[Six-Point Worked Example]{.kicker}

On the 6-point benchmark, summing the five CDF gaps ($0.25 + 0.35 + 0.50 + 0.35 + 0.25 = 1.7$), solving the 36-variable linear program, and evaluating the monotone quantile coupling all return $W_1 = 1.7000$:

@!divergences-distances-w1
:::

::: {.slide title="Squared Cost W₂ and Fréchet Inception Distance"}
[Strict Convexity & Gaussian Closed Form]{.kicker}

::: {.cols .vc}
::: {.col}
Replacing $|x - y|$ with squared distance $\|x - y\|^2$ defines **Wasserstein-2 ($W_2$)**:

$$W_2^2\big(\mathcal{N}(\mu_1, \sigma_1^2), \mathcal{N}(\mu_2, \sigma_2^2)\big) = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$$

- **Strict 1D uncrossing:** For $x < x'$ and $y < y'$, uncrossing saves $2(x' - x)(y' - y) > 0$, making the monotone quantile coupling strictly unique.
- **Fréchet Inception Distance (FID):** Multivariate Gaussian $W_2^2$ evaluates image generators in feature space.
:::

::: {.col .fig}
@fig:ch37-info-w2-uncrossing-fid
:::
:::
:::

::: {.slide title="Entropic Regularization and the Gibbs Kernel"}
[Fast Differentiable Transport]{.kicker}

In dimensions $n \ge 2$, no 1D CDF formula exists and solving an $N^2$-variable linear program costs $\mathcal{O}(N^3 \log N)$. **Entropic optimal transport** (Cuturi, 2013) adds an entropy penalty of strength $\varepsilon > 0$ to encourage smooth plans:

$$\min_{\boldsymbol{\gamma} \in \Pi(\mathbf{p}, \mathbf{q})} \sum_{i,j=1}^N \gamma_{ij} C_{ij} + \varepsilon \sum_{i,j=1}^N \gamma_{ij}(\log \gamma_{ij} - 1)$$

- **Lagrangian diagonal-scaling solution:** Introducing multipliers $\xi_i$ for row sums $\sum_j \gamma_{ij} = p_i$ and $\zeta_j$ for column sums $\sum_i \gamma_{ij} = q_j$ and setting $\frac{\partial \mathcal{L}}{\partial \gamma_{ij}} = C_{ij} + \varepsilon \log \gamma_{ij} - \xi_i - \zeta_j = 0$ yields:
  $$\gamma_{ij} = e^{\xi_i/\varepsilon} e^{-C_{ij}/\varepsilon} e^{\zeta_j/\varepsilon} \iff \boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\,\mathbf{K}\,\textrm{diag}(\mathbf{b}), \qquad K_{ij} = e^{-C_{ij}/\varepsilon}$$
- The optimal regularized coupling is simply the elementwise **Gibbs kernel** $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$ rescaled by positive row and column vectors $\mathbf{a}, \mathbf{b} > \mathbf{0}$!
:::

::: {.slide title="Sinkhorn Matrix Scaling and the ε Trade-Off"}
[Alternating Marginal Projections]{.kicker}

::: {.cols .vc}
::: {.col}
To enforce $\boldsymbol{\gamma}_\varepsilon^\top\mathbf{1} = \mathbf{q}$ and $\boldsymbol{\gamma}_\varepsilon\mathbf{1} = \mathbf{p}$ on $\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\mathbf{K}\textrm{diag}(\mathbf{b})$, **Sinkhorn–Knopp** alternates two $\mathcal{O}(N^2)$ matrix-vector divisions:

$$\mathbf{b} \leftarrow \frac{\mathbf{q}}{\mathbf{K}^\top \mathbf{a}}, \qquad \mathbf{a} \leftarrow \frac{\mathbf{p}}{\mathbf{K}\mathbf{b}}$$

- **Differentiable on GPUs:** Gradients backpropagate directly through unrolled Sinkhorn steps.
- **The $\varepsilon$ trade-off:** Large $\varepsilon$ converges fast but blurs $\boldsymbol{\gamma}_\varepsilon$; small $\varepsilon \to 0$ recovers the exact LP, but the step count grows fast (over $800\times$ from $\varepsilon = 0.5$ to $0.1$).
:::

::: {.col .fig}
@fig:ch37-info-sinkhorn-gibbs-scaling
:::
:::
:::

::: {.slide title="Sinkhorn Convergence Across ε on the 6-Point Grid"}
[Sharpness vs. Iteration Budget]{.kicker}

With squared cost $C_{ij} = (x_i - x_j)^2$ ($W_2^2 = 3.50000$ across $10$ monotone routes), lowering $\varepsilon$ from $2.0$ to $0.1$ sharpens the heatmap into the exact staircase while increasing iterations from $50$ to $\sim 150{,}000$:

@!divergences-distances-sinkhorn
:::

::: {.slide title="Fixed Kernels vs. Critics: Curse of Dimension"}
[Statistical & Computational Trade-Offs]{.kicker}

::: {.cols .vc}
::: {.col}
- **Fixed-Kernel MMD:** Closed-form $\mathcal{O}(N^2)$ U-statistic with **dimension-free sample error $\mathcal{O}(N^{-1/2})$** in any dimension $n$, but tail gradients vanish for $d \gg \ell$.
- **Wasserstein-1 ($W_1$):** Unit gradient at all distances $d$, but empirical sample-to-sample $W_1$ suffers the **curse of dimensionality $\mathcal{O}(N^{-1/n})$** for $n > 2$ (requiring neural critic restriction in WGAN).
:::

::: {.col .fig .big}
@fig:ch37-info-mmd-vs-w1-dimension
:::
:::
:::

::: {.slide title="Concept Check: Empirical TV, 1D W₁, and Critics"}
[Check Your Understanding]{.kicker}

Test your understanding of IPMs and optimal transport:

1. **Empirical Total Variation failure:** You draw two independent samples of $N = 1{,}000$ points each from $\mathcal{N}(0, 1)$. What is the exact Total Variation distance between the two empirical distributions $\hat{P}_N$ and $\hat{Q}_N$, and what does the unbiased RBF-$\textrm{MMD}^2$ estimator report instead?
2. **1D Wasserstein vs. TV under rescaling:** For $P = (0.6, 0.3, 0.1)$ and $Q = (0.2, 0.5, 0.3)$ supported on atoms $\{0, 1, 2\}$, compute $\textrm{TV}(P, Q)$ and $W_1(P, Q)$. What happens to both quantities if the atoms are stretched to $\{0, 2, 4\}$?
3. **Optimal critic for variance shift:** For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(0, 4)$, the 1D monotone coupling maps $X \mapsto Y = 2X$. Use this coupling to compute $W_1(P, Q)$ and identify the optimal 1-Lipschitz critic $g^\star(x)$.
:::

::: {.slide title="Concept Check: Empirical TV, 1D W₁, and Critics (cont.)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. **Empirical TV:** Because continuous draws share zero points almost surely, the indicator of the first sample gives $\textrm{TV}(\hat{P}_N, \hat{Q}_N) = 1.0$, whereas unbiased RBF-$\textrm{MMD}^2$ cancels first-order noise and reports $\approx 0.000 \pm 0.002$.
2. **1D $W_1$ on 3 atoms:** $\textrm{TV} = 0.4$. The CDFs are $F_P = (0.6, 0.9, 1.0)$ and $F_Q = (0.2, 0.7, 1.0)$, with gaps $|F_P - F_Q| = (0.4, 0.2)$ across unit intervals, so $W_1 = 0.4(1) + 0.2(1) = 0.6$. Stretching atoms to $\{0, 2, 4\}$ doubles interval widths ($\Delta x = 2$), doubling $W_1 \to 1.2$ while $\textrm{TV} = 0.4$ stays invariant.
3. **Gaussian variance shift:** Under $Y = 2X$, $|X - Y| = |X|$, so $W_1 = \mathbb{E}_{X \sim \mathcal{N}(0,1)}[|X|] = \sqrt{2/\pi} \approx 0.798$. Because $F_P(t) < F_Q(t)$ for $t < 0$ and $F_P(t) > F_Q(t)$ for $t > 0$, the optimal critic has slope $g'(t) = -\textrm{sign}(t)$, giving $g^\star(x) = -|x|$ (pulling wide samples inward).
:::
:::

::: {.slide title="Summary: Integral Probability Metrics & Transport"}
[Key Takeaways]{.kicker}

| Method / Distance | Mathematical Definition | Sample Complexity / Computation | Geometric Range & Deep Learning Role |
|:--|:---|:---|:---|
| **IPM Template** | $\sup_{g \in \mathcal{G}}(\mathbb{E}_P[g] - \mathbb{E}_Q[g])$ | Governed by richness of auditor class $\mathcal{G}$ | Unifies TV ($|g|\le 1/2$), MMD (RKHS ball), and $W_1$ ($\textrm{Lip}_1$) |
| **Kernel MMD** | $\|\mathbf{m}_P - \mathbf{m}_Q\|_2 = (\mathbb{E}[k_{PP} + k_{QQ} - 2k_{PQ}])^{1/2}$ | $\mathcal{O}(N^{-1/2})$ in any dimension $n$; closed-form U-statistic | Characteristic for RBF kernel; ideal for two-sample tests, flat for $d \gg \ell$ |
| **Wasserstein-1 ($W_1$)** | $\inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_\gamma\|x-y\| = \sup_{\|g\|_{\textrm{Lip}}\le 1}(\mathbb{E}_P g - \mathbb{E}_Q g)$ | 1D: $\int |F_P - F_Q|\,dt$; $n$-D: $\mathcal{O}(N^{-1/n})$ empirical rate | Constant unit slope at all distances $d$; foundation of WGAN |
| **Wasserstein-2 ($W_2$)** | $(\inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_\gamma\|x-y\|^2)^{1/2}$ | Unique monotone plan in 1D; closed-form Gaussian Fréchet distance | Penalizes crossing paths; powers FID evaluation & flow-matching OT |
| **Sinkhorn Entropic OT** | $\min_{\gamma \in \Pi(\mathbf{p},\mathbf{q})} \langle \boldsymbol{\gamma}, \mathbf{C} \rangle + \varepsilon \langle \boldsymbol{\gamma}, \log \boldsymbol{\gamma} - \mathbf{1} \rangle$ | $\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})e^{-\mathbf{C}/\varepsilon}\textrm{diag}(\mathbf{b})$; $\mathcal{O}(N^2)$ per step | Fast, GPU-differentiable transport; trades blur ($\varepsilon$) for iteration count |
:::

