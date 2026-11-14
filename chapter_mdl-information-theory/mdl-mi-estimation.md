# Estimating Mutual Information
:label:`sec_mdl-mi-estimation`

The previous section computed mutual information from known densities. In
practice we hold only samples: two crops of one photograph, an image and its
caption, a sentence and its continuation. We would still like to know how
much the two share, and representation learning goes further and trains
models to make that amount large. This section asks what samples alone can
tell us about dependence.

A familiar loss makes the question concrete. A contrastive model such as
SimCLR or CLIP learns to pick the true partner of an example out of a batch
of candidates, say $128$ of them, and reports a cross-entropy loss. Papers
often subtract that loss from $\ln 128 = 4.85$ and quote the difference as
the information that the two views share. That number can never exceed
$4.85$ nats, however much the views share, because a model that always
picks the right partner has shown only that it can tell $128$ candidates
apart. Since models are trained on such numbers and papers read them as
measurements, we need to tell a measurement from a bound, and the slack of a
bound from the error of a model.

There are two routes. Nonparametric estimators count nearest neighbors. They
are accurate for a pair of numbers, but for pairs of vectors in more than a
few dimensions they report far less than the pair shares. Variational
estimators replace the unknown densities by a model, a decoder or a *critic*
(a function that scores pairs), chosen so that the result is a lower bound
on the information, and then maximize the bound. Two results matter most:
the limit on what any estimator can guarantee without assumptions
(:numref:`subsec_mdl-mi-distribution-free`), and the InfoNCE bound with its
ceiling (:numref:`sec_mdl-infonce`). The nearest-neighbor estimator can be
skipped on a first reading.

The running example is the standard bivariate Gaussian pair of
:numref:`subsec_mdl-mi-gaussian`: $X$ and $Y$ are standard normal with
correlation $\rho$, and they share

$$I(X; Y) = -\tfrac{1}{2}\log\left(1 - \rho^2\right)$$

nats, $0.83$ at $\rho = 0.9$ and $1.96$ at $\rho = 0.99$. Every estimator
below is tested against this value. Logarithms are natural, so information is
measured in nats.

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-3}
from scipy.spatial.distance import cdist
from scipy.special import digamma, logsumexp
```

Every bound below also uses the *pointwise mutual information*
(:numref:`subsec_mdl-pmi`), the log ratio
$\textrm{pmi}(x, y) = \log \frac{p(x, y)}{p(x)\,p(y)}$, which compares how
likely a particular pair is under the joint distribution and under
independence. For the Gaussian pair it is a quadratic in $x$ and $y$:

$$\textrm{pmi}(x, y) = -\tfrac{1}{2}\log(1 - \rho^2) - \frac{x^2 - 2\rho x y + y^2}{2(1 - \rho^2)} + \frac{x^2 + y^2}{2}.$$

Its average over pairs drawn from the joint distribution is $I(X; Y)$. The
average of its exponential over *independent* draws is $1$, because
$\int\!\!\int p(x)\,p(y)\,\frac{p(x, y)}{p(x)\,p(y)}\,dx\,dy = \int\!\!\int p(x, y)\,dx\,dy = 1$.
The cell defines the pair and its pmi and checks both facts on samples. For
the second it pairs each $x$ with the $y$ of a different sample, which
destroys the dependence.

```{.python .input #mdl-mi-estimation-estimating-mutual-information-2}
def gauss_pair(rng, rho, n):
    """Samples of the standard Gaussian pair; n is a count or a shape."""
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x, y

def mi_gauss(rho):
    """I(X;Y) of the pair, in nats."""
    return -0.5 * np.log(1 - rho**2)

def pmi_gauss(x, y, rho):
    """Pointwise mutual information pmi(x, y) of the pair, in nats."""
    c = 1 - rho**2
    return (-0.5 * np.log(c) - (x**2 - 2 * rho * x * y + y**2) / (2 * c)
            + (x**2 + y**2) / 2)

rng = np.random.default_rng()
x, y = gauss_pair(rng, 0.9, 200_000)
print(f'I(X;Y) = {mi_gauss(0.9):.3f} nats, mean pmi over pairs = '
      f'{pmi_gauss(x, y, 0.9).mean():.3f}')
print(f'mean exp(pmi) over independent pairs = '
      f'{np.exp(pmi_gauss(x, np.roll(y, 1), 0.9)).mean():.3f}')
```

Both averages agree with their targets to about a hundredth. Every bound in
this section rests on these two identities, and the second will cause most
of the trouble.

## Nonparametric Estimation
:label:`sec_mdl-mi-hard`

The direct approach estimates the probability of small regions from the
fraction of samples that fall into them and plugs the fractions into the
definition. Fixed regions fail in high dimension, adaptive regions do better
in a few dimensions, and without assumptions no estimator can guarantee
much.

### Histograms in High Dimension
:label:`subsec_mdl-mi-high-dim`

The simplest estimator bins each coordinate into $b$ intervals, takes the
cell frequencies of the histogram as probabilities, and applies the discrete
formula. It errs in both directions. Bins that are too coarse hide
dependence: when the mass sits on a ridge thinner than the bins, as for the
pair at $\rho = 0.99$ in :numref:`subsec_mdl-mi-gaussian`, the estimate
falls short. Bins that are too fine invent it: for an independent pair the
estimate is about $(b - 1)^2/(2n)$ on average from $n$ samples, not zero
(:exref:`ex_mdl-mutual-information-histogram-bias`). And a monotone map of
either variable changes the estimate, although it leaves the information
unchanged.

<!-- WRITER: this cell and the next moved here from the Mutual Information
section (its histogram narration is cut there). They need a lead-in sentence
each and a reading after the output. Ranges below are over 15 development
runs (3 NumPy probes + 3 runs of each of the four tabs). Table: 200,000
samples, equal-width bins, b = 10 / 40 / 160: rho 0: 0.000 / 0.003 /
0.033-0.038; rho 0.9: 0.60-0.64 / 0.81-0.82 / 0.842-0.853 (truth 0.830);
rho 0.99: 1.00-1.14 / 1.75-1.81 / 1.945-1.957 (truth 1.959); parabola
Y = X^2 + 0.3 eps: 0.41-0.57 / 0.94-1.01 / 1.127-1.137 (truth 1.141). So
the "(b - 1)^2/(2n) on average" sentence above does not match the printed
b = 160 value: (b-1)^2/(2n) = 0.063, printed 0.033-0.038 (equal-width bins
leave the tail cells nearly empty; rank bins would give 0.065). Quote "a few
hundredths of a nat" and leave the prediction to the exercise. At b = 160
the upward bias and the lost ridge partly cancel at rho = 0.99: do not read
that row as "fine bins are right". Second cell, rho = 0.9, b = 40:
correlation 0.900 -> 0.694-0.700 under x -> x^3; raw estimate 0.81-0.82 ->
0.38-0.51; the rank estimate (0.813-0.819) is identical for x and x^3 in
every run; truth 0.830 printed. -->

```{.python .input #mutual-information-gaussian-binned}
def binned_mi(x, y, bins):
    """Plug-in MI of the bins x bins histogram of (x, y), in nats."""
    p = np.histogram2d(x, y, bins=bins)[0] / len(x)       # cell frequencies
    q = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
    nz = p > 0                                            # occupied cells
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())

mi_parabola = 1.141  # by quadrature, sec_mdl-mi-nonlinear-correlation
n = 200_000
x = rng.standard_normal(n)
pairs = {'rho = 0': (gauss_pair(rng, 0.0, n), 0.0),
         'rho = 0.9': (gauss_pair(rng, 0.9, n), mi_gauss(0.9)),
         'rho = 0.99': (gauss_pair(rng, 0.99, n), mi_gauss(0.99)),
         'parabola': ((x, x**2 + 0.3 * rng.standard_normal(n)), mi_parabola)}
print('pair        truth  b = 10  b = 40  b = 160')
for name, ((u, v), truth) in pairs.items():
    print(f'{name:10}  {truth:5.3f}'
          + ''.join(f'  {binned_mi(u, v, b):6.3f}' for b in (10, 40, 160)))
```

```{.python .input #mdl-mutual-information-mutual-information-as-nonlinear-correlation-2}
x, y = gauss_pair(rng, 0.9, n)
rank = lambda a: np.argsort(np.argsort(a))       # position in sorted order
print(f'true I = {mi_gauss(0.9):.3f} nats; histograms with b = 40:')
for name, u in (('x', x), ('x**3', x**3)):
    print(f'{name:4}: correlation {np.corrcoef(u, y)[0, 1]:.3f}, '
          f'raw {binned_mi(u, y, 40):.3f}, '
          f'ranks {binned_mi(rank(u), rank(y), 40):.3f} nats')
```

In more dimensions the histogram fails outright. With $b$ bins per axis, a
histogram over a $d$-dimensional space has $b^d$ cells. At $d = 10$ and
$b = 10$ that is $10^{10}$ cells, nearly all empty for any sample we can
afford, so the estimate is sampling noise in the few occupied cells. A
better estimator must let the data choose the resolution, with small cells
where samples are dense and large cells where they are sparse.

### The KSG Estimator
:label:`subsec_mdl-ksg`

The estimator of :citet:`Kraskov.Stogbauer.Grassberger.2004`, KSG for short,
replaces fixed cells by neighborhoods. Distances are measured by the largest
coordinate difference (the Chebyshev distance), so a ball is a square in the
plane and a box in higher dimension. Around sample $i$ draw the smallest box
in the joint $(x, y)$ space that reaches its $m$-th nearest neighbor, and
call its half-width $r_i$. The box holds the fraction $m/n$ of the $n$
samples, an estimate of the joint probability near the point. The strip
$|x - x_i| < r_i$ of the same half-width holds $n_x^{(i)}$ other samples and
the strip $|y - y_i| < r_i$ holds $n_y^{(i)}$; their fractions estimate the
two marginal probabilities. The logarithm of the joint fraction over the
product of the marginal fractions is a pointwise mutual information computed
by counting. :numref:`fig_mdl-ksg-box` shows the three counts at one point.

![The KSG count at one sample point $i$ (the star), with $m = 3$. The box of half-width $r_i$ reaches the third nearest neighbor in the joint plane (the ring on its left edge); two neighbors lie strictly inside. The vertical strip of the same half-width holds $n_x^{(i)}$ other samples and the horizontal strip holds $n_y^{(i)}$ (blue and green dots lie in one strip only). The summand of the estimator is approximately $\log\frac{m/n}{(n_x/n)(n_y/n)}$, a pointwise mutual information from counts.](../img/mdl-it-ksg-box.svg)
:label:`fig_mdl-ksg-box`

Averaging these local log ratios, with the digamma function
$\psi = (\log \Gamma)'$ in place of the logarithm to remove the bias of small
counts (the mass $p$ of the box that reaches the $m$-th of $n - 1$
neighbors has $\mathbb{E}[\log p] = \psi(m) - \psi(n)$, not $\log(m/n)$;
we do not derive the correction), gives the estimator

$$\hat{I}_{\textrm{KSG}} = \psi(m) + \psi(n) - \frac{1}{n} \sum_{i=1}^{n} \left(\psi\left(n_x^{(i)} + 1\right) + \psi\left(n_y^{(i)} + 1\right)\right).$$
:eqlabel:`eq_mdl-ksg`

Since $\psi(j) \approx \log j$ for large $j$, each summand is approximately
the local log ratio of the box fraction to the product of the strip
fractions. The radii $r_i$ adapt to the data, so there
is no bin width or bandwidth to choose. Both strip counts use the strict
inequality and exclude the point itself, so a neighbor on the boundary of a
strip is not counted.

The implementation counts the neighbors of all points at once from the
$n \times n$ matrices of Chebyshev distances in each space (`cdist`). The
joint distance is their elementwise maximum, because the largest coordinate
difference of $(x, y)$ is the larger of the two largest differences.

```{.python .input #mdl-mutual-information-the-curse-of-estimation}
def ksg_counts(x, y, m=5):
    """Strip counts n_x, n_y of every sample point; x, y are (n, d)."""
    dx, dy = cdist(x, x, 'chebyshev'), cdist(y, y, 'chebyshev')
    d = np.maximum(dx, dy)                       # joint Chebyshev distance
    np.fill_diagonal(d, np.inf)
    r = np.partition(d, m - 1, axis=1)[:, m - 1]  # half-width: m-th neighbor
    n_x = (dx < r[:, None]).sum(axis=1) - 1      # strict, self excluded
    n_y = (dy < r[:, None]).sum(axis=1) - 1
    return n_x, n_y

def ksg_mi(x, y, m=5):
    """KSG nearest-neighbor estimate of I(X;Y) in nats; x, y are (n, d)."""
    n_x, n_y = ksg_counts(x, y, m)
    return float(digamma(m) + digamma(len(x))
                 - (digamma(n_x + 1) + digamma(n_y + 1)).mean())
```

Each summand of :eqref:`eq_mdl-ksg` involves only counts, and counting alone
bounds how large the estimate can be.

**Proposition (sample-size ceiling of KSG).** *For any sample of $n$ points
in which no two joint distances from the same point are equal,*

$$\hat{I}_{\textrm{KSG}} \leq \psi(n) - \psi(m),$$
:eqlabel:`eq_mdl-ksg-ceiling`

*with equality when $Y = X$.*

**Proof.** Without ties, exactly $m - 1$ samples lie strictly closer to point
$i$ than its $m$-th neighbor. Each of them satisfies $|x_j - x_i| < r_i$ and
$|y_j - y_i| < r_i$, because the joint distance is the larger of the two
coordinate distances, so it lies in both strips: $n_x^{(i)} \geq m - 1$ and
$n_y^{(i)} \geq m - 1$. The digamma function is increasing, so each summand
is at most $\psi(m) + \psi(n) - 2\psi(m) = \psi(n) - \psi(m)$. For $Y = X$
the three distances coincide, both strips hold exactly the $m - 1$ points
inside the box, and every summand attains the bound. $\blacksquare$

At $n = 2{,}000$ and $m = 5$ the ceiling is $\psi(2000) - \psi(5) = 6.09$
nats, close to $\log(n/m) = 5.99$. A pair with infinite information, $Y = X$
for continuous $X$, is reported as $6.1$ nats, and doubling the sample raises
the ceiling by only about $\log 2$. Counting neighbors can report only as
much information as the sample has distinguishable neighborhoods, about
$n/m$ of them. This is the first ceiling of the section, and every later one
has the same shape: an estimator cannot report more than the logarithm of the
number of things it can tell apart.

### KSG in High Dimension
:label:`subsec_mdl-ksg-dimension`

In more dimensions the estimator fails well below its ceiling, and for a
different reason. Take $d$ independent copies of the pair, each with
$\rho = 0.9$, so that $X$ and $Y$ are $d$-dimensional. Information adds over
independent coordinates, because the log ratio is a sum of $d$ terms, so the
truth is $d \times 0.83$ nats. The estimator assumes that the density is
nearly constant over each box. A box that holds a fraction $m/n$ of a roughly
uniform sample in $2d$ dimensions has sides about $(m/n)^{1/(2d)}$ times the
width of the data: $0.05$ at $d = 1$, but $0.83$ at $d = 16$, where the
neighborhoods are no longer local. The estimate should therefore fall behind
the truth as $d$ grows. By how much? The cell also runs $Y = X$, which attains
the ceiling.

```{.python .input #mdl-mi-estimation-ksg-in-dimension}
n, rho = 2000, 0.9
print(' d  true I    KSG  mean strip count')
for dim in (1, 2, 4, 8, 16):
    x, y = gauss_pair(rng, rho, (n, dim))        # dim independent pairs
    n_x, n_y = ksg_counts(x, y)
    print(f'{dim:2d}  {dim * mi_gauss(rho):6.2f}  {ksg_mi(x, y):5.2f}  '
          f'{(n_x.mean() + n_y.mean()) / 2:16.1f}')
x = rng.standard_normal((n, 1))
print(f'Y = X:  KSG = {ksg_mi(x, x):.2f}, '
      f'ceiling psi(n) - psi(m) = {digamma(n) - digamma(5):.2f} nats')
```

In one dimension KSG lands within about $0.1$ nat of the truth. At $d = 4$
it reports about $2.5$ for $3.3$, almost a nat low. At $d = 16$ it reports
$2.8$ for $13.3$, less than half of its $6.1$-nat ceiling. The estimator has
saturated: the truth quadrupled between the last two rows and the output
barely moved.

<!-- WRITER: the cell now has rows d = 1, 2, 4, 8, 16 and prints the mean
strip count (average of n_x and n_y over the sample). Ranges over 15
development runs (3 NumPy probes + 3 runs of each tab): KSG 0.83-0.88 /
1.59-1.70 / 2.46-2.60 / 2.79-2.93 / 2.76-2.81 for truth 0.83 / 1.66 / 3.32 /
6.64 / 13.29; mean strip count 69.5-71.6 / 48.1-50.9 / 29.9-32.1 /
24.6-26.4 / 25.8-26.7. So d = 2 is still accurate, the saturation is
complete by d = 8, and "(the cell does not print them; ... computes them)"
below is now false. The counting is split into `ksg_counts` (strip counts)
and `ksg_mi` (estimate from them); the description before the KSG cell still
holds. -->

The strip counts show the saturation directly. They hold about $70$ of the
$2{,}000$ points at $d = 1$, about $31$ at $d = 4$ and about $26$ at
$d = 16$ (the cell does not print them;
:exref:`ex_mdl-mi-estimation-ksg-ceiling` computes them). The smoothness
that makes neighbor counts consistent needs a sample that grows
exponentially with the dimension.

### Limits of Distribution-Free Estimation
:label:`subsec_mdl-mi-distribution-free`

Perhaps a cleverer estimator does better. Ask only for a lower bound: a
function $B$ of $n$ samples such that $B \leq I(X; Y)$ with probability at
least $1 - \delta$ *for every distribution* of the pair, with failure
probability $\delta < \tfrac{1}{2}$. Lower bounds are the useful direction, because a
representation learner wants to certify that two views share a lot.

The smallest case shows the obstacle. Let $Y = X$ be a copy of a discrete
variable, so that $I(X; Y) = H(X)$. Suppose that $B$ is a function of which
values repeat in the sample and how often (its *type*), as it must be for
any estimator that treats the symbols as labels without meaning. Four steps
give the limit, and the third is the one to watch.

1. Let $X$ be uniform on $M$ symbols and draw $n$ samples. Two given samples
   coincide with probability $1/M$, so by the union bound some two of the
   $n(n-1)/2$ pairs coincide with probability at most $n(n-1)/(2M)$.
1. Apply the guarantee to the uniform distribution on $M = n^2$ symbols,
   whose entropy is $2 \log n$: on a sample from *that* distribution,
   $B \leq 2 \log n$ with probability at least $1 - \delta > \tfrac{1}{2}$.
1. By the first step, a sample from that distribution consists of $n$
   distinct values with probability more than $\tfrac{1}{2}$. Two events of
   probability more than $\tfrac{1}{2}$ overlap, so $B \leq 2 \log n$ on
   some all-distinct sample.
1. All samples of $n$ distinct values have the same type, so $B$ reports at
   most $2 \log n$ on every one of them, whatever source produced it.

Now take $M = 2^{256}$ symbols, entropy $256 \ln 2 = 177$ nats. At
$n = 10^6$ the sample is all distinct with probability at least
$1 - 4.3 \times 10^{-66}$, and on that sample $B$ reports at most
$2 \log 10^6 = 27.6$ nats. A million samples cannot certify $177$ nats, and
they cannot certify $30$. :exref:`ex_mdl-mutual-information-entropy-bounds-samples`
sharpens the constant and asks what sample would suffice.

:citet:`McAllester.Stratos.2020` proved the general statement, without the
assumption on types:

> **Theorem (McAllester–Stratos, informal).** Any distribution-free lower
> bound on $I(X; Y)$ that holds with fixed confidence from $n$ samples is
> $O(\log n)$.

The reparameterization invariance of the previous section is part of the
reason: mutual information does not change under an invertible relabelling of
either variable, so an estimator that assumes nothing about the coordinates
must cope with every relabelling of the sample at once, and all it can rely
on is which values repeat. Any estimator that reports more than the
logarithm of its sample size does so by assuming something: that distances
in the given coordinates mean something and the density is smooth in them,
as KSG does, or a model family, as the bounds that follow do.

## Variational Lower Bounds
:label:`sec_mdl-mi-variational-bounds`

Mutual information is an expectation of $\log\frac{p(x, y)}{p(x)\,p(y)}$, and
the densities in it are unknown. Variational methods replace them by an
auxiliary model, chosen so that the result is a *lower bound* on the
information for every model, and optimizing the model tightens the bound. We
say that a model *certifies* the value of its bound, and mean the population
inequality; a batch average only estimates the certified value. Contrastive
methods use the resulting bound as their training objective, and
:citet:`Poole.Oord.Alemi.ea.2019` give a unified treatment. Every proof below
rests on Gibbs' inequality (:numref:`sec_mdl-information_theory`), the
nonnegativity of the KL divergence. One estimation problem thus becomes an
optimization problem, and the result carries the model's approximation error
and the finite-sample error.

### The Barber–Agakov Bound

The first bound replaces the unknown posterior $p(x \mid y)$ by a model of it.

**Proposition (Barber–Agakov bound).** *For any conditional density
$q(x \mid y)$,*

$$I(X; Y) \geq H(X) + \mathbb{E}_{(x,y) \sim P_{X,Y}}\left[\log q(x \mid y)\right],$$
:eqlabel:`eq_mdl-ba_bound`

*with equality if and only if $q(x \mid y) = p(x \mid y)$. For continuous
$X$ the differential entropy $h(X)$ takes the place of $H(X)$.*

**Proof.** Write $I(X; Y) = H(X) - H(X \mid Y)$ and insert the model:

$$
I(X; Y) - H(X) - \mathbb{E}\left[\log q(x \mid y)\right]
= \mathbb{E}\left[\log \frac{p(x \mid y)}{q(x \mid y)}\right]
= \mathbb{E}_{y}\left[D_{\textrm{KL}}\big(p(\cdot \mid y) \,\|\, q(\cdot \mid y)\big)\right]
\geq 0
$$

by Gibbs' inequality, applied for each value of $y$ and averaged. Equality
holds if and only if every divergence vanishes, that is, when $q$ is the
posterior. $\blacksquare$

The bound :cite:`Barber.Agakov.2003` is the information-theoretic reading of
an encoder–decoder: if a decoder reconstructs $X$ from $Y$ with low
cross-entropy, the pair must share a lot of information. Its weak point is
$H(X)$, which is unknown in the situations where we need the bound. It is a
constant, however, so *maximizing* the bound over $q$ never needs it. Substituting
$H(X) = -\mathbb{E}\log p(x)$ gives the equivalent ratio form
$I(X; Y) \geq \mathbb{E}\left[\log \left(q(x \mid y)/p(x)\right)\right]$,
which InfoNCE will use.

Two instances show what the bound certifies. First, exchange the roles of
the variables, which is allowed because $I$ is symmetric, and let
$q(y \mid x)$ be a classifier. The bound is then the loss certificate
:eqref:`eq_mdl-info-certificate`: for $k$ balanced classes and held-out
cross-entropy $L$, $I(X; Y) \geq \log k - L$. Since $L \geq 0$, no 10-class
classifier certifies more than $\ln 10 = 2.30$ nats. InfoNCE will have this
shape.
Second, for the Gaussian pair, a decoder trained by squared error alone is a
Gaussian $q(x \mid y) = \mathcal{N}(x; a y, 1)$ with unit variance. With
$h(X) = \tfrac{1}{2}\log(2\pi e)$ the bound becomes
$\tfrac{1}{2} - \tfrac{1}{2}\mathbb{E}(X - aY)^2 = \tfrac{1}{2} - \tfrac{1}{2}(1 - 2a\rho + a^2)$,
which is largest at $a = \rho$, where it equals $\rho^2/2$. At $\rho = 0.9$
this decoder certifies $0.41$ nats of the true $0.83$, and it never
certifies more than $\tfrac{1}{2}$ however strong the dependence. The gap is
the bias of a decoder whose variance is fixed at $1$ instead of the true
conditional variance $1 - \rho^2$
(:exref:`ex_mdl-mutual-information-ba-gaussian-decoder`).

### The Donsker–Varadhan Bound

The second bound needs neither a normalized model nor $H(X)$. It writes the
KL divergence, and hence mutual information, as a supremum over arbitrary
critic functions $T$, with no requirement that anything integrate to one.
This is the convex-duality pattern behind the f-GAN construction of
:numref:`sec_mdl-f-gan-dual`.

**Proposition (Donsker–Varadhan bound).** *For distributions $P$ and $Q$ and
any function $T$ for which $\mathbb{E}_P[T]$ is defined and
$\mathbb{E}_Q[e^T] < \infty$,*

$$D_{\textrm{KL}}(P \| Q) \geq \mathbb{E}_P[T] - \log \mathbb{E}_Q\left[e^{T}\right],$$
:eqlabel:`eq_mdl-dv_bound`

*with equality if and only if $T = \log \frac{dP}{dQ} + c$ for a constant
$c$. When $P \ll Q$ and the divergence is finite, the supremum over $T$
therefore attains $D_{\textrm{KL}}(P \| Q)$, which is the Donsker–Varadhan
representation of the divergence* :cite:`Donsker.Varadhan.1983`.

**Proof.** If $P$ is not absolutely continuous with respect to $Q$, the left
side is infinite. Otherwise, given $T$, define the *Gibbs distribution*
$\tilde{Q}$ with density $\tilde{q}(x) = q(x)\, e^{T(x)} / Z$ and normalizer
$Z = \mathbb{E}_Q[e^T]$: the distribution $Q$ tilted by the critic. Then

$$
D_{\textrm{KL}}(P \| Q) - \left(\mathbb{E}_P[T] - \log Z\right)
= \mathbb{E}_P\left[\log \frac{p}{q} - T + \log Z\right]
= \mathbb{E}_P\left[\log \frac{p}{\tilde{q}}\right]
= D_{\textrm{KL}}(P \| \tilde{Q}) \geq 0,
$$

by Gibbs' inequality. Equality holds if and only if $P = \tilde{Q}$, which
means $p/q = e^{T}/Z$, or $T = \log(p/q) + \log Z$. $\blacksquare$

The gap of the bound is the divergence from $P$ to the tilted distribution,
so a critic is good when tilting $Q$ by $e^T$ reproduces $P$. Specializing
to $P = P_{X,Y}$ and $Q = P_X \otimes P_Y$ bounds mutual information: for
any critic $T(x, y)$,

$$
I(X; Y) \geq \mathbb{E}_{P_{X,Y}}[T] - \log \mathbb{E}_{P_X \otimes P_Y}\left[e^{T}\right].
$$

The first expectation is an average over the sample pairs. The second needs
pairs with no dependence, which a single `roll` of the batch supplies, as in
the running-example cell. *MINE* (mutual information neural estimation,
:citet:`Belghazi.Baratin.Rajeshwar.ea.2018`) is this bound with a neural
network as the critic; its gradient uses a moving average of
$\mathbb{E}_Q[e^T]$ across minibatches, because the minibatch gradient of the
logarithm of an average is biased.

### The NWJ Bound

The logarithm in front of $\mathbb{E}_Q[e^T]$ couples all the samples of a
batch, and it biases the minibatch estimate. By Jensen's inequality the
logarithm of a batch average is on average below the logarithm of the
population average, so the minibatch DV value
$\overline{T} - \log \overline{e^T}$ overestimates the bound in expectation.

The tangent-line inequality $\log z \leq z/e$, with equality at $z = e$,
removes the logarithm: applied to $Z = \mathbb{E}_Q[e^T]$ in
:eqref:`eq_mdl-dv_bound`, it gives, for every critic $T$,

$$I(X; Y) \geq \mathbb{E}_{P_{X,Y}}[T] - e^{-1}\, \mathbb{E}_{P_X \otimes P_Y}\left[e^{T}\right].$$
:eqlabel:`eq_mdl-nwj_bound`

This is the *NWJ bound* :cite:`Nguyen.Wainwright.Jordan.2010`, and it is
tight at $T^* = 1 + \log\frac{dP}{dQ}$, the critic for which $Z = e$. For
any other critic it is smaller than the DV bound, which is what the tangent
inequality says. Since $e^{-1}\, \mathbb{E}_Q[e^T] = \mathbb{E}_Q[e^{T-1}]$,
it is the f-GAN bound on the KL divergence,
$D_{\textrm{KL}}(P \| Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T-1}]$, that
:numref:`sec_mdl-f-gan-dual` derived from the convex conjugate of
$u \log u$.

No logarithm of an average appears, so the minibatch value of
:eqref:`eq_mdl-nwj_bound` is an unbiased estimate of the bound. Unbiasedness
has a price, and it is paid in the term $e^T$. For the optimal critic,
$e^{T^*-1}$ is the density ratio, which is enormous on rare pairs and tiny on
typical ones, so its average over a batch is dominated by pairs that most
batches do not contain. :numref:`subsec_mdl-exact-critics` works out what
this does to a batch and measures it.

## InfoNCE
:label:`sec_mdl-infonce`

### The Contrastive Loss

The $e^T$ of the DV and NWJ bounds is unnormalized, so a single pair can
dominate a batch. InfoNCE normalizes it within the batch: it makes the
candidates compete. Draw one *positive* pair $(x, y^+) \sim P_{X,Y}$ and
$N - 1$ *negatives* $y_2, \ldots, y_N$ drawn independently from $P_Y$. Score
every candidate with the critic $T(x, y)$, and ask a softmax over the $N$
scores to point at the positive. The *InfoNCE loss*
:cite:`Oord.Li.Vinyals.2018` is the resulting cross-entropy,

$$\mathcal{L}_{\textrm{NCE}} = -\mathbb{E}\left[\log \frac{e^{T(x, y^+)}}{e^{T(x, y^+)} + \sum_{j=2}^{N} e^{T(x, y_j)}}\right],$$
:eqlabel:`eq_mdl-infonce_def`

where the expectation is over the positive pair and the negatives. The
associated estimate of mutual information is the logarithm of the number of
candidates minus the loss,

$$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}.$$
:eqlabel:`eq_mdl-infonce_est`

Many papers, following :citet:`Oord.Li.Vinyals.2018`, write the loss with a
$\frac{1}{N}$ inside the logarithm and a compensating $\log N$ outside; the
two conventions give the same number. The loss is an $N$-class cross-entropy
whose classes are the other examples in the batch rather than fixed labels.
It is the loss of CPC, SimCLR and CLIP
:cite:`Oord.Li.Vinyals.2018,Chen.Kornblith.Norouzi.ea.2020,radford2021learning`,
where $x$ and $y$ are two views or two modalities of one example and the
batch supplies the negatives.

![The InfoNCE game. Left: an anchor $x$ is scored by the critic against the positive $y^+$, drawn jointly with it, and $N-1$ negatives drawn independently from the marginal; the rays carry the scores $T(x, y_j)$. Right: a softmax over the $N$ scores must point at the positive (nearer candidates score higher). The resulting cross-entropy certifies $I(X;Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}$, a bound that can never exceed $\log N$.](../img/mdl-it-infonce-pos-neg.svg)
:label:`fig_mdl-infonce-pos-neg`

:numref:`fig_mdl-infonce-pos-neg` shows the game. Its smallest case is
$N = 2$: one negative, and the loss is the binary cross-entropy of telling
the true partner from an impostor. Even a perfect answer shows only that one
of two candidates was right, and the estimate is at most $\ln 2 = 0.69$ nats.

### The InfoNCE Bound

The proposition says that a small loss certifies information, and that the
certificate is capped :cite:`Poole.Oord.Alemi.ea.2019`.

**Proposition (InfoNCE bound).** *For any measurable critic $T$ and any
$N \geq 1$,*

$$I(X; Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}.$$
:eqlabel:`eq_mdl-infonce_bound`

*The right-hand side is at most $\log N$.*

**Proof.** Number the candidates $y_1 = y^+, y_2, \ldots, y_N$ and write
$V = (Y_1, \ldots, Y_N)$ for the whole bag. The negatives $Y_{2:N}$ are
independent of the pair $(X, Y_1)$. The proof replaces $Y$ by the bag, finds
a normalized model of the bag given $X$, and applies Barber–Agakov.

*Step 1: independent negatives add no information.* The chain rule of the
previous section splits the information in the bag,
$I(X; Y_1, Y_{2:N}) = I(X; Y_1) + I(X; Y_{2:N} \mid Y_1)$. Given $Y_1$, the
negatives are still independent of $X$, because they are independent of
$(X, Y_1)$; the second term is therefore $0$, and $I(X; V) = I(X; Y)$.

*Step 2: the softmax ratio is a normalized model.* This is the step that is
not routine. Define

$$
\gamma(x, v) = \frac{e^{T(x, y_1)}}{\frac{1}{N}\sum_{j=1}^N e^{T(x, y_j)}},
$$

which lies between $0$ and $N$. Let $p(v)$ be the distribution of a bag of
$N$ independent draws from $P_Y$, the positive slot included. The $N$
candidates of such a bag are exchangeable, so the $N$ ratios
$e^{T(x, y_i)} / \frac{1}{N}\sum_j e^{T(x, y_j)}$ have the same expectation,
and for every fixed $x$

$$
\mathbb{E}_{p(v)}\left[\gamma(x, V)\right]
= \frac{1}{N}\sum_{i=1}^{N} \mathbb{E}_{p(v)}\left[\frac{e^{T(x, Y_i)}}{\frac{1}{N}\sum_{j} e^{T(x, Y_j)}}\right]
= \mathbb{E}_{p(v)}\left[\frac{\frac{1}{N}\sum_{i} e^{T(x, Y_i)}}{\frac{1}{N}\sum_{j} e^{T(x, Y_j)}}\right] = 1.
$$

Hence $q(v \mid x) = p(v)\, \gamma(x, v)$ is a conditional distribution over
bags.

*Step 3: Barber–Agakov on the bag.* The marginal distribution of the bag
$V$ is $p(v)$, because the positive is marginally a draw from $P_Y$ and is
independent of the negatives. Apply the ratio form of
:eqref:`eq_mdl-ba_bound` to the pair $(X, V)$ with the model $q(v \mid x)$,
the two variables' roles exchanged, which the symmetry of $I$ allows:

$$
I(X; V) \geq \mathbb{E}_{p(x, v)}\left[\log \frac{q(v \mid x)}{p(v)}\right]
= \mathbb{E}\left[\log \gamma(X, V)\right]
= \log N - \mathcal{L}_{\textrm{NCE}}.
$$

The last equality holds because $\log \gamma$ is $\log N$ plus the log-softmax
probability of the positive. With Step 1 this proves the bound. The softmax
probability of the positive is at most $1$, so $\mathcal{L}_{\textrm{NCE}} \geq 0$
and the right-hand side is at most $\log N$. $\blacksquare$

Naming the right partner out of $N$ candidates, even with certainty,
demonstrates $\log N$ nats, the information in one choice among $N$; no
critic can make the game show more. The loss is smallest for the critics
$T^*(x, y) = \textrm{pmi}(x, y) + c(x)$, with any function $c$ of the anchor
alone (:exref:`ex_mdl-mutual-information-optimal-critic`). A larger $N$
raises the ceiling, which is the reason usually given for the large batches
of CPC, SimCLR and CLIP.

### The Ceiling with an Exact Critic
:label:`subsec_mdl-infonce-ceiling`

One might guess that with the exact critic $T = \textrm{pmi}$ the estimate
follows the truth up to $\log N$ and stops there. To test the guess, the
cell scores all $N \times N$ pairs of a batch at once: row $i$ holds anchor
$x_i$ against every $y_j$, its diagonal entry is the positive, and the other
$N - 1$ entries are its negatives. It sweeps the true information for
$N = 16$, $128$ and $1024$, with dotted lines at the three ceilings $\ln N$.

```{.python .input #mutual-information-log-n-ceiling}
def infonce_bound(f):
    """log N minus the InfoNCE loss of an N x N score matrix f[i, j]."""
    return np.log(len(f)) + np.mean(np.diag(f) - logsumexp(f, axis=1))

def exact_scores(x, y, rho):
    """The exact critic T = pmi on all N x N pairs (x_i, y_j) of a batch."""
    return pmi_gauss(x[:, None], y[None, :], rho)

true_mi = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7])
rhos = np.sqrt(1 - np.exp(-2 * true_mi))                 # invert mi_gauss
Ns = (16, 128, 1024)
ests = np.array([[np.mean([infonce_bound(exact_scores(*gauss_pair(rng, r, N), r))
                            for _ in range(200 if N <= 128 else 50)])
                  for r in rhos] for N in Ns])              # (N, level)
ceilings = [[np.log(N)] * len(true_mi) for N in Ns]
d2l.plot(true_mi, [*ests, true_mi, *ceilings], 'true I(X;Y) (nats)',
         'InfoNCE estimate (nats)',
         legend=[f'N = {N}' for N in Ns] + ['truth'] + [f'ln {N}' for N in Ns],
         fmts=('C0-', 'C1-', 'C2-', 'k--', 'C0:', 'C1:', 'C2:'),
         figsize=(5.5, 3), legend_loc='outside')
for I in (1, 4, 7):
    k = list(true_mi).index(I)
    print(f'I = {I}:  ' + '   '.join(f'N = {N}: {e:.2f}'
                                     for N, e in zip(Ns, ests[:, k])))
```

<!-- WRITER: the cell now prints the estimates at I = 1, 4, 7 for the three
N under its plot; the dotted ceilings take their curve's colour and the
legend sits beside the axes. Ranges over 15 development runs, N = 16 / 128 /
1024: I = 1: 0.88-0.91 / 0.98-0.99 / 0.99-1.00; I = 4: 2.52-2.57 /
3.67-3.69 / 3.94-3.96; I = 7: 2.76-2.77 / 4.75-4.76 / 6.31-6.32. The 3.7 and
6.3 below are the printed values. -->

The guess is wrong. Each curve leaves the diagonal early and smoothly, not
at its ceiling. At $I = 4$ the $N = 128$ curve reads about $3.7$, although
its ceiling is $4.85$, and at $I = 7$ the $N = 1024$ curve reads about $6.3$
against a ceiling of $6.93$. Inserting $T = \textrm{pmi}$ into the loss
shows where the gap comes from:

$$\hat{I}_{\textrm{NCE}} = I(X; Y) - \mathbb{E}\left[\log \frac{1}{N}\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}\right].$$

The correction is the bound's own slack, and it is nonnegative by the
proposition. Its average contains the positive's own term
$e^{\textrm{pmi}(x, y^+)}/N$, so the correction is at least
$\mathbb{E}[\textrm{pmi}(x, y^+)] - \log N = I - \log N$, which is the cap
again. The other $N - 1$ terms average to about $1$ only when $N$ is large
enough to contain the rare negatives that carry
$\mathbb{E}[e^{\textrm{pmi}}] = 1$, so the slack vanishes only as
$N \to \infty$ (:exref:`ex_mdl-mutual-information-optimal-critic`). The cap
is the crudest consequence of this slack.

Fano's inequality (:numref:`subsec_mdl-fano`) gives the cap a scale:
$5\%$ error on $1{,}000$ balanced classes needs $6.36$ nats about the
label. A loss with $N = 128$ candidates cannot show that two views share
even that much, since it reports at most $4.85$, and by the curves above
$N = 1024$ comes close only when the views share about $7$. How good such
representations are for classification cannot be read off this number.

### Exact Critics Compared
:label:`subsec_mdl-exact-critics`

The three bounds can now be compared on equal terms: $N = 128$, and the
exact critic $T^* = 1 + \textrm{pmi}$ for each (adding a constant does not
change InfoNCE). The rare pairs of the NWJ subsection let us predict the
outcome. While $I$ is small, independent pairs that look like partners are
common and the three bounds should agree. At $I = 6$ a typical batch's
average of $e^{\textrm{pmi}}$ over its $128$ mismatched pairs is near zero.
On such a batch NWJ reports $\mathbb{E}_P[T^*] - 0 = 1 + I$, one nat high,
and DV reports the average pmi minus the logarithm of a near-zero average,
many nats high. The batches that do hold a mismatched pair with
$e^{\textrm{pmi}} \geq N$ lose a nat or more each, and they must hold
NWJ's mean at the truth. InfoNCE normalizes within the batch, so a negative
that looks like a partner costs at most one wrong softmax, never a factor
$e^{\textrm{pmi}}$: it should be steady, and low.

The cell draws $200$ batches at each level and prints the median and the
quartiles of the batch values, which describe heavy tails better than a
standard deviation does, and the mean for NWJ. The plot shows the medians
as solid lines and the quartiles as dotted lines of the same colour.

<!-- WRITER: the plot now shows only the three medians, the truth (dashed)
and the ceiling ln N (dotted), legend beside the axes; the quartile lines
are gone ("quartiles as dotted lines of the same colour" above is false).
The table gained two columns per level: "<0.1" = share of the 200 batches
whose mean e^pmi over the 128 mismatched (rolled) pairs is below 0.1,
">=N" = share holding a mismatched pair with e^pmi >= N. Ranges over 15
development runs, I = 4 / 5 / 6: "<0.1" 4-11% / 26-36% / 56-66%; ">=N"
5-8% / 27-40% / 22-30%; for I <= 3 both are 0-3%. So "What the table cannot
show ..." below is now false: the table shows it. Other columns at I = 6
(12 notebook runs): NWJ median 6.89-6.94, NWJ mean 5.65-6.05, InfoNCE
4.60-4.62, DV median 11-17. -->

```{.python .input #mutual-information-perfect-critic-bounds}
def quartiles(a):
    """Median and quartiles of a list of batch values, as text."""
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return f'{med:5.2f} [{q1:5.2f}, {q3:5.2f}]'

N, true_mi = 128, np.array([0.5, 1, 2, 3, 4, 5, 6])
medians = []
# Last two columns: share of batches whose mean e^pmi over mismatched pairs
# is below 0.1, and share holding a mismatched pair with e^pmi >= N.
print(f'{"I":>3}  {"DV median [IQR]":20}  {"NWJ median [IQR]":20}  '
      f'{"NWJ mean":>8}  {"InfoNCE median [IQR]":20}  {"<0.1":>4}  {">=N":>4}')
for I in true_mi:
    rho, batches = np.sqrt(1 - np.exp(-2 * I)), []
    for _ in range(200):
        x, y = gauss_pair(rng, rho, N)
        T_joint = 1 + pmi_gauss(x, y, rho)               # T* on true pairs
        T_prod = 1 + pmi_gauss(x, np.roll(y, 1), rho)    # T* on mismatched
        e_pmi = np.exp(T_prod - 1)                       # e^pmi, mismatched
        batches.append((T_joint.mean() - (logsumexp(T_prod) - np.log(N)),  # DV
                        T_joint.mean() - e_pmi.mean(),                     # NWJ
                        infonce_bound(exact_scores(x, y, rho)),
                        e_pmi.mean() < 0.1, e_pmi.max() >= N))
    dv, nwj, nce, low, rare = np.array(batches).T
    medians.append([np.median(a) for a in (dv, nwj, nce)])
    print(f'{I:3.1f}  {quartiles(dv)}  {quartiles(nwj)}  {nwj.mean():8.2f}'
          f'  {quartiles(nce)}  {low.mean():4.0%}  {rare.mean():4.0%}')
d2l.plot(true_mi, [*np.array(medians).T, true_mi, [np.log(N)] * len(true_mi)],
         'true I(X;Y) (nats)', 'median batch value (nats)',
         legend=['DV', 'NWJ', 'InfoNCE', 'truth', 'ln N'], ylim=[0, 8],
         fmts=('C0-', 'C1-', 'C2-', 'k--', 'k:'), figsize=(5.5, 3),
         legend_loc='outside')
```

Up to $I = 3$ the three medians are within about $0.15$ nat of the truth,
and InfoNCE has the narrowest spread. At $I = 6$ the table is the one
predicted. NWJ's median is about $7 = I + 1$ and its upper quartile sits on
top of the median, so half of the batches overshoot by about a nat, while
its mean is usually within half a nat of $6$. DV's median lies several nats
above the truth, with an upper quartile in the tens, so its curve leaves
the plot. InfoNCE reads $4.6$ with an interquartile range below $0.1$.

What the table cannot show is where NWJ's mean comes from. At $I = 6$ more
than half of the batches have an average of $e^{\textrm{pmi}}$ below $0.1$,
and about one in four holds a pair with $e^{\textrm{pmi}} \geq N$. The
mismatched-pair term has variance
$(e^{2I} - 1)/N$, a standard deviation of $36$ nats at $I = 6$, carried by
batches so rare that $200$ draws seldom include one
(:exref:`ex_mdl-mi-estimation-heavy-tails` derives the variance and
simulates the fractions).

None of this contradicts the bounds, which hold for population
expectations. But the minibatch DV value is not a lower bound even in
expectation, and the minibatch NWJ value is unbiased only through batches we
rarely see. This is how the variational bounds live with the limit of
:numref:`subsec_mdl-mi-distribution-free`: InfoNCE reports at most $\log N$
with $N$ at most the sample size, so it respects the limit by construction,
while DV and NWJ can report more, and their batch values are then no longer
guarantees.

### Training a Critic
:label:`subsec_mdl-infonce-learned-critic`

In practice the critic is learned. When a contrastive loss reports less than
the truth, is the critic at fault, or the bound? We train a critic of the
dual-encoder form that CLIP and SimCLR use: two small networks
$g, h : \mathbb{R} \to \mathbb{R}^{16}$ embed $x$ and $y$, and the critic is
the inner product $T(x, y) = g(x)^\top h(y)$. The $N \times N$ score matrix
of a batch is then one matrix product of the two embedding matrices, so $N$
anchors against $N$ candidates cost $2N$ network evaluations instead of
$N^2$. The data are two disjoint pools of $4{,}096$ pairs at $\rho = 0.99$,
one for training and one for evaluation; the true information is $1.96$
nats.

```{.python .input #mdl-mi-estimation-training-a-critic-1}
rho, n_pool = 0.99, 4096
x_tr, y_tr = (a.astype(np.float32) for a in gauss_pair(rng, rho, n_pool))
x_ev, y_ev = (a.astype(np.float32) for a in gauss_pair(rng, rho, n_pool))
```

Each encoder is a multilayer perceptron with two hidden layers of $64$ ReLU
units and Glorot-uniform initialization. The loss is the InfoNCE loss of the
score matrix, a cross-entropy whose label for row $i$ is $i$, the index of
the positive; every framework has it as its standard classification loss.

```{.python .input #mdl-mi-estimation-training-a-critic-2}
#@tab mxnet
def encoder(d_out=16, hidden=64):
    """An MLP that embeds a scalar in d_out dimensions."""
    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.Dense(hidden, activation='relu'),
            gluon.nn.Dense(hidden, activation='relu'), gluon.nn.Dense(d_out))
    net.initialize(init.Xavier())
    net.hybridize()
    return net

def scores(g, h, x, y):
    """All N x N critic values T(x_i, y_j) = g(x_i) . h(y_j)."""
    return mnp.dot(g(x.reshape(-1, 1)), h(y.reshape(-1, 1)).T)

cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def infonce_loss(f):
    """Cross-entropy of each row's softmax at its positive, the diagonal."""
    return cross_entropy(f, mnp.arange(len(f))).mean()

g, h = encoder(), encoder()
```

```{.python .input #mdl-mi-estimation-training-a-critic-2}
#@tab pytorch
def encoder(d_out=16, hidden=64):
    """An MLP that embeds a scalar in d_out dimensions."""
    net = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, d_out))
    for layer in net[::2]:                          # the Linear layers
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    return net

def scores(g, h, x, y):
    """All N x N critic values T(x_i, y_j) = g(x_i) . h(y_j)."""
    return g(x[:, None]) @ h(y[:, None]).T

def infonce_loss(f):
    """Cross-entropy of each row's softmax at its positive, the diagonal."""
    return torch.nn.functional.cross_entropy(f, torch.arange(len(f)))

g, h = encoder(), encoder()
```

```{.python .input #mdl-mi-estimation-training-a-critic-2}
#@tab tensorflow
def encoder(d_out=16, hidden=64):
    """An MLP that embeds a scalar in d_out dimensions."""
    return tf.keras.Sequential([
        tf.keras.Input((1,)), tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dense(d_out)])

def scores(g, h, x, y):
    """All N x N critic values T(x_i, y_j) = g(x_i) . h(y_j)."""
    return tf.matmul(g(x[:, None]), h(y[:, None]), transpose_b=True)

def infonce_loss(f):
    """Cross-entropy of each row's softmax at its positive, the diagonal."""
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        tf.range(tf.shape(f)[0]), f, from_logits=True))

g, h = encoder(), encoder()
```

```{.python .input #mdl-mi-estimation-training-a-critic-2}
#@tab jax
def init_encoder(key, sizes=(1, 64, 64, 16)):
    """Glorot-uniform weights and zero biases of an MLP with these widths."""
    glorot = jax.nn.initializers.glorot_uniform()
    keys = jax.random.split(key, len(sizes) - 1)
    return [(glorot(k, (a, b)), jnp.zeros(b))
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]

def encode(params, x):
    """Embed each scalar of x with the MLP."""
    z = x[:, None]
    for W, b in params[:-1]:
        z = jax.nn.relu(z @ W + b)
    W, b = params[-1]
    return z @ W + b

def scores(params, x, y):
    """All N x N critic values T(x_i, y_j) = g(x_i) . h(y_j)."""
    params_g, params_h = params
    return encode(params_g, x) @ encode(params_h, y).T

def infonce_loss(params, x, y):
    """Cross-entropy of each row's softmax at its positive, the diagonal."""
    f = scores(params, x, y)
    return optax.softmax_cross_entropy_with_integer_labels(
        f, jnp.arange(len(f))).mean()

key_g, key_h = jax.random.split(jax.random.key(int(rng.integers(2**31))))
params = (init_encoder(key_g), init_encoder(key_h))
```

Training runs $1{,}000$ Adam steps with learning rate $10^{-3}$ on batches of
$N = 128$ pairs drawn from the training pool. Afterwards, `critic_np`
evaluates the trained critic on NumPy arrays, so that one evaluation cell
serves every framework.

```{.python .input #mutual-information-infonce-train}
#@tab mxnet
trainer = gluon.Trainer(
    list(g.collect_params().values()) + list(h.collect_params().values()),
    'adam', {'learning_rate': 1e-3})
N = 128
for step in range(1000):
    idx = rng.choice(n_pool, N, replace=False)
    with autograd.record():
        loss = infonce_loss(scores(g, h, mnp.array(x_tr[idx]),
                                   mnp.array(y_tr[idx])))
    loss.backward()
    trainer.step(1)

def critic_np(a, b):
    """The trained critic on NumPy inputs, as an N x N NumPy array."""
    return scores(g, h, mnp.array(a), mnp.array(b)).asnumpy()
```

```{.python .input #mutual-information-infonce-train}
#@tab pytorch
opt = torch.optim.Adam([*g.parameters(), *h.parameters()], lr=1e-3)
N = 128
for step in range(1000):
    idx = rng.choice(n_pool, N, replace=False)
    loss = infonce_loss(scores(g, h, torch.tensor(x_tr[idx]),
                               torch.tensor(y_tr[idx])))
    opt.zero_grad()
    loss.backward()
    opt.step()

def critic_np(a, b):
    """The trained critic on NumPy inputs, as an N x N NumPy array."""
    with torch.inference_mode():
        return scores(g, h, torch.tensor(a), torch.tensor(b)).numpy()
```

```{.python .input #mutual-information-infonce-train}
#@tab tensorflow
opt = tf.keras.optimizers.Adam(1e-3)
params = g.trainable_variables + h.trainable_variables

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss = infonce_loss(scores(g, h, x, y))
    opt.apply_gradients(zip(tape.gradient(loss, params), params))

N = 128
for step in range(1000):
    idx = rng.choice(n_pool, N, replace=False)
    train_step(tf.constant(x_tr[idx]), tf.constant(y_tr[idx]))

def critic_np(a, b):
    """The trained critic on NumPy inputs, as an N x N NumPy array."""
    return scores(g, h, tf.constant(a), tf.constant(b)).numpy()
```

```{.python .input #mutual-information-infonce-train}
#@tab jax
opt = optax.adam(1e-3)
state = opt.init(params)

@jax.jit
def train_step(params, state, x, y):
    grads = jax.grad(infonce_loss)(params, x, y)
    updates, state = opt.update(grads, state)
    return optax.apply_updates(params, updates), state

N = 128
for step in range(1000):
    idx = rng.choice(n_pool, N, replace=False)
    params, state = train_step(params, state, x_tr[idx], y_tr[idx])

def critic_np(a, b):
    """The trained critic on NumPy inputs, as an N x N NumPy array."""
    return np.asarray(scores(params, a, b))
```

The comparison that answers the question puts both critics on the *same*
batches. The evaluation cell cuts the evaluation pool into disjoint batches
of $N$ pairs, for $N$ from $2$ to $128$, and computes the InfoNCE bound of
the learned critic and of the exact critic on each.

```{.python .input #mdl-mi-estimation-training-a-critic-3}
def infonce_eval(score, x, y, N):
    """Mean InfoNCE bound over the disjoint batches of N pairs in (x, y)."""
    return np.mean([infonce_bound(score(x[i:i + N], y[i:i + N]))
                    for i in range(0, len(x), N)])

exact_np = lambda a, b: exact_scores(a, b, rho)   # the exact critic T = pmi

print('   N   ln N   learned   exact')
for N_ev in (2, 8, 32, 128):
    print(f'{N_ev:4d}  {np.log(N_ev):5.2f}  '
          f'{infonce_eval(critic_np, x_ev, y_ev, N_ev):7.2f}  '
          f'{infonce_eval(exact_np, x_ev, y_ev, N_ev):6.2f}')
print(f'true I(X;Y) = {mi_gauss(rho):.2f} nats')
```

The learned critic matches the exact one to within a few hundredths of a nat
at every $N$. The gap to the true $1.96$ nats belongs to the bound, and it
closes only as $N$ grows. At $N = 8$ the ceiling $\ln 8 = 2.08$
already exceeds the truth, so the shortfall of about $0.55$ nat there is the
slack of the bound, not its cap. A scalar pair gives the critic an easy job.
In high dimension its error is no longer negligible, and without a known
truth the two errors cannot be separated.

SimCLR-style systems use a different critic: they normalize the two
embeddings $z_x$ and $z_y$ and divide their cosine similarity by a
*temperature* $\tau$, $T(x, y) = \textrm{sim}(z_x, z_y)/\tau$
:cite:`Chen.Kornblith.Norouzi.ea.2020`. Lowering $\tau$ sharpens the softmax
over the candidates and concentrates the loss on the hardest negatives, those
that score closest to the positive. The temperature also caps the bound: a
cosine lies in $[-1, 1]$, so a cosine critic certifies less than $2/\tau$
nats whatever $N$ is (:exref:`ex_mdl-mutual-information-temperature`).

A tighter bound does not mean a better representation.
:citet:`Tschannen.Djolonga.Rubenstein.ea.2020` found that downstream quality
correlates poorly with the tightness of the bound being maximized: looser
bounds sometimes give better representations, and the critic architecture
and the choice of views matter as much as the information. A contrastive
loss is a capped lower bound on the information, and the objective should be
judged by the representations it trains.

### The Compression-Phase Debate
:label:`subsec_mdl-compression-debate`

The ceilings of this section have shaped a debate about deep learning itself.
:citet:`Shwartz-Ziv.Tishby.2017` binned the activations of each layer $Z$ of
ordinary classifiers during training and plotted the estimated pairs
$\big(I(X; Z), I(Y; Z)\big)$, with input $X$ and label $Y$. They reported a
fast *fitting* phase, in which both informations rise, followed by a long
*compression* phase, in which $I(X; Z)$ falls while $I(Y; Z)$ holds, and they
proposed compression as the mechanism by which deep networks generalize.
:citet:`Saxe.Bansal.Dapello.ea.2018` found that the compression appears with
saturating activations such as $\tanh$, which push activations into the edge
bins, and disappears with ReLU networks, which generalize just as well.

The binned estimate explains the discrepancy. In a deterministic network the
binned layer is a function of the input, so $H(Z_{\textrm{bin}} \mid X) = 0$
and the binned estimate is
$I(X; Z_{\textrm{bin}}) = H(Z_{\textrm{bin}}) - H(Z_{\textrm{bin}} \mid X) = H(Z_{\textrm{bin}})$,
the entropy of a quantization of the layer. It is at most the logarithm of
the number of inputs, $\ln 4096 = 8.3$ nats in the original experiment. For
continuous inputs and activations whose distribution has no atoms, the true
$I(X; Z)$ is infinite. The curves therefore track how the activations spread
over the bins, which saturating units change as they move into the edge bins
(:exref:`ex_mdl-mutual-information-compression-phase`).

Two questions remain open: how to estimate a large mutual information in
high dimension at all, when every method here is capped by the logarithm of
something it can afford, and why loose bounds still train good
representations.

## Summary

From samples, mutual information can be measured when it is small and the
dimension is low; beyond that, every method reports at most the logarithm of
something it can afford. Histograms and nearest-neighbor counts work in a
few dimensions, the latter never report more than
$\psi(n) - \psi(m) \approx \log(n/m)$, and a distribution-free lower bound
from $n$ samples is of order $\log n$ nats at most. Variational methods
replace the unknown densities by a decoder (Barber–Agakov) or a critic
(Donsker–Varadhan, NWJ, InfoNCE) whose value is a lower bound, and maximize
it. With the optimal critic the DV and NWJ bounds are exact, but rare pairs
carry their expectations, so a typical batch overshoots. InfoNCE is steady,
below the truth even with an exact critic, and at most $\log N$. On the
Gaussian pair a learned critic reproduces the exact one to a few hundredths
of a nat, so the gap there belongs to the bound, which only a larger $N$
closes; the tightness of the bound does not predict the quality of the
representation it trains.

## Exercises

<!-- WRITER: three exercises overlap the new cells. Ex. 1 asks the reader to
"Write a function binned_mi(x, y, bins)": the section now defines it (same
signature, equal-width np.histogram2d bins), and (c) is now the body's
second histogram cell (x^3 and ranks at rho = 0.9). Ex. 2(e) asks to
simulate the share of batches with a pair e^pmi >= N and the batch average
at I = 6: the exact-critics table now prints both shares. Ex. 3(b) asks to
modify ksg_mi to return the mean of n_x and run d = 1, 4, 16: the body now
has `ksg_counts` and prints the mean strip count for d = 1, 2, 4, 8, 16. -->

1. [code] **Bias of the histogram estimator.** Write a function
   `binned_mi(x, y, bins)` that bins each coordinate into $b$ equal-width
   bins over its range (`np.histogram2d` does this), takes the cell
   frequencies $p_{ij}$ with marginals $p_i$ and $p_j$, and returns
   $\sum_{i,j} p_{ij} \log\frac{p_{ij}}{p_i\, p_j}$ over the occupied cells.
   Its estimate depends on the sample size $n$, on $b$, and on the
   coordinates of the data; mutual information depends on none of them.
    1. For an independent pair, $2n$ times the plug-in estimate is the
       $G$-statistic of the test for independence. When every cell's
       expected count is large, this statistic is approximately
       $\chi^2$-distributed with $(b-1)^2$ degrees of freedom. Predict the
       mean estimate at $\rho = 0$ for $n = 200{,}000$ and $b = 40$, run
       `binned_mi` on such a pair (you should find about $0.003$), and explain why
       the printed value is smaller than the prediction.
    1. Replace each coordinate by its rank divided by $n$ before binning, so
       that every bin of each axis holds the same number of samples. Average
       the estimate over $20$ independent pairs with $\rho = 0$ for
       $n \in \{2000, 20{,}000\}$ and $b \in \{10, 40\}$, and compare with the
       prediction of (a). For which setting does the prediction fail, and
       why?
    1. Apply $x \mapsto x^3$ to the first coordinate of a pair with
       $\rho = 0.9$ and $n = 200{,}000$, and estimate $I(X;Y)$ with and
       without the rank transform. Why is the rank-based estimate exactly
       unchanged?
    1. Repeat (c) at $\rho = 0.99$ with $b \in \{40, 100\}$, using ranks.
       Together with (b), what does this suggest about how $b$ should grow
       with $n$?
   :exlabel:`ex_mdl-mutual-information-histogram-bias`

1. **Heavy tails of the exponentiated pmi.** ● Let $(X, Y)$ be the Gaussian
   pair with correlation $\rho$, $I = -\tfrac{1}{2}\log(1 - \rho^2)$, and
   $Q = P_X \otimes P_Y$.
    1. Show that $\mathbb{E}_Q[e^{\textrm{pmi}}] = 1$ for every $\rho$,
       although at $\rho = 0.99$ a typical independent pair has
       $e^{\textrm{pmi}}$ near $0$. Which pairs carry the expectation?
    1. Show that
       $\mathbb{E}_Q[e^{2\,\textrm{pmi}}] = \int p^2/q = 1/(1 - \rho^2) = e^{2I}$.
       *Hint:* the integrand is proportional to a Gaussian density with
       precision matrix $2\boldsymbol{\Sigma}^{-1} - \mathbf{I}$, where
       $\boldsymbol{\Sigma}$ is the covariance of the pair, and
       $\det(2\boldsymbol{\Sigma}^{-1} - \mathbf{I}) = 1$.
    1. A random permutation of a batch leaves on average one $y$ paired with
       its own $x$. Show that such a fixed point adds $e^{\textrm{pmi}}/N$ to
       the batch mean of $e^{\textrm{pmi}}$ over the permuted pairs, with
       expectation $e^{2I}/N$: $0.43$, $23$ and $1270$ at $I = 2, 4, 6$ with
       $N = 128$. Why does `np.roll` avoid this?
    1. For the NWJ critic $T^* = 1 + \textrm{pmi}$, show that the batch
       estimate has mean $I$ when the mismatched pairs are independent draws,
       and that its mismatched-pair term then has variance $(e^{2I} - 1)/N$:
       standard deviations of $0.65$, $4.8$ and $36$ nats at $I = 2, 4, 6$
       with $N = 128$. Show that the rolled pairs $(x_j, y_{j+1})$ of the cell
       give the same variance for $N \geq 3$, although neighboring terms
       share a variable. *Hint:* condition on the shared pair.
    1. Simulate the batch average of $e^{\textrm{pmi}}$ over $128$ rolled
       pairs at $I = 6$, and report its median and the fraction of batches
       that contain a pair with $e^{\textrm{pmi}} \geq N$. What does each of
       DV and NWJ report on a batch whose average is near zero, and what does
       NWJ report on a batch that contains such a pair? Use the answers to
       explain the medians and the NWJ mean that
       :numref:`subsec_mdl-exact-critics` reports.
   :exlabel:`ex_mdl-mi-estimation-heavy-tails`

1. [code] **The sample-size ceiling of KSG.** The function `ksg_mi` counts
   neighbors with strict inequalities and excludes each point from its own
   counts.
    1. Run `ksg_mi(x, x)` for $n \in \{500, 2000\}$ and compare with
       $\psi(n) - \psi(m)$ and with $\log(n/m)$ for $m = 5$. How much does
       doubling $n$ add to the ceiling?
    1. Modify `ksg_mi` to also return the mean of `n_x`. Run $d \in \{1, 4, 16\}$
       at $\rho = 0.9$ and $n = 2000$. Using $\psi(j) \approx \log j$ and
       $n_x \approx n_y$, show that the estimate is about
       $\log\frac{m n}{\bar{n}_x^2}$, and reproduce the reported value at
       $d = 16$ from the mean count to within a few tenths of a nat.
    1. Repeat (b) at $d = 16$ with $\rho = 0.5$ and with $\rho = 0.99$,
       where the truth is $2.3$ and $31$ nats. Does the estimate still
       increase with the dependence? Could it rank the dependence of
       several pairs, and could it measure it?
    1. Let $Y$ be $X$ plus Gaussian noise of standard deviation $10^{-6}$,
       with $X$ standard normal and one-dimensional. What is the true
       $I(X; Y)$, and what does KSG report at $n = 2000$? Which ceiling
       binds?
   :exlabel:`ex_mdl-mi-estimation-ksg-ceiling`

1. **Barber–Agakov with a Gaussian decoder.** Take the standard bivariate
   Gaussian pair with correlation $\rho$, whose information is
   $-\tfrac{1}{2}\log(1 - \rho^2)$, and the decoders
   $q(x \mid y) = \mathcal{N}(x; a y, s^2)$ with parameters $a \in \mathbb{R}$
   and $s^2 > 0$. In the bound :eqref:`eq_mdl-ba_bound`, use the
   differential entropy $h(X) = \tfrac{1}{2}\log(2\pi e)$.
    1. Evaluate the right-hand side of :eqref:`eq_mdl-ba_bound` as a
       function of $a$ and $s^2$. Maximize it over both parameters and show
       that the maximum is the true information.
    1. A decoder trained with the squared error alone corresponds to the
       fixed variance $s^2 = 1$. Show that the best bound in this family is
       $\rho^2/2$, and compare it with the true value at $\rho = 0.9$ and
       $\rho = 0.99$. How does the gap behave as $\rho \to 1$?
    1. Let $\hat{x}(y)$ be any predictor of $X$ with mean squared error $D$.
       Use the decoder $\mathcal{N}(x; \hat{x}(y), D)$ to show that
       $I(X; Y) \geq -\tfrac{1}{2}\log D$. What is the corresponding bound
       when $X \sim \mathcal{N}(0, v)$? Compare it with the Gaussian
       rate–distortion function $R(D) = \tfrac{1}{2}\log(v/D)$ for $D < v$
       of :numref:`subsec_mdl-rate-distortion`, the least information any
       description of $X$ needs to reach squared error $D$.
   :exlabel:`ex_mdl-mutual-information-ba-gaussian-decoder`

1. **Temperature of a cosine critic.** SimCLR-style systems score candidates
   with $T(x, y) = s(x, y)/\tau$, where
   $s(x, y) = \textrm{sim}(z_x, z_y) \in [-1, 1]$ is a cosine similarity and
   $\tau > 0$ is the temperature. For one anchor $x$, write
   $s_j = s(x, y_j)$, with $y_1 = y^+$ the positive, and let $\ell$ be the
   loss inside the expectation :eqref:`eq_mdl-infonce_def`.
    1. Show that
       $\mathcal{L}_{\textrm{NCE}} \geq \log\big(1 + (N-1)e^{-2/\tau}\big)$,
       and hence $\hat{I}_{\textrm{NCE}} < 2/\tau$ for every $N$. Evaluate
       the cap $\log N - \log\big(1 + (N-1)e^{-2/\tau}\big)$ for $N = 4096$
       and $\tau \in \{1, 0.5, 0.1\}$ (you should find about $2.0$, $4.0$ and
       $8.3$ nats). When does the temperature rather than the batch size
       limit the bound?
    1. Show that $\partial \ell / \partial s_j = p_j/\tau$ for every negative
       $j \geq 2$, where $p_j$ is the softmax probability of candidate $j$.
       Express $\partial \ell / \partial s_1$ through the $p_j$.
    1. Show that $\tau \ell \to \max\{0, \max_{j \geq 2} s_j - s_1\}$ as
       $\tau \to 0$, and that

       $$\ell = \log N - \frac{1}{\tau}\Big(s_1 - \frac{1}{N}\sum_{j=1}^N s_j\Big) + O(\tau^{-2}) \quad \textrm{as } \tau \to \infty.$$

       Which negatives drive the gradient in each limit
       :cite:`Wang.Liu.2021`?
   :exlabel:`ex_mdl-mutual-information-temperature`

1. **Entropy bounds from samples.** ●
   :numref:`subsec_mdl-mi-distribution-free` states the theorem of
   :citet:`McAllester.Stratos.2020` informally and proves a version of it
   for entropy; this problem sharpens and applies that proof. Draw a sample
   $S$ of $n$ i.i.d. values from a discrete distribution $p$. The *type* of
   $S$ records, for each $i \geq 1$, how many distinct values occur exactly
   $i$ times in $S$; it forgets which values these are. Let $B$ be a
   function of the type that is a lower confidence bound for every discrete
   distribution: $P_{S \sim p^n}\big(B \leq H(p)\big) \geq 1 - \delta$ for
   every $p$, with $\delta < 1/2$. Let $\mathcal{T}_0$ be the type of a
   sample of $n$ distinct values.
    1. The text applies the guarantee to the uniform distribution on $n^2$
       values. Show that any integer $M \geq n^2/(2(1 - \delta))$ serves as
       well, so that $B(\mathcal{T}_0) \leq \log M$. By how much does this
       improve on $2 \log n$ at $\delta = 0.05$?
    1. Let $p$ be uniform on $M = 2^{256}$ values. How large must $n$ be
       before $2 \log n$ reaches $H(p)$? What does this say about measuring
       the entropy of a cryptographic key from samples?
    1. For $X = Y$ uniform on $M$ values, $I(X;Y) = H(X)$. Explain why a
       distribution-free lower bound on $I(X;Y)$ that depends on the sample
       of pairs only through its type obeys the same limit. Which assumption
       must an estimator add to certify more than $2 \log n$ nats from $n$
       samples?
   :exlabel:`ex_mdl-mutual-information-entropy-bounds-samples`

1. **Optimal InfoNCE critic.** ● For one anchor $x$ and one bag
   $v = (y_1, \ldots, y_N)$ with $y_1 = y^+$, write
   $\ell(x, v) = -\log\big(e^{T(x, y_1)} / \sum_{j=1}^N e^{T(x, y_j)}\big)$,
   so that $\mathcal{L}_{\textrm{NCE}}$ of :eqref:`eq_mdl-infonce_def` is
   its expectation.
    1. Show that replacing $T(x, y)$ by $T(x, y) + c(x)$, for any function
       $c$, leaves $\mathcal{L}_{\textrm{NCE}}$ unchanged.
    1. Shuffle the bag so that the positive sits at a uniformly random
       position $J$ and the other $N - 1$ positions hold the negatives. Show
       that

       $$P(J = i \mid x, y_1, \ldots, y_N) = \frac{e^{\textrm{pmi}(x, y_i)}}{\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}}.$$

       Use Gibbs' inequality to conclude that every critic
       $T(x, y) = \textrm{pmi}(x, y) + c(x)$ minimizes
       $\mathcal{L}_{\textrm{NCE}}$.
    1. For this critic, show that

       $$\hat{I}_{\textrm{NCE}} = I(X;Y) - \mathbb{E}\left[\log \frac{1}{N}\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}\right].$$

       Use the law of large numbers to argue that the correction tends to
       zero as $N \to \infty$ when $I(X;Y)$ is finite. Evaluate the
       correction by Monte Carlo at $\rho = 0.99$ and $N = 8$, and compare it
       with the gap between the exact-critic value (about $1.4$) and the
       truth ($1.96$) in :numref:`subsec_mdl-infonce-learned-critic`. How
       does the correction explain the curves of
       :numref:`subsec_mdl-infonce-ceiling`?
   :exlabel:`ex_mdl-mutual-information-optimal-critic`

1. **In-batch negatives and the symmetric loss.** The cells of
   :numref:`subsec_mdl-infonce-learned-critic` score all $N \times N$ pairs
   of a batch of $N$ independent pairs $(x_i, y_i)$ and use row $i$'s
   off-diagonal entries as the negatives of anchor $x_i$.
    1. Show that each row satisfies the hypotheses of the InfoNCE
       proposition: its candidates $y_j$, $j \neq i$, are draws from $P_Y$
       that are independent of $(x_i, y_i)$. Conclude that the mean row loss
       $\mathcal{L}$ certifies $I(X; Y) \geq \log N - \mathcal{L}$ in
       expectation.
    1. The rows share candidates, so their losses are dependent. Does this
       affect the validity of the bound, the expectation of the mean row
       loss, or its variance?
    1. CLIP uses the symmetric loss, the mean of the row-wise and the
       column-wise cross-entropies. Show that it bounds $I(X; Y)$ in the same
       way, with the same ceiling $\log N$.
    1. Count the network evaluations needed for the $N \times N$ scores of a
       joint critic $T(x, y)$, a network applied to the concatenated pair,
       and of the separable critic $g(x)^\top h(y)$. CLIP trains with
       batches of $N = 32{,}768$. How many evaluations does each critic need
       per batch?
   :exlabel:`ex_mdl-mi-estimation-in-batch-negatives`

1. **Negative sampling as shifted PMI.** ● Skip-gram with negative
   sampling of :numref:`subsec_negative-sampling` scores a center word $w$
   and a context word $o$ by $s(w, o) = \mathbf{u}_o^\top \mathbf{v}_w$ and
   draws $K$ noise words from a distribution $P$. Let $p(w, o)$ be the
   frequency of the pair among all (center, context) pairs of the corpus,
   with marginals $p(w)$ and $p(o)$. Averaged over the corpus, the loss of
   :numref:`subsec_negative-sampling` is

   $$\ell = -\sum_{w, o} p(w, o) \log \sigma\big(s(w, o)\big) - K \sum_{w} p(w) \sum_{o} P(o) \log \sigma\big(-s(w, o)\big),$$

   where $\sigma(s) = 1/(1 + e^{-s})$ is the sigmoid. Assume that the
   embeddings have enough dimensions for every score $s(w, o)$ to be chosen
   independently.
    1. Show that $\ell$ is minimized at
       $s^*(w, o) = \log \frac{p(w, o)}{p(w)\, P(o)} - \log K$, which equals
       $\textrm{pmi}(w, o) - \log K$ when $P$ is the unigram distribution
       $p(o)$ :cite:`Levy.Goldberg.2014`.
    1. word2vec draws noise words with probability proportional to their
       count raised to the power $0.75$ (:numref:`sec_word2vec_data`). Show
       that $s^*$ then differs from $\textrm{pmi}(w, o) - \log K$ by a term
       that depends on $o$ alone. For which context words is this term
       negative? How does it act on the rare pairs of problem
       :exref:`ex_mdl-mutual-information-rare-pairs`?
    1. Compare $s^*$ with the optimal InfoNCE critic of problem
       :exref:`ex_mdl-mutual-information-optimal-critic`b. Why does the
       binary classification of negative sampling fix the additive term that
       InfoNCE leaves free?
   :exlabel:`ex_mdl-mutual-information-negative-sampling-pmi`

1. [extended] **Compression phase at small scale.** ●●
   :citet:`Shwartz-Ziv.Tishby.2017` trained fully connected tanh networks of
   widths 12-10-7-5-4-3-2 on all $4096$ binary input patterns of $12$ bits.
   For each hidden layer $Z$ they binned every unit's activation into $30$
   equal intervals of $[-1, 1]$ and computed $I(X;Z)$ and $I(Y;Z)$ from the
   binned values. :numref:`subsec_mdl-compression-debate` describes the
   debate about the compression phase they reported.
    1. Let $X$ be uniform on the $4096$ patterns and let $Z_{\textrm{bin}}$
       be the binned layer of a deterministic network. Show that the binned
       estimate of $I(X;Z)$ equals $H(Z_{\textrm{bin}})$ and never exceeds
       $\ln 4096 \approx 8.32$ nats. When does it decrease during training?
    1. Fix a random vector $\mathbf{w} \in \mathbb{R}^{12}$ and let $Y = 1$
       when $\mathbf{w}^\top \mathbf{x}$ exceeds its median over the $4096$
       patterns. Train the tanh network, with a softmax over its two output
       units, by SGD with the cross-entropy loss on a random $85\%$ of the
       patterns. Plot the trajectory of every hidden layer in the
       information plane $\big(I(X;Z), I(Y;Z)\big)$ over training.
    1. Repeat (b) with ReLU hidden units, binning each unit into $30$ equal
       intervals between $0$ and its largest activation over the run.
    1. Record the fraction of tanh units with $|z| > 0.99$ during training.
       Does the decrease of the binned $I(X;Z)$ in (b) track this fraction?
       Which conclusions about compression survive the change of activation
       in (c) :cite:`Saxe.Bansal.Dapello.ea.2018`?
   :exlabel:`ex_mdl-mutual-information-compression-phase`

[Discussions](https://d2l.discourse.group/t/36866)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §28.5]{.kicker}

Estimating Mutual Information<br>**neighbors · bounds · InfoNCE**
:::
:::

::: {.slide title="KSG Reports 2.8 Nats for a Pair That Shares 13.3"}
[The problem]{.kicker}

- Nearest-neighbor counts, $2{,}000$ samples of a Gaussian pair:
  $d = 1$, truth $0.83$, estimate $0.8$–$0.9$.
- The same with $16$ coordinates: truth $13.3$, estimate $2.8$.
- A contrastive loss with $128$ candidates reads as information
  $\ln 128 - \mathcal{L}$, and can never exceed $\ln 128 = 4.85$.

::: {.d2l-note .warn}
Which of these numbers is a measurement?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- run KSG and name its two ceilings: sample size and dimension;
- derive the decoder and critic lower bounds;
- prove the InfoNCE bound and its $\log N$ cap;
- tell the error of a critic from the slack of its bound.

**The idea:** from samples, mutual information can only be bounded from
below, and the contrastive bound is a classification loss capped at
$\log N$.
**Running example:** $X, Y$ standard normal with correlation $\rho$,
$I = -\tfrac12\log(1-\rho^2)$: $0.83$ nats at $\rho = 0.9$, $1.96$ at
$\rho = 0.99$.
:::

::: {.slide title="Estimation Closes the Information-Theory Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Entropy, Cross-Entropy, and KL | surprise, KL, the loss floor |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| Divergences and Distances | four ways to compare two laws |
| Mutual Information | dependence and what it limits |
| **Estimating Mutual Information** | **lower bounds from samples** |
:::

::: {.col .narrow}
- **Uses:** MI as a KL divergence, the Gaussian pair, the f-GAN bound.
- **Skips:** upper bounds on MI; the bottleneck objective.
:::
:::
:::

::: {.slide title="A Histogram Errs in Both Directions"}
[Nonparametric]{.kicker}

Bin each axis into $b$ cells, apply the discrete formula to the
frequencies; $200{,}000$ samples:

@!mutual-information-gaussian-binned

- Coarse bins hide the ridge: $\rho = 0.99$ at $b = 10$ reads about half
  the truth.
- Fine bins invent information: an independent pair reads a few hundredths
  at $b = 160$.
:::

::: {.slide title="An Invertible Map Moves the Histogram, Not the Ranks"}
[Nonparametric]{.kicker}

Gaussian pair at $\rho = 0.9$, then $x \mapsto x^3$; $I$ stays $0.830$:

@!mdl-mutual-information-mutual-information-as-nonlinear-correlation-2

- Equal-width bins are stretched by the extreme values of $x^3$.
- Ranks do not move under an increasing map, so neither does their
  histogram.
:::

::: {.slide title="A Histogram Needs $b^d$ Cells"}
[Nonparametric]{.kicker}

- 2-D histograms already need the bin width chosen by hand.
- $d = 10$, $b = 10$: $10^{10}$ cells, nearly all empty.

::: {.d2l-note}
Let the data set the resolution: small cells where samples are dense.
:::
:::

::: {.slide title="KSG Counts Neighbors Instead of Cells"}
[Nonparametric]{.kicker}

::: {.cols .vc}
::: {.col}
Box of half-width $r_i$ to the $m$-th joint neighbor; strips of the same
width hold $n_x^{(i)}$ and $n_y^{(i)}$ points.

$$\hat I = \psi(m) + \psi(n) - \overline{\psi(n_x+1) + \psi(n_y+1)}$$

With $\psi(j) \approx \log j$, each term is
$\log\frac{m/n}{(n_x/n)(n_y/n)}$: a pmi from counts.
:::

::: {.col .fig}
@fig:mdl-it-ksg-box
:::
:::
:::

::: {.slide title="KSG Cannot Report More Than $\psi(n) - \psi(m)$"}
[Nonparametric]{.kicker}

- The $m - 1$ points inside the box lie in both strips:
  $n_x, n_y \geq m - 1$.
- $\psi$ increases, so each term is at most
  $\psi(m) + \psi(n) - 2\psi(m) = \psi(n) - \psi(m)$.
- $Y = X$: the strips hold exactly the box's points, equality.

::: {.d2l-note}
$n = 2{,}000$, $m = 5$: ceiling $6.09$ nats $\approx \log(n/m) = 5.99$.
Doubling $n$ adds about $0.69$.
:::
:::

::: {.slide title="KSG Saturates Near 3 Nats in 16 Dimensions"}
[Nonparametric]{.kicker}

@!mdl-mi-estimation-ksg-in-dimension

- A box holding $m/n$ of the data in $2d$ dimensions has sides about
  $(m/n)^{1/(2d)}$ of its width: $0.05$ at $d = 1$, $0.83$ at $d = 16$.
- The strips hold about $26$ of $2{,}000$ points at $d = 8$ and at
  $d = 16$: the counts stop falling while the truth doubles.
:::

::: {.slide title="Check: What Does KSG Print for $Y = X$?"}
[Check]{.kicker}

$2{,}000$ samples of $Y = X$, $X$ standard normal, $m = 5$. The true
information is infinite. What does KSG print?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\psi(2000) - \psi(5) = 6.09$ nats, the ceiling: the strips
hold only the $4$ points inside each box.
:::
:::
:::

::: {.slide title="A Million Samples Cannot Certify 30 Nats"}
[Distribution-free]{.kicker}

- $n$ draws, uniform on $M$ symbols: a repeat with probability at most
  $n(n-1)/(2M)$.
- $M = n^2$: all distinct with probability $> \tfrac12$, so a bound valid
  at $\delta < \tfrac12$ reports at most $2\log n$ on an all-distinct
  sample.
- $M = 2^{256}$, entropy $177$ nats: at $n = 10^6$ all distinct except with
  probability $4.3 \times 10^{-66}$; the bound reports at most $27.6$.

[McAllester & Stratos (2020): any distribution-free lower bound is
$O(\log n)$.]{.small}
:::

::: {.slide title="A Decoder Certifies a Lower Bound"}
[Barber–Agakov]{.kicker}

$$I(X;Y) \geq H(X) + \mathbb{E}\log q(x \mid y)$$

The gap is $\mathbb{E}_y D_{\textrm{KL}}\big(p(\cdot\mid y)\,\|\,q(\cdot\mid y)\big) \geq 0$.

- A 10-class classifier at loss $0.3$ certifies $\ln 10 - 0.3 = 2.0$
  nats, never more than $2.30$.
- A squared-error decoder of the pair certifies $\rho^2/2 = 0.41$ of
  $0.83$ nats, never more than $0.5$.
:::

::: {.slide title="A Critic Certifies a Lower Bound"}
[Donsker–Varadhan]{.kicker}

$$D_{\textrm{KL}}(P\|Q) \geq \mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T]$$

- Tilt $Q$ by the critic: $\tilde q = q\,e^T/Z$, $Z = \mathbb{E}_Q[e^T]$.
- The gap is $\mathbb{E}_P\log(p/\tilde q) = D_{\textrm{KL}}(P\|\tilde Q) \geq 0$.
- Equality at $T = \log(p/q) + c$.
- For MI: $P = P_{XY}$, $Q = P_X P_Y$, estimated by rolling the batch.
:::

::: {.slide title="The Tangent Line $\log z \le z/e$ Gives NWJ"}
[NWJ]{.kicker}

$$I(X;Y) \geq \mathbb{E}_{P_{XY}}[T] - e^{-1}\,\mathbb{E}_{P_X P_Y}[e^T]$$

- No logarithm of an average: the minibatch value is unbiased for the
  bound. The minibatch DV value is biased upward (Jensen).
- Tight at $T^* = 1 + \textrm{pmi}$.
- But $e^{T^*-1}$ is the density ratio: huge on rare pairs, tiny on
  typical ones.
:::

::: {.slide title="InfoNCE Is an $N$-Class Cross-Entropy"}
[InfoNCE]{.kicker}

::: {.cols .vc}
::: {.col}
One positive $y^+$, $N - 1$ negatives from $P_Y$:

$$\mathcal{L} = -\mathbb{E}\log\frac{e^{T(x,y^+)}}{\sum_{j} e^{T(x,y_j)}}$$

Estimate: $\log N - \mathcal{L}$. The classes are the other examples of
the batch: CPC, SimCLR, CLIP.
:::

::: {.col .fig}
@fig:mdl-it-infonce-pos-neg
:::
:::
:::

::: {.slide title="Proof I: Independent Negatives Add Nothing"}
[InfoNCE bound]{.kicker}

Bag $V = (Y_1 = Y^+, Y_2, \ldots, Y_N)$; negatives independent of
$(X, Y_1)$.

$$I(X; V) = I(X; Y_1) + \underbrace{I(X; Y_{2:N} \mid Y_1)}_{=\,0} = I(X;Y)$$

Given $Y_1$, the negatives are still independent of $X$.
:::

::: {.slide title="Proof II: The Softmax Ratio Is a Normalized Model"}
[InfoNCE bound]{.kicker}

$$\gamma(x, v) = \frac{e^{T(x, y_1)}}{\frac1N \sum_j e^{T(x, y_j)}} \in (0, N]$$

- Under $p(v)$, all $N$ candidates i.i.d. from $P_Y$, so exchangeable.
- The $N$ ratios have equal means, and they sum to $N$:
  $\mathbb{E}_{p(v)}[\gamma(x, V)] = 1$.
- So $q(v \mid x) = p(v)\,\gamma(x, v)$ is a distribution over bags.
:::

::: {.slide title="Proof III: Barber–Agakov on the Bag"}
[InfoNCE bound]{.kicker}

$$I(X;V) \geq \mathbb{E}\log\frac{q(V\mid X)}{p(V)} = \mathbb{E}\log \gamma = \log N - \mathcal{L}$$

::: {.d2l-note}
$I(X;Y) \geq \log N - \mathcal{L}$, and $\mathcal{L} \geq 0$, so the
estimate is at most $\log N$.
:::

Naming the right one of $N$ candidates shows $\log N$ nats, never more.
:::

::: {.slide title="Check: The Most a Batch of 256 Can Certify"}
[Check]{.kicker}

InfoNCE with $N = 256$ candidates. What is the largest estimate? And with
a cosine critic $T = \mathrm{sim}/\tau$ at $\tau = 0.5$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\ln 256 = 5.55$ nats. A cosine lies in $[-1, 1]$, so the loss
is at least $\log(1 + 255\,e^{-4}) = 1.74$: at most $3.8 < 2/\tau = 4$ nats.
The temperature binds first.
:::
:::
:::

::: {.slide title="An Exact Critic Still Bends Below $\log N$"}
[InfoNCE ceiling]{.kicker}

@!mutual-information-log-n-ceiling

- $T = \textrm{pmi}$, $N = 128$: $3.7$ at $I = 4$, cap $4.85$.
- Slack: $\mathbb{E}\log\frac1N\sum_j e^{\textrm{pmi}(x, y_j)} \geq I - \log N$.
:::

::: {.slide title="At $I = 6$ NWJ's Typical Batch Reads 7, DV's Reads Far More"}
[Exact critics]{.kicker}

$N = 128$, exact critics, median over $200$ batches:

| true $I$ | DV | NWJ | InfoNCE |
|---|---|---|---|
| 2 | 2.0–2.1 | 2.0–2.1 | 1.95 |
| 4 | 4.0–4.3 | 4.0–4.3 | 3.7 |
| 6 | far above 6 | 6.9 | 4.6 |

- Typical batch: no mismatched pair looks like a partner, so
  $\overline{e^{\textrm{pmi}}} \approx 0$: NWJ reads $I + 1$.
- NWJ's variance is $(e^{2I} - 1)/N$: sd $36$ nats at $I = 6$.
:::

::: {.slide title="The Learned Critic Matches the Exact One"}
[Training a critic]{.kicker}

Dual encoder $T = g(x)^\top h(y)$, $1{,}000$ Adam steps at $N = 128$,
$\rho = 0.99$:

@!mdl-mi-estimation-training-a-critic-3

The gap to $1.96$ is the bound's: $\ln 8 = 2.08 > 1.96$, yet both critics
read $1.4$.
:::

::: {.slide title="A Tighter Bound Is Not a Better Representation"}
[Caveats]{.kicker}

- Downstream quality correlates poorly with the tightness of the bound;
  critic architecture and views matter as much.
- Binned $I(X; Z)$ of a deterministic layer is $H(Z_{\mathrm{bin}})$, at
  most $\ln 4096 = 8.3$: the "compression phase" tracks the binning.

[Tschannen et al. (2020); Shwartz-Ziv & Tishby (2017); Saxe et al.
(2018).]{.small}
:::

::: {.slide title="Check: Whose Fault Is the Missing Half Nat?"}
[Check]{.kicker}

At $N = 8$ the learned critic reports about $1.4$ nats for a pair that shares
$1.96$. Critic, optimizer, or bound?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The bound: the exact critic on the same batches also reports
about $1.4$. Only a larger $N$ closes the gap.
:::
:::
:::

::: {.slide title="Every Estimate Here Is Capped by a Logarithm"}
[Wrap-up]{.kicker}

- KSG: $0.8$–$0.9$, $2.5$, $2.8$ for $0.83$, $3.3$, $13.3$; ceiling
  $\psi(n) - \psi(m) = 6.1$.
- Distribution-free: at most about $2\log n$, $27.6$ nats at $n = 10^6$.
- Decoder: $0.41$ of $0.83$. At $I = 6$, $N = 128$: NWJ $7$, DV several
  nats high, InfoNCE $4.6 < 4.85$.
- Learned critic = exact critic to a few hundredths.

**Next:** the chapter's further reading; then dynamics: ODEs, SDEs and
diffusion.
:::
