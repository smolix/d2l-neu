# Estimating Mutual Information
:label:`sec_mdl-mi-estimation`

Every mutual information we have computed so far came from a closed-form joint
density. In practice, data never come with a density attached. Instead, we are
given a collection of pairs, such as spoken sentences with their transcripts or
search queries with the documents that answered them, and from those samples
alone we want to quantify how much one member of a pair reveals about the other.
Representation learning pushes this one step further: it trains models to make
that shared information large, turning the estimate itself into a training
objective.

A familiar contrastive loss makes the challenge concrete. Systems such as SimCLR
and CLIP train on paired *views* of the same example, such as two random crops
of a photograph or an image paired with its caption. Given one view, the model
learns to identify its true partner among a batch of candidates, say $128$ of
them, by minimizing a cross-entropy loss known as the InfoNCE loss. Papers
frequently subtract this loss from $\ln 128 = 4.85$ and report the difference as
the mutual information between the two views. Yet that difference can never
exceed $4.85$ nats, no matter how much information the views actually share: a
model that picks the right partner every time has only proved that it can
distinguish $128$ candidates. Because we both train models on such quantities
and interpret them as measurements, we need to distinguish a genuine measurement
from a lower bound, and to separate the slack of a bound from the error of a
learned model.

There are two main ways to estimate mutual information from samples. The direct
approach counts samples, either inside histogram bins or within nearest-neighbor
regions; it works well for small amounts of information between low-dimensional
variables, but breaks down for vectors. The second approach replaces the unknown
densities with a learned model arranged so that the resulting expectation is a
lower bound on $I(X; Y)$, and then maximizes that bound. Two results anchor the
rest of this section: the fundamental limit on what any estimator can guarantee
without distributional assumptions (:numref:`subsec_mdl-mi-distribution-free`),
and the variational bound behind the InfoNCE loss together with its logarithmic
ceiling (:numref:`sec_mdl-infonce`).

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

## A Pair with Known Information

To judge an estimator fairly, we need a benchmark whose exact mutual information
we already know. Throughout this section we use the standard bivariate Gaussian
pair from :numref:`subsec_mdl-mi-gaussian` with correlation $\rho$, which shares

$$I(X; Y) = -\tfrac{1}{2}\log\left(1 - \rho^2\right)$$

nats: $0.83$ nats at $\rho = 0.9$ and $1.96$ nats at $\rho = 0.99$. All
logarithms in this section are natural.

The variational bounds we will derive rely on the *pointwise mutual information*
(:numref:`subsec_mdl-pmi`),
$\textrm{pmi}(x, y) = \log \frac{p(x, y)}{p(x)\,p(y)}$, which measures how much
more likely a specific pair $(x, y)$ is under the joint distribution than under
independence. For our standard Gaussian pair, the pmi is a simple quadratic
function:

$$\textrm{pmi}(x, y) = -\tfrac{1}{2}\log(1 - \rho^2) - \frac{x^2 - 2\rho x y + y^2}{2(1 - \rho^2)} + \frac{x^2 + y^2}{2}.$$

Two identities follow immediately from this definition. First, averaging
$\textrm{pmi}(x, y)$ over pairs drawn from the joint distribution gives
$I(X; Y)$. Second, averaging its exponential $e^{\textrm{pmi}(x, y)}$ over
*independent* draws of $x$ and $y$ gives $1$, because
$p(x)\,p(y)\,e^{\textrm{pmi}(x, y)} = p(x, y)$ integrates to $1$. Let us verify
both identities on samples. For the second identity, we break the dependence by
pairing each $x$ with the $y$ from the next sample in the array. We call these
*mismatched pairs*; each one is an independent draw from $P_X \otimes P_Y$, and
we will reuse this simple trick whenever a bound requires independent pairs.

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

Both sample averages match their theoretical values to within about $0.01$.
Every variational bound in this section builds on these two identities, and it
is the second one that will cause almost all of our statistical trouble. At
$\rho = 0.9$, $e^{\textrm{pmi}}$ is well behaved: its variance over independent
pairs is $e^{2I} - 1$ (Exercise :exref:`ex_mdl-mi-estimation-heavy-tails`),
which is only $4.3$ here. At $I = 6$, however, that variance explodes to
$160{,}000$, leaving the same $200{,}000$-sample average uncertain by roughly
$0.9$.

## Nonparametric Estimation
:label:`sec_mdl-mi-hard`

The most direct way to estimate a probability distribution from data is to count
how many samples fall into each region of space. The key design choice is how to
define those regions. We begin with the simplest choice: a fixed grid.

### Histograms
:label:`subsec_mdl-mi-high-dim`

The histogram estimator partitions the range of each variable into $b$
equal-width intervals, counts how many samples land in each of the $b \times b$
cells, and plugs the resulting empirical frequencies into the discrete formula
for mutual information. Despite its simplicity, this estimator suffers from two
competing biases that push in opposite directions.

The first bias pulls the estimate *down*. Replacing a continuous value by its
bin index applies a deterministic function to $X$, and by the data-processing
inequality (:numref:`subsec_mdl-dpi`) any function of $X$ shares at most as
much information with $Y$ as $X$ itself does. Binning can therefore only destroy
information, and the loss is severe whenever the dependence is narrower than a
bin. At $\rho = 0.99$, for example, the conditional standard deviation of $y$
given $x$ is $\sqrt{1 - \rho^2} = 0.14$. Dividing a sample that spans roughly
$[-4.5, 4.5]$ into $10$ bins produces cells of width $0.9$, more than six times
wider than the conditional spread.

The second bias pulls the estimate *up*. The plug-in estimate is the KL
divergence between the empirical joint table and the product of its row and
column marginals. It is therefore always nonnegative, and equals zero only if
the table factorizes exactly. Even when $X$ and $Y$ are completely independent,
random sampling fluctuations prevent exact factorization, so noise masquerades
as dependence. The fewer samples each cell contains, the worse this upward bias
becomes.

On top of these two errors, a histogram has a hard ceiling. Its estimate cannot
exceed the entropy of the binned $X$, and $b$ bins can carry at most $\log b$
nats: just $2.30$ nats for ten bins, no matter how tightly $X$ and $Y$ are
linked. We will encounter a version of this ceiling in every estimator in this
section: no method can report more information than the logarithm of the number
of states it can distinguish.

Let us implement this estimator as `binned_mi` and test it on $200{,}000$
samples from four distributions using $b = 10$, $40$, and $160$: an independent
pair, the Gaussian pair at $\rho = 0.9$ and $\rho = 0.99$, and the parabolic
pair $Y = X^2 + 0.3\,\epsilon$ from
:numref:`sec_mdl-mi-nonlinear-correlation`, which has zero linear correlation
yet shares $1.141$ nats.

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

Both biases show up clearly in the table. With $b = 10$, every dependent pair
reads low: roughly $0.6$ instead of $0.83$ nats at $\rho = 0.9$, and only about
half the true value at $\rho = 0.99$ and for the parabola. With $b = 160$, the
independent pair is credited with a few hundredths of a nat, and the pair at
$\rho = 0.9$ actually overshoots its true value, something coarse binning alone
could never cause. At $\rho = 0.99$ with $b = 160$, the estimate lands within
about $0.01$ of the truth, but only through a fortunate cancellation of opposite
errors; the row directly above shows the same grid overshooting. When the true
value is unknown, nothing in the table tells us which column to trust.

A third weakness is sensitivity to coordinates. Applying an invertible
transformation to either variable leaves mutual information unchanged
(:numref:`sec_mdl-mi-nonlinear-correlation`), yet it distorts a fixed-width
grid. Suppose we replace $x$ with $x^3$ in the Gaussian pair at $\rho = 0.9$.
The largest of $200{,}000$ standard normal draws is around $\pm 4.5$, so $x^3$
spans roughly $\pm 90$. Each of $40$ equal-width bins is then about $4.5$ wide,
which packs roughly nine out of ten samples into the two or three bins around
zero. A clean fix is to bin the *ranks*, meaning each value's position in the
sorted sample. Any strictly increasing map preserves ranks, and every rank bin
automatically holds the same number of samples.

```{.python .input #mdl-mutual-information-mutual-information-as-nonlinear-correlation-2}
x, y = gauss_pair(rng, 0.9, n)
rank = lambda a: np.argsort(np.argsort(a))       # position in sorted order
print(f'true I = {mi_gauss(0.9):.3f} nats; histograms with b = 40:')
for name, u in (('x', x), ('x**3', x**3)):
    print(f'{name:4}: correlation {np.corrcoef(u, y)[0, 1]:.3f}, '
          f'raw {binned_mi(u, y, 40):.3f}, '
          f'ranks {binned_mi(rank(u), rank(y), 40):.3f} nats')
```

Even though the true mutual information remains $0.830$ nats, cubing $x$ drops
the linear correlation from $0.90$ to about $0.7$ and slashes the raw histogram
estimate to between $0.3$ and $0.5$ nats (with the exact value depending on the
few extreme draws that set the bin span). Binning the ranks, by contrast, gives
the exact same estimate for $x$ and $x^3$, digit for digit, and stays within
$0.02$ nats of the truth.

Ranks fix the coordinate problem, but nothing can save histograms from high
dimension. A pair of five-dimensional vectors requires a ten-dimensional grid;
putting just ten bins along each axis creates $10^{10}$ cells, almost all of
which will be empty for any realistic sample size. To handle vectors, an
estimator must let the data themselves set the resolution, using small regions
where samples are dense and larger ones where they are sparse.

### The KSG Estimator
:label:`subsec_mdl-ksg`

The estimator of :citet:`Kraskov.Stogbauer.Grassberger.2004`, known as KSG,
replaces fixed histogram cells with adaptive nearest-neighbor regions around
each data point. It is the standard nonparametric tool for pairs of
low-dimensional variables. (This subsection and the next can be skipped on a
first reading.)

KSG measures distances using the largest coordinate difference (the Chebyshev,
or $\ell_\infty$, distance), so that a ball of radius $r$ is a square in two
dimensions and a hypercube in higher dimensions. Around each sample point $i$,
we draw the smallest box in the joint $(x, y)$ space that reaches its $m$-th
nearest neighbor, and let $r_i$ denote the half-width of that box. By
construction, this box contains a fraction $m/n$ of the $n$ samples, giving a
local estimate of the joint probability mass near $(x_i, y_i)$.

Next, we measure the two marginal probabilities at the exact same scale $r_i$.
Let $n_x^{(i)}$ be the number of other samples falling strictly inside the
vertical strip $|x - x_i| < r_i$, and let $n_y^{(i)}$ be the number falling
strictly inside the horizontal strip $|y - y_i| < r_i$. Sharing the half-width
$r_i$ is what lets raw counts stand in for probability densities: if the
densities are roughly constant across the box, the box holds probability mass
about $p(x_i, y_i)\,(2r_i)^2$ while the two strips hold about
$p(x_i)\,2r_i$ and $p(y_i)\,2r_i$. When we divide the box fraction by the
product of the two strip fractions, the geometric factors $2r_i$ cancel out.
Taking the logarithm produces a local estimate of the pointwise mutual
information computed purely by counting. :numref:`fig_mdl-ksg-box` illustrates
these three counts at a single point.

![The KSG count at one sample point $i$ (the star), with $m = 3$. The box of half-width $r_i$ reaches the third nearest neighbor in the joint plane (the ring on its left edge, which lies in the horizontal strip and counts toward $n_y^{(i)}$ only), while two neighbors lie strictly inside. The vertical strip of the same half-width holds $n_x^{(i)}$ other samples and the horizontal strip holds $n_y^{(i)}$ (blue and green dots lie in one strip only). Each summand of the estimator is approximately $\log\frac{m/n}{(n_x/n)(n_y/n)}$, a pointwise mutual information computed from counts.](../img/mdl-it-ksg-box.svg)
:label:`fig_mdl-ksg-box`

Averaging these local log ratios across all $n$ points estimates $I(X; Y)$,
with one finite-sample correction. Taking the logarithm of a small count is a
biased estimate of a log probability: for the box reaching the $m$-th of
$n - 1$ neighbors, the expected log probability mass is
$\mathbb{E}[\log p] = \psi(m) - \psi(n)$ rather than $\log(m/n)$, where
$\psi = (\log \Gamma)'$ is the digamma function (we omit the derivation).
Replacing logarithms with $\psi$ gives the KSG estimator:

$$\hat{I}_{\textrm{KSG}} = \psi(m) + \psi(n) - \frac{1}{n} \sum_{i=1}^{n} \left(\psi\left(n_x^{(i)} + 1\right) + \psi\left(n_y^{(i)} + 1\right)\right).$$
:eqlabel:`eq_mdl-ksg`

Because $\psi(j) \approx \log j$ for large $j$, each summand is approximately
$\log\frac{m\,n}{n_x^{(i)}\, n_y^{(i)}}$, matching the local log ratio of the
box fraction to the product of the strip fractions. The $+1$ inside the strip
terms aligns them with the box: $\psi(m)$ corresponds to a box whose boundary
passes through its $m$-th neighbor, so a strip containing $n_x^{(i)}$ points
strictly inside is treated as if its boundary passed through the
$(n_x^{(i)} + 1)$-th point. Both strip counts use strict inequalities and
exclude point $i$ itself, so a neighbor lying right on a strip boundary is not
counted. Because every radius $r_i$ adapts automatically to the local sample
density, there is no bin width or kernel bandwidth to tune.

We can compute all neighbor counts at once from the $n \times n$ matrices of
Chebyshev distances in $x$ and $y$ using `cdist`, taking their elementwise
maximum for the joint distance.

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

Because every summand in :eqref:`eq_mdl-ksg` depends solely on neighbor counts,
counting alone places a hard ceiling on what KSG can report.

**Proposition (sample-size ceiling of KSG).** *For any sample of $n$ points
in which no two joint distances from the same point are equal,*

$$\hat{I}_{\textrm{KSG}} \leq \psi(n) - \psi(m),$$
:eqlabel:`eq_mdl-ksg-ceiling`

*with equality when $Y = X$.*

**Proof.** Without distance ties, exactly $m - 1$ samples lie strictly closer
to point $i$ than its $m$-th neighbor. Since the joint Chebyshev distance is
the larger of the two coordinate distances, each of these $m - 1$ points
satisfies both $|x_j - x_i| < r_i$ and $|y_j - y_i| < r_i$, so it lies in both
strips: $n_x^{(i)} \geq m - 1$ and $n_y^{(i)} \geq m - 1$. Because $\psi$ is
strictly increasing, every summand is at most
$\psi(m) + \psi(n) - 2\psi(m) = \psi(n) - \psi(m)$. When $Y = X$, the joint
and marginal distances coincide, so both strips contain only the $m - 1$ points
strictly inside the box and every summand attains the bound. $\blacksquare$

For $n = 2{,}000$ and $m = 5$, this ceiling is $\psi(2000) - \psi(5) = 6.09$
nats, close to $\log(n/m) = 5.99$. Even for $Y = X$ with continuous $X$, where
the true mutual information is infinite, KSG reports $6.1$ nats, and doubling
the sample size raises that ceiling by only about $\log 2$. We are seeing the
histogram's $\log b$ ceiling again, with adaptive neighborhoods in place of
bins: a neighbor-counting estimator can report at most the logarithm of the
number of distinct neighborhoods the sample affords, roughly $n/m$.

### KSG in High Dimension
:label:`subsec_mdl-ksg-dimension`

Unfortunately, in higher dimensions KSG breaks down well below its sample-size
ceiling, and for a geometric reason. Consider $d$ independent copies of our
Gaussian pair, each with $\rho = 0.9$, so that $X$ and $Y$ are $d$-dimensional
vectors. Because the log density ratio splits into a sum of $d$ independent
terms, mutual information is additive across independent coordinates, making
the true value $d \times 0.83$ nats.

KSG relies on the joint and marginal densities being nearly constant across
each neighbor box, an assumption that collapses as dimension grows. With
$n = 2{,}000$ and $m = 5$, each box captures a fraction $m/n = 0.0025$ of the
sample. If points were spread uniformly across $2d$ dimensions, such a box
would span a fraction $(m/n)^{1/(2d)}$ of the data's width along every axis:
$0.05$ at $d = 1$, jumping to $0.47$ at $d = 4$ and $0.83$ at $d = 16$. Yet the
dependence lives inside a thin band around $y = \rho x$: given $x$, each
coordinate of $y$ has standard deviation $\sqrt{1 - \rho^2} = 0.44$, roughly
$0.07$ of the total span (about $-3.3$ to $3.3$) of the sample. A box many
times wider than that band smears out the dependence just like an overly coarse
histogram bin. Let us run KSG on $2{,}000$ samples for $d = 1, 2, 4, 8, 16$,
recording both the estimate and the average strip count, along with the
$Y = X$ case that attains the ceiling.

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

In one and two dimensions, KSG stays within about $0.1$ nat of the truth. By
$d = 4$, it drops to roughly $2.5$ nats against a true value of $3.3$, nearly a
full nat low. After that it stalls: at both $d = 8$ and $d = 16$, KSG reports
just under $3$ nats even as the true mutual information doubles from $6.6$ to
$13.3$. Notice that the sample-size ceiling is not what stops it here, since
$3$ nats is less than half of the $6.1$-nat ceiling printed for $Y = X$.

The average strip counts expose why the estimate stalls. Each summand is
approximately $\log\frac{m\,n}{n_x n_y}$, so the estimate can climb only if
the marginal strips empty out. They drop from about $70$ points at $d = 1$ to
about $31$ at $d = 4$, and then level off near $26$. To keep boxes as local at
$d = 16$ as they were at $d = 1$, we would need $m/n = 0.05^{32}$, requiring a
sample of roughly $10^{42}$ points. The sample size needed to maintain local
neighborhoods grows exponentially with the dimension.

### Limits of Distribution-Free Estimation
:label:`subsec_mdl-mi-distribution-free`

Perhaps a more clever estimator could bypass this bottleneck. To test that
hope, let us ask for something weaker than an exact measurement: a *lower
bound*. Specifically, we seek a function $B$ of $n$ samples such that
$B \leq I(X; Y)$ with probability at least $0.95$ *for every distribution* of
the pair $(X, Y)$. (Any confidence level above $1/2$ would work just as well.)
Lower bounds are also what representation learning wants, since we want to be
confident that two views share a large amount of information.

The simplest possible case already reveals the fundamental barrier. Set
$Y = X$ for a discrete variable $X$, so that $I(X; Y) = H(X)$. If an estimator
treats the discrete symbols as arbitrary labels with no built-in metric or
ordering, it cannot exploit *which* symbols appeared. All it can inspect is the
pattern of repetitions: how many distinct values appeared once, how many
appeared twice, and so on. This summary is called the *type* of the sample, and
we assume $B$ depends only on it. Four short steps establish the limit, with
the third step doing the real work:

1. Let $X$ be uniform over $M$ symbols and draw $n$ samples. Any given pair of
   draws collides with probability $1/M$, so by the union bound the probability
   that *some* two of the $n(n-1)/2$ pairs coincide is at most $n(n-1)/(2M)$.
1. Apply the $0.95$ guarantee to the uniform distribution on $M = n^2$ symbols,
   whose entropy is $2 \log n$: on a sample drawn from *this* distribution,
   $B \leq 2 \log n$ with probability at least $0.95$.
1. By the first step, a sample of size $n$ from $M = n^2$ symbols consists of
   $n$ distinct values with probability greater than $\tfrac{1}{2}$. Because
   two events whose probabilities sum to more than $1$ must overlap, there
   exists at least one all-distinct sample on which $B \leq 2 \log n$.
1. Every sample of $n$ distinct values has the exact same type, so $B$ must
   return the same value (at most $2 \log n$) on *every* all-distinct sample,
   no matter what distribution actually generated it.

Now test this bound on a uniform distribution over $M = 2^{256}$ symbols, whose
entropy is $256 \ln 2 = 177$ nats. With $n = 10^6$ samples, all draws are
distinct with probability at least $1 - 4.3 \times 10^{-66}$, and on that
sample $B$ can certify at most $2 \log 10^6 = 27.6$ nats. A million samples
cannot guarantee $177$ nats; in fact, they cannot even guarantee $30$. Exercise
:exref:`ex_mdl-mutual-information-entropy-bounds-samples` sharpens the constant
and asks how many samples would be needed.

:citet:`McAllester.Stratos.2020` proved the general theorem without restricting
the estimator to depend only on the sample type:

> **Theorem (McAllester–Stratos, informal).** Any distribution-free lower
> bound on $I(X; Y)$ that holds with fixed confidence from $n$ samples is
> $O(\log n)$.

The invariance of mutual information lies behind this barrier. Because any
invertible relabelling of $X$ or $Y$ leaves $I(X; Y)$ unchanged
(:numref:`sec_mdl-mi-nonlinear-correlation`), an estimator that makes no
assumptions about the coordinates must behave consistently under every possible
relabelling, leaving only the pattern of repetitions to work with. To break
past $O(\log n)$, an estimator must assume something about the data. KSG
assumes that coordinate distances are meaningful and that densities are smooth,
yielding an estimate with no finite-sample guarantee. The variational bounds of
:numref:`sec_mdl-mi-variational-bounds` can report numbers larger than
$\log n$, and in :numref:`subsec_mdl-exact-critics` we will see how much we
can actually trust them when they do.

### The Compression-Phase Debate
:label:`subsec_mdl-compression-debate`

The two limitations we have just seen, coordinate-dependent histograms and the
logarithmic sample ceiling, shed light on a famous debate in deep learning.
:citet:`Shwartz-Ziv.Tishby.2017` trained a small classifier on the $4{,}096$
binary patterns of $12$ bits, binned the activations of each hidden layer $Z$
throughout training, and plotted estimates of $I(X; Z)$ and $I(Y; Z)$ for input
$X$ and label $Y$. They observed a brief *fitting* phase in which both
quantities rose, followed by a long *compression* phase in which $I(X; Z)$
dropped while $I(Y; Z)$ remained high, and proposed that this compression
explains why deep networks generalize. Later,
:citet:`Saxe.Bansal.Dapello.ea.2018` showed that this compression phase appears
with saturating activations like $\tanh$, but disappears with ReLU
activations, even though ReLU networks generalize just as well.

What was the histogram actually measuring? In a deterministic network, the
binned activation $Z_{\textrm{bin}}$ is a deterministic function of the input
$X$. Consequently, $H(Z_{\textrm{bin}} \mid X) = 0$, and the binned estimate
simplifies to $I(X; Z_{\textrm{bin}}) = H(Z_{\textrm{bin}})$, simply the
entropy of the binned activations. That quantity can never exceed the logarithm
of the number of inputs, $\ln 4096 = 8.3$ nats, and it drops whenever distinct
inputs collapse into the same histogram cell, exactly what happens when
training drives saturating $\tanh$ units into the two outer bins near $\pm 1$.
The true mutual information $I(X; Z)$ itself never moves: as long as distinct
inputs produce distinct activations, $I(X; Z) = H(X)$ for a finite input set,
and for continuous inputs $I(X; Z)$ is infinite
(:numref:`subsec_mdl-mi-gaussian`). The plotted curves track how activations
cluster across fixed bins, a property of the estimator rather than of the
network's true mutual information (Exercise
:exref:`ex_mdl-mutual-information-compression-phase`).

## Variational Lower Bounds
:label:`sec_mdl-mi-variational-bounds`

Because counting cannot measure large mutual information in high dimensions,
variational methods change the goal. Rather than estimating the unknown
densities in $\log\frac{p(x, y)}{p(x)\,p(y)}$ directly, we introduce an
auxiliary model arranged so that its expected score is guaranteed to be a
*lower bound* on $I(X; Y)$, no matter how imperfect the model is. Training the
model to maximize that expectation tightens the bound without ever overshooting
$I(X; Y)$. Estimation thus becomes optimization. Its output carries two
distinct sources of error: approximation error from the auxiliary model, which
may fall short of the optimal one, and sampling error, because we must replace
population expectations with finite batch averages.

We examine four variational bounds in turn, unified in a single framework by
:citet:`Poole.Oord.Alemi.ea.2019`. The first uses a normalized decoder
(Barber–Agakov). The next two use an unnormalized scoring function, yielding
the Donsker–Varadhan (DV) bound and the bound of Nguyen, Wainwright, and Jordan
(NWJ). The fourth, InfoNCE, is important enough to merit its own section. Every
proof relies on a single tool, Gibbs' inequality
(:numref:`sec_mdl-information_theory`): a KL divergence is always nonnegative.

### The Barber–Agakov Bound

The simplest variational bound replaces the unknown posterior $p(x \mid y)$ with
a tractable model $q(x \mid y)$.

**Proposition (Barber–Agakov bound).** *For any conditional density
$q(x \mid y)$,*

$$I(X; Y) \geq H(X) + \mathbb{E}_{(x,y) \sim P_{X,Y}}\left[\log q(x \mid y)\right],$$
:eqlabel:`eq_mdl-ba_bound`

*with equality if and only if $q(x \mid y) = p(x \mid y)$. For continuous
$X$, the differential entropy $h(X)$ replaces $H(X)$.*

**Proof.** Starting from $I(X; Y) = H(X) - H(X \mid Y)$, we subtract the
right-hand side of :eqref:`eq_mdl-ba_bound`:

$$
\begin{aligned}
I(X; Y) - H(X) - \mathbb{E}\left[\log q(x \mid y)\right]
&= \mathbb{E}\left[\log \frac{p(x \mid y)}{q(x \mid y)}\right] \\
&= \mathbb{E}_{y}\left[D_{\textrm{KL}}\big(p(\cdot \mid y) \,\|\, q(\cdot \mid y)\big)\right]
\geq 0.
\end{aligned}
$$

The inequality follows from Gibbs' inequality applied for each $y$ and averaged
over $P_Y$. Equality holds if and only if the KL divergence vanishes for every
$y$, meaning $q(x \mid y) = p(x \mid y)$. $\blacksquare$

The Barber–Agakov bound :cite:`Barber.Agakov.2003` gives an
information-theoretic reading of an encoder–decoder: if a decoder $q(x \mid y)$
reconstructs $X$ from $Y$ with low cross-entropy, then $X$ and $Y$ must share a
large amount of information. Its practical limitation is the term $H(X)$, which
is usually unknown. Fortunately, $H(X)$ does not depend on $q$, so *maximizing*
the bound over $q$ never requires knowing it. Writing
$H(X) = -\mathbb{E}[\log p(x)]$ yields the equivalent ratio form
$I(X; Y) \geq \mathbb{E}\left[\log \left(q(x \mid y)/p(x)\right)\right]$, which
will be the key step in proving the InfoNCE bound.

Throughout this section, we say that a model *certifies* the value of its
bound. Keep in mind that this certification holds at the level of population
expectations; a finite batch average only estimates the certified value and can
fluctuate above or below it.

A standard classifier is a familiar special case. By the symmetry of mutual
information, we can swap $X$ and $Y$ and treat a classifier $q(y \mid x)$ as a
decoder of the label. This recovers the loss certificate of
:numref:`subsec_mdl-loss-certificate`: for $k$ balanced classes and held-out
cross-entropy $L$, $I(X; Y) \geq \log k - L$. Because $L \geq 0$, a $10$-class
classifier can never certify more than $\ln 10 = 2.30$ nats. InfoNCE will take
the exact same form.

Our Gaussian pair illustrates how much a mismatched decoder can lose. Training a
predictor by squared error alone corresponds to a unit-variance Gaussian
decoder $q(x \mid y) = \mathcal{N}(x; a y, 1)$. Since
$h(X) = \tfrac{1}{2}\log(2\pi e)$, the Barber–Agakov bound evaluates to
$\tfrac{1}{2} - \tfrac{1}{2}\mathbb{E}[(X - aY)^2] = \tfrac{1}{2} - \tfrac{1}{2}(1 - 2a\rho + a^2)$,
which peaks at $a = \rho$ with value $\rho^2/2$. At $\rho = 0.9$, this decoder
certifies only $0.41$ nats out of the true $0.83$, and no matter how close
$\rho$ gets to $1$, it can never certify more than $\tfrac{1}{2}$ nat. The
missing information is the penalty for fixing the decoder's variance at $1$
instead of matching the true conditional variance $1 - \rho^2$ (Exercise
:exref:`ex_mdl-mutual-information-ba-gaussian-decoder`).

### The Donsker–Varadhan Bound

Our second bound removes both requirements of Barber–Agakov: it needs neither
$H(X)$ nor a normalized density $q$. Because mutual information is the KL
divergence from the joint distribution to the product of its marginals
(:numref:`sec_mdl-mutual-information`), any lower bound on KL divergence
immediately bounds $I(X; Y)$. The Donsker–Varadhan bound achieves this using an
arbitrary scoring function $T$, called a *critic* (as in
:numref:`sec_mdl-f-gan-dual`), with no normalization constraint whatsoever.

**Proposition (Donsker–Varadhan bound).** *For distributions $P$ and $Q$ and
any function $T$ for which $\mathbb{E}_P[T]$ is defined and
$\mathbb{E}_Q[e^T] < \infty$,*

$$D_{\textrm{KL}}(P \| Q) \geq \mathbb{E}_P[T] - \log \mathbb{E}_Q\left[e^{T}\right],$$
:eqlabel:`eq_mdl-dv_bound`

*with equality if and only if $T = \log \frac{dP}{dQ} + c$ for some constant
$c$. When $P \ll Q$ and the divergence is finite, the supremum over $T$
attains $D_{\textrm{KL}}(P \| Q)$, giving the Donsker–Varadhan
representation of KL divergence* :cite:`Donsker.Varadhan.1983`.

**Proof.** If $P$ is not absolutely continuous with respect to $Q$, the
left-hand side is infinite and the inequality is trivial. Otherwise, use the
critic to reweight $Q$: define the *tilted distribution* $\tilde{Q}$ with
density $\tilde{q}(x) = q(x)\, e^{T(x)} / Z$, where $Z = \mathbb{E}_Q[e^T]$ is
the normalizing constant. Then

$$
\begin{aligned}
D_{\textrm{KL}}(P \| Q) - \left(\mathbb{E}_P[T] - \log Z\right)
&= \mathbb{E}_P\left[\log \frac{p}{q} - T + \log Z\right] \\
&= \mathbb{E}_P\left[\log \frac{p}{\tilde{q}}\right]
= D_{\textrm{KL}}(P \| \tilde{Q}) \geq 0
\end{aligned}
$$

by Gibbs' inequality. Equality holds if and only if $P = \tilde{Q}$, which
means $p/q = e^{T}/Z$, or equivalently $T = \log(p/q) + \log Z$. $\blacksquare$

The slack in the bound is simply $D_{\textrm{KL}}(P \| \tilde{Q})$, so a critic
is accurate precisely when tilting $Q$ by $e^T$ reproduces $P$. Notice also
that shifting $T$ by any constant $c$ cancels between the two terms. Setting
$P = P_{X,Y}$ and $Q = P_X \otimes P_Y$ gives a lower bound on mutual
information for any critic $T(x, y)$:

$$
I(X; Y) \geq \mathbb{E}_{P_{X,Y}}[T] - \log \mathbb{E}_{P_X \otimes P_Y}\left[e^{T}\right],
$$

whose optimal critic is $\textrm{pmi}(x, y)$ up to an arbitrary additive
constant. In practice, we estimate the first expectation by averaging $T$ over
observed pairs and the second by averaging $e^T$ over mismatched pairs from the
same minibatch. Parameterizing $T$ with a neural network yields *MINE* (mutual
information neural estimation) :cite:`Belghazi.Baratin.Rajeshwar.ea.2018`.

### The NWJ Bound

Although the Donsker–Varadhan bound is tight in expectation, the logarithm
outside $\mathbb{E}_Q[e^T]$ creates a subtle bias on minibatches. Because
$\log$ is concave, Jensen's inequality implies that the logarithm of a batch
average is on average smaller than the logarithm of the true expectation.
Subtracting a smaller number pushes the batch estimate
$\overline{T} - \log \overline{e^T}$ *upward* on average, an undesirable
property for an estimator of a lower bound.

We can eliminate the logarithm using the elementary tangent inequality
$\log z \leq z/e$, which holds for all $z > 0$ with equality at $z = e$.
Applying it to $Z = \mathbb{E}_Q[e^T]$ in :eqref:`eq_mdl-dv_bound` gives, for
every critic $T$,

$$I(X; Y) \geq \mathbb{E}_{P_{X,Y}}[T] - e^{-1}\, \mathbb{E}_{P_X \otimes P_Y}\left[e^{T}\right].$$
:eqlabel:`eq_mdl-nwj_bound`

This is the *NWJ bound* :cite:`Nguyen.Wainwright.Jordan.2010`. For any fixed
critic $T$, NWJ is never larger than DV, matching it only when $Z = e$. It is
tight at $T^* = 1 + \textrm{pmi}$: whereas DV is invariant to adding a constant
to the critic, NWJ pins that constant to $1$. Writing
$e^{-1}\, \mathbb{E}_Q[e^T] = \mathbb{E}_Q[e^{T-1}]$ shows that NWJ is simply
the general f-divergence dual bound of :numref:`sec_mdl-f-gan-dual` specialized
to the KL divergence,
$D_{\textrm{KL}}(P \| Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T-1}]$,
derived there from the convex conjugate of $u \log u$. For KL divergence,
Donsker–Varadhan strictly sharpens that general bound.

Because no logarithm wraps the second expectation, evaluating
:eqref:`eq_mdl-nwj_bound` on a minibatch yields an *unbiased* estimate of the
lower bound. Unbiased, however, does not mean reliable. At the optimal critic,
$e^{T^*-1} = e^{\textrm{pmi}}$ is the density ratio $p(x,y)/(p(x)p(y))$. Its
expectation over independent pairs is $1$, but when $X$ and $Y$ are strongly
dependent, that expectation is dominated by a tiny fraction of huge values
mixed among many values near zero. Any batch that happens to miss those rare
spikes severely underestimates the second term and therefore overshoots the
bound. We will quantify and measure this behavior in
:numref:`subsec_mdl-exact-critics`.

## InfoNCE
:label:`sec_mdl-infonce`

### The Contrastive Loss

Because the exponential term $e^T$ in the DV and NWJ bounds is unnormalized, a
single extreme pair can dominate an entire batch. InfoNCE avoids this
instability by normalizing *within* the batch, forcing the candidates to compete
against one another. Specifically, we draw one *positive* pair
$(x, y^+) \sim P_{X,Y}$, calling $x$ the *anchor*, together with $N - 1$
*negatives* $y_2, \ldots, y_N$ drawn independently from the marginal $P_Y$. We
score all $N$ candidates with a critic $T(x, y)$ and pass the scores through a
softmax to predict which candidate is the true partner. The *InfoNCE loss*
:cite:`Oord.Li.Vinyals.2018` is the resulting cross-entropy:

$$\mathcal{L}_{\textrm{NCE}} = -\mathbb{E}\left[\log \frac{e^{T(x, y^+)}}{e^{T(x, y^+)} + \sum_{j=2}^{N} e^{T(x, y_j)}}\right],$$
:eqlabel:`eq_mdl-infonce_def`

where the expectation runs over both the positive pair and the $N - 1$
negatives. The corresponding mutual information estimate subtracts this loss
from $\log N$:

$$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}.$$
:eqlabel:`eq_mdl-infonce_est`

Some papers write the denominator as an average
$\frac{1}{N}\sum_{j=1}^N e^{T(x, y_j)}$, in which case the expected log ratio
directly equals $\hat{I}_{\textrm{NCE}}$. Conceptually, $\mathcal{L}_{\textrm{NCE}}$
is simply an $N$-class cross-entropy whose classes are the other examples in the
batch rather than fixed labels. This is the training objective behind CPC,
SimCLR, and CLIP
:cite:`Oord.Li.Vinyals.2018,Chen.Kornblith.Norouzi.ea.2020,radford2021learning`,
where $x$ and $y$ are two views or two modalities of the same example and the
remaining examples in the batch serve as negatives.

![The InfoNCE task. Left: an anchor $x$ is scored by the critic against its positive partner $y^+$, drawn jointly with it, and $N-1$ negatives drawn independently from the marginal; the rays carry the scores $T(x, y_j)$. Right: a softmax over the $N$ scores must identify the positive. The resulting cross-entropy certifies $I(X;Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}$, a bound that can never exceed $\log N$.](../img/mdl-it-infonce-pos-neg.svg)
:label:`fig_mdl-infonce-pos-neg`

:numref:`fig_mdl-infonce-pos-neg` illustrates the setup. In the smallest
nontrivial case, $N = 2$, the model faces a single negative and the loss
reduces to binary cross-entropy: telling the true partner from one distractor.
Even a perfect model can then demonstrate only that it chose the right option
out of two, so the estimate is capped at $\ln 2 = 0.69$ nats.

### The InfoNCE Bound

The following proposition confirms that minimizing the InfoNCE loss certifies a
lower bound on mutual information, and shows where the $\log N$ ceiling comes
from :cite:`Poole.Oord.Alemi.ea.2019`.

**Proposition (InfoNCE bound).** *For any measurable critic $T$ and any
$N \geq 1$,*

$$I(X; Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}.$$
:eqlabel:`eq_mdl-infonce_bound`

*Moreover, the right-hand side is at most $\log N$.*

**Proof.** Label the candidates $y_1 = y^+, y_2, \ldots, y_N$ and collect them
into a single random bag $V = (Y_1, \ldots, Y_N)$, where the negatives
$Y_{2:N}$ are independent of $(X, Y_1)$. The strategy is to replace $Y$ with
the bag $V$, construct a normalized conditional distribution over bags given
$X$, and invoke the Barber–Agakov bound.

*Step 1: independent negatives add no information.* By the chain rule for
mutual information (:numref:`sec_mdl-mutual-information`),
$I(X; Y_1, Y_{2:N}) = I(X; Y_1) + I(X; Y_{2:N} \mid Y_1)$. Because $Y_{2:N}$ is
independent of $(X, Y_1)$, it remains independent of $X$ given $Y_1$. The
second term is therefore $0$, giving $I(X; V) = I(X; Y)$.

*Step 2: the softmax ratio defines a normalized model.* Here is the key
calculation. Define the rescaled softmax score

$$
\gamma(x, v) = \frac{e^{T(x, y_1)}}{\frac{1}{N}\sum_{j=1}^N e^{T(x, y_j)}},
$$

which takes values in $(0, N]$. Let $p(v)$ be the distribution of a bag of $N$
independent draws from $P_Y$, including the first slot. Because all $N$ entries
are i.i.d., permuting them leaves $p(v)$ unchanged. Consequently, all $N$
ratios $e^{T(x, Y_i)} / \left(\frac{1}{N}\sum_j e^{T(x, Y_j)}\right)$ have the
same expectation under $p(v)$, and for every fixed $x$ we have

$$
\begin{aligned}
\mathbb{E}_{p(v)}\left[\gamma(x, V)\right]
&= \frac{1}{N}\sum_{i=1}^{N} \mathbb{E}_{p(v)}\left[\frac{e^{T(x, Y_i)}}{\frac{1}{N}\sum_{j} e^{T(x, Y_j)}}\right] \\
&= \mathbb{E}_{p(v)}\left[\frac{\frac{1}{N}\sum_{i} e^{T(x, Y_i)}}{\frac{1}{N}\sum_{j} e^{T(x, Y_j)}}\right] = 1.
\end{aligned}
$$

Thus $q(v \mid x) = p(v)\, \gamma(x, v)$ is a valid conditional probability
distribution over bags.

*Step 3: apply Barber–Agakov to the bag.* Because $Y_1$ has marginal $P_Y$ and
is independent of $Y_{2:N}$, the marginal distribution of $V$ is also $p(v)$.
Applying the ratio form of :eqref:`eq_mdl-ba_bound` to $(X, V)$ with the roles
of the two variables swapped (using the symmetry of $I$) gives

$$
I(X; V) \geq \mathbb{E}_{p(x, v)}\left[\log \frac{q(v \mid x)}{p(v)}\right]
= \mathbb{E}\left[\log \gamma(X, V)\right]
= \log N - \mathcal{L}_{\textrm{NCE}}.
$$

The final equality holds because $\log \gamma$ equals $\log N$ plus the
log-softmax probability assigned to the positive. Combining this with Step 1
proves the bound. Finally, since the softmax probability of the positive cannot
exceed $1$, we have $\mathcal{L}_{\textrm{NCE}} \geq 0$, and hence the
right-hand side is at most $\log N$. $\blacksquare$

Intuitively, picking the true partner out of $N$ candidates, even with zero
error, demonstrates at most $\log N$ nats: the information contained in a
single choice among $N$ alternatives. No critic can squeeze more information
out of that task, just as a $10$-class classifier could never certify more than
$\ln 10$ nats. Increasing $N$ raises the ceiling, which is why contrastive
methods such as CPC, SimCLR, and CLIP rely on large batches. The loss is
minimized by any critic of the form $T^*(x, y) = \textrm{pmi}(x, y) + c(x)$,
where $c(x)$ is an arbitrary function of the anchor alone (Exercise
:exref:`ex_mdl-mutual-information-optimal-critic`).

### The Ceiling with an Exact Critic
:label:`subsec_mdl-infonce-ceiling`

One might naturally expect that with the exact critic $T = \textrm{pmi}$,
$\hat{I}_{\textrm{NCE}}$ tracks the true $I(X; Y)$ all the way up to $\log N$
and then flattens out at a sharp corner. Substituting $T = \textrm{pmi}$ into
:eqref:`eq_mdl-infonce_est` reveals a softer, and more sobering, behavior:

$$\hat{I}_{\textrm{NCE}} = I(X; Y) - \mathbb{E}\left[\log \frac{1}{N}\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}\right].$$

The estimate equals the true mutual information minus a slack term, which is
nonnegative by the proposition. We can estimate the size of this slack with a
quick back-of-the-envelope calculation. Inside the logarithm sits an average of
$N$ terms: the $N - 1$ negatives are independent of $x$, so each has mean $1$
(the second identity we checked at the start of the section), contributing
roughly $1$ to the average; meanwhile, the positive contributes
$e^{\textrm{pmi}(x, y^+)}/N$, which is roughly $e^I / N$ for a typical pair.
Replacing the random average by this typical value yields

$$\hat{I}_{\textrm{NCE}} \approx I - \log\left(1 + \frac{e^I}{N}\right) = -\log\left(e^{-I} + \frac{1}{N}\right).$$
:eqlabel:`eq_mdl-infonce-smooth-min`

The right-hand side is a smooth minimum of $I$ and $\log N$. It tracks $I$
closely when $e^I \ll N$, approaches $\log N$ when $e^I \gg N$, and bends well
below both in the transition region: right at $I = \log N$, it reads
$\log N - \log 2$, falling $0.69$ nats short. Our initial guess forgot that the
positive sits inside its own denominator alongside negatives that average $1$;
the positive dominates the denominator only once $e^I$ greatly exceeds $N$.

Because replacing a heavy-tailed average with its mean is a rough heuristic, we
should check :eqref:`eq_mdl-infonce-smooth-min` empirically. For $N = 128$ and
$I = 4$, it predicts $3.6$ nats even though the ceiling $\ln 128 = 4.85$ still
looks comfortably far away. Below, we score all $N \times N$ pairs in a batch
simultaneously: row $i$ pairs anchor $x_i$ with every $y_j$, so its diagonal
entry is the positive and the $N - 1$ off-diagonal entries are its negatives.
Although the rows share candidates, the bound remains valid (Exercise
:exref:`ex_mdl-mi-estimation-in-batch-negatives`). Let us sweep $I(X; Y)$ for
$N = 16$, $128$, and $1024$ (ceilings $2.77$, $4.85$, and $6.93$ nats) and
compare the empirical estimates against the truth.

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

None of the curves has a sharp corner. Each peels away from the diagonal early
and bends smoothly toward its ceiling, with all nine printed values falling
within $0.1$ nat of :eqref:`eq_mdl-infonce-smooth-min`. At $I = 1$, every batch
size lands within about $0.1$ nat of the truth. By $I = 4$, a batch of $128$
reports only $3.7$ nats, and even $N = 1024$ begins to lag slightly behind. At
$I = 7$, just past its ceiling of $6.93$, $N = 1024$ reports only $6.3$.

Two exact facts explain why this happens. First, the average inside the
logarithm is at least the positive's own term $e^{\textrm{pmi}(x, y^+)}/N$, so
the slack is at least $\mathbb{E}[\textrm{pmi}(x, y^+)] - \log N = I - \log N$,
recovering the $\log N$ ceiling. Second, the slack disappears only in the limit
$N \to \infty$ (Exercise :exref:`ex_mdl-mutual-information-optimal-critic`). The
hard ceiling $\log N$ is merely the crudest symptom of this finite-batch slack.

This smooth compression also makes the estimate difficult to invert. With
$N = 128$, a reading of $3.7$ corresponds to $4$ nats and a reading near $4.75$
corresponds to $7$ nats, squeezing every true value from $4$ nats to infinity
into the top $1.2$ nats of the scale.

Image classification puts these numbers in perspective. Suppose two views of an
image each determine its class among $1{,}000$ balanced categories. Because the
label is a function of either view, the data-processing inequality
(:numref:`subsec_mdl-dpi`) implies that the two views share at least the
entropy of the label, $\ln 1000 = 6.91$ nats. An InfoNCE loss with $N = 128$
candidates can never certify anything close to that, and even with $N = 1024$
the estimate reads only $6.3$ when the true shared information is $7$. Whether
two views retain enough information for fine-grained classification simply
cannot be read off such a number.

### Exact Critics Compared
:label:`subsec_mdl-exact-critics`

We can now compare DV, NWJ, and InfoNCE on equal footing: batch size $N = 128$,
each given the exact optimal critic $T^* = 1 + \textrm{pmi}$ (adding a constant
leaves InfoNCE unchanged). DV and NWJ draw their independent pairs from the $N$
mismatched pairs of the batch, and the heavy-tailed behavior we noted earlier
lets us predict what will happen before running a single line of code.

A typical joint pair has pmi close to $I$. Under the product distribution
$P_X \otimes P_Y$, such a pair occurs with probability roughly $e^{-I}$, since
$p(x)\,p(y) = p(x, y)\,e^{-\textrm{pmi}}$. In other words, roughly one
mismatched pair in $e^I$ looks plausible enough to pass for a genuine pair.
Each of these *rare pairs* carries a huge weight $e^{\textrm{pmi}} \approx e^I$,
and together they account for the entire expectation $\mathbb{E}[e^{\textrm{pmi}}] = 1$.
A batch of size $N$ contains roughly $N e^{-I}$ such pairs on average: about
six at $I = 3$, two at $I = 4$, and only $0.3$ at $I = 6$.

As long as a batch contains several rare pairs, its sample average of
$e^{\textrm{pmi}}$ stays near $1$ and all three bounds should agree. Once the
expected count $N e^{-I}$ drops below $1$, most batches contain no rare pairs
at all. Call such a batch *starved*. On a starved batch, the average of
$e^{\textrm{pmi}}$ is close to zero: NWJ then reports the batch mean of $T^*$
minus almost nothing (about $I + 1$, one nat too high), while DV subtracts the
logarithm of a tiny number and overshoots by many nats.

Because NWJ is unbiased, something must balance those starved batches. That
counterweight is a *hit* batch, one that happens to contain a mismatched pair
with $e^{\textrm{pmi}} \geq N$. A single such pair adds at least $1$ to the
batch mean of $e^{\textrm{pmi}}$ and drags NWJ down by at least a nat, leaving
it near $I$ or occasionally far below. At $I = 6$, a rare pair has
$e^{\textrm{pmi}} \approx e^6 = 403 > N$, so every batch that contains one is a
hit batch. With an average of $0.3$ rare pairs per batch, roughly one batch in
four should be hit, while most of the remaining batches are starved.

InfoNCE behaves completely differently. Because it normalizes within the batch,
a convincing negative affects at most one row of the softmax. It should stay
steady right where :eqref:`eq_mdl-infonce-smooth-min` predicts, around $4.6$
nats at $I = 6$. Notice that the crossover $N e^{-I} \approx 1$ occurs at
$I \approx \ln N$. The logarithm of the batch size limits DV and NWJ just as
much as it limits InfoNCE; DV and NWJ simply hide that limit behind exploding
variance instead of a visible ceiling.

Let us test these predictions by drawing $200$ batches at each level of $I$.
Because heavy-tailed distributions distort sample means and standard
deviations, we report the median and interquartile range for each bound, along
with the sample mean for NWJ. The final two columns track the fraction of
starved batches (mean $e^{\textrm{pmi}} < 0.1$) and the fraction of hit batches
($\max e^{\textrm{pmi}} \geq N$), and the plot compares the three medians.

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

Up to $I = 3$, all three medians stay within about $0.2$ nat of the truth,
almost no batch is starved or hit, and InfoNCE displays the tightest spread.
Beyond that point, starved batches appear right on schedule: at most about one
in ten at $I = 4$, roughly a third at $I = 5$, and more than half at $I = 6$.

In the $I = 6$ row, NWJ's median climbs to about $7 = I + 1$ with its upper
quartile sitting right on top of it, meaning that half of all batches overshoot
by roughly a full nat. The quarter or so of hit batches drag the sample mean
back to within about half a nat of $6$. DV fares even worse: its median sits
several nats above the truth, and its upper quartile reaches into the tens,
sending its curve right off the top of the plot. InfoNCE, by contrast, reads a
rock-steady $4.6$ nats with an interquartile range under $0.1$.

We can also calculate NWJ's spread analytically. Its mismatched-pair term is a
batch average of $e^{\textrm{pmi}}$, whose variance is $(e^{2I} - 1)/N$. At
$I = 6$ and $N = 128$, that works out to a standard deviation of $36$ nats,
driven by extreme hit batches so rare that $200$ draws rarely contain one
(Exercise :exref:`ex_mdl-mi-estimation-heavy-tails`).

None of this contradicts the variational inequalities, which hold for
population expectations. Rather, it shows how variational bounds reconcile with
the distribution-free limit of :numref:`subsec_mdl-mi-distribution-free`.
InfoNCE cannot exceed $\log N$ (with $N$ at most the sample size), so it obeys
the logarithmic limit by construction. DV and NWJ are free to report larger
numbers, but whenever they do, their batch estimates cease to be reliable
lower bounds.

### Training a Critic
:label:`subsec_mdl-infonce-learned-critic`

In real applications we do not know the true pmi, so the critic must be
learned from data. When a contrastive loss reports less than the true mutual
information, how do we tell whether the shortfall comes from an imperfect
critic or from the slack of the bound itself? To answer this, let us train a
dual-encoder critic of the exact form used by CLIP and SimCLR: two small
networks $g, h : \mathbb{R} \to \mathbb{R}^{16}$ embed $x$ and $y$, and the
critic is their inner product $T(x, y) = g(x)^\top h(y)$. Computing all
$N \times N$ scores for a batch then takes just one matrix multiplication of
the two embedding matrices, reducing $N^2$ network evaluations to $2N$. We
generate two disjoint pools of $4{,}096$ pairs at $\rho = 0.99$ (true
information $1.96$ nats), one for training and one for evaluation.

```{.python .input #mdl-mi-estimation-training-a-critic-1}
rho, n_pool = 0.99, 4096
x_tr, y_tr = (a.astype(np.float32) for a in gauss_pair(rng, rho, n_pool))
x_ev, y_ev = (a.astype(np.float32) for a in gauss_pair(rng, rho, n_pool))
```

Each encoder is a multilayer perceptron with two hidden layers of $64$ ReLU
units and Glorot-uniform initialization. On the resulting $N \times N$ score
matrix, the InfoNCE loss is simply standard multi-class cross-entropy where the
target class for row $i$ is $i$, the index of its diagonal positive pair.

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

We train for $1{,}000$ Adam steps with learning rate $10^{-3}$ on minibatches
of $N = 128$ pairs sampled from the training pool, then wrap the trained model
as `critic_np` so it accepts NumPy arrays just like `exact_scores`.

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

To separate critic error from bound slack cleanly, we evaluate both the learned
critic and the exact critic on the *same* held-out batches. Below, we partition
the evaluation pool into disjoint batches of size $N \in \{2, 8, 32, 128\}$ and
average the InfoNCE bound across them. Notice that a critic trained at $N = 128$
can be evaluated at any other batch size, because the optimal critic
$\textrm{pmi}(x, y) + c(x)$ does not depend on $N$.

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

Across every batch size, the learned critic matches the exact critic to within
a few hundredths of a nat. The remaining gap to the true $1.96$ nats therefore
belongs entirely to the bound, closing gradually as $N$ increases until both
critics reach about $1.9$ nats at $N = 128$. Look closely at $N = 8$: the
ceiling $\ln 8 = 2.08$ already sits above the true value of $1.96$, so the
shortfall of roughly $0.55$ nat is not caused by hitting the ceiling. It is the
smooth-minimum slack of the bound, which :eqref:`eq_mdl-infonce-smooth-min`
predicts at about $0.6$ nat. Of course, a scalar Gaussian pair gives the critic
an easy task; in high dimensions, critic approximation error is no longer
negligible, and without a known ground truth the two sources of error cannot be
disentangled.

Contrastive systems such as SimCLR modify the critic slightly: they normalize
the embeddings $z_x$ and $z_y$ to unit length and divide their cosine
similarity by a *temperature* $\tau$, setting
$T(x, y) = \textrm{sim}(z_x, z_y)/\tau$ :cite:`Chen.Kornblith.Norouzi.ea.2020`.
The temperature controls how sharply the softmax separates candidates, and it
introduces a second ceiling of its own. Because cosine similarity lies in
$[-1, 1]$, no two scores can differ by more than $2/\tau$, so a cosine critic
can never certify $2/\tau$ nats or more regardless of how large $N$ is
(Exercise :exref:`ex_mdl-mutual-information-temperature`): that caps the bound
below $4$ nats at $\tau = 0.5$ and below $20$ nats at $\tau = 0.1$. A large
temperature can therefore bottle up the bound well before the batch size does.

In practice, which estimator should we choose? For a pair of one- or
two-dimensional variables sharing at most a few nats, nearest-neighbor counting
is the right tool: KSG landed within about $0.1$ nat of the truth in that
regime. For high-dimensional vectors, none of the estimators in this section
can accurately measure a large mutual information, so the question shifts from
*how much* information two views share to *at least how much*. InfoNCE answers
that question reliably: train it with as many negatives as memory allows and
interpret $\log N - \mathcal{L}_{\textrm{NCE}}$ strictly as a lower bound. A
reading within a nat or so of $\log N$ tells us only that the views share at
least that much, while a low reading could simply reflect a weak critic. As for
DV and NWJ, they have no sweet spot of their own: several nats below $\log N$,
where their batch estimates remain trustworthy, InfoNCE is nearly as accurate
and far steadier; higher up, their typical minibatch overshoots.

Finally, a tighter lower bound does not automatically produce a better
representation. :citet:`Tschannen.Djolonga.Rubenstein.ea.2020` showed that
downstream task performance correlates poorly with the tightness of the mutual
information bound being maximized: looser bounds often yield stronger
representations, and architectural choices for the encoder and critic matter at
least as much as the estimated information. Two fundamental questions thus
remain open: how to estimate large mutual information reliably in high
dimensions, and why maximizing loose variational bounds still produces such
effective representations.

## Summary

From finite samples alone, mutual information can be measured directly only
when the variables are low-dimensional and the shared information is modest.
Beyond that regime, every estimator is constrained by the logarithm of the
number of outcomes it can distinguish: $\log b$ for a histogram with $b$ bins,
roughly $\log(n/m)$ for KSG nearest-neighbor counts (which saturate far below
even that ceiling in high dimensions), and $O(\log n)$ for any
distribution-free lower bound. Variational methods sidestep density estimation
by maximizing a lower bound parameterized by a decoder or a critic. InfoNCE,
the cross-entropy objective of contrastive learning, is low-variance and well
behaved, yielding roughly $-\log(e^{-I} + 1/N)$ with an exact critic and never
exceeding $\log N$ for $N$ candidates. Donsker–Varadhan and NWJ have no hard
ceiling, yet their minibatch estimates remain reliable only a couple of nats
below $\log N$. On our Gaussian benchmark, a learned dual-encoder critic
matched the exact pmi critic to within a few hundredths of a nat, confirming
that the observed gap belonged to the bound itself. Even so, the tightness of
an information bound does not by itself predict the quality of a learned
representation.

## Exercises

1. **Heavy tails of the exponentiated pmi.** ● Let $(X, Y)$ be the Gaussian
   pair with correlation $\rho$, $I = -\tfrac{1}{2}\log(1 - \rho^2)$, and
   $Q = P_X \otimes P_Y$. Under $Q$ the mean of $e^{\textrm{pmi}}$ is $1$
   for every $\rho$, although a typical independent pair has
   $e^{\textrm{pmi}}$ near $0$ when $\rho$ is close to $1$.
    1. Show that
       $\mathbb{E}_Q[e^{2\,\textrm{pmi}}] = \int p^2/q = 1/(1 - \rho^2) = e^{2I}$,
       so that $e^{\textrm{pmi}}$ has variance $e^{2I} - 1$ under $Q$.
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
   :exlabel:`ex_mdl-mi-estimation-heavy-tails`

1. **Barber–Agakov with a Gaussian decoder.** Take the standard bivariate
   Gaussian pair with correlation $\rho$, whose information is
   $-\tfrac{1}{2}\log(1 - \rho^2)$, and the decoders
   $q(x \mid y) = \mathcal{N}(x; a y, s^2)$ with parameters $a \in \mathbb{R}$
   and $s^2 > 0$. In the bound :eqref:`eq_mdl-ba_bound`, use the
   differential entropy $h(X) = \tfrac{1}{2}\log(2\pi e)$.
    1. Evaluate the right-hand side of :eqref:`eq_mdl-ba_bound` as a
       function of $a$ and $s^2$. Maximize it over both parameters and show
       that the maximum is the true information.
    1. The text fixes $s^2 = 1$, which is what training with the squared
       error alone amounts to, and finds the best bound $\rho^2/2$. Show
       that its gap to the true information is
       $D_{\textrm{KL}}\big(\mathcal{N}(0, 1 - \rho^2) \,\|\, \mathcal{N}(0, 1)\big)$,
       as the proof of the proposition says it must be. How does the gap
       behave as $\rho \to 1$?
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
       the ceiling $\log N - \log\big(1 + (N-1)e^{-2/\tau}\big)$ for $N = 4096$
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
   distribution: $P_{S \sim p^n}\big(B \leq H(p)\big) \geq 0.95$ for every
   $p$. Let $\mathcal{T}_0$ be the type of a sample of $n$ distinct values.
    1. The text applies the guarantee to the uniform distribution on $n^2$
       values. Show that any integer $M \geq n^2/1.9$ serves as well, so
       that $B(\mathcal{T}_0) \leq \log M$. By how much does this improve
       on $2 \log n$?
    1. Let $p$ be uniform on $M = 2^{256}$ values. How large must $n$ be
       before $2 \log n$ reaches $H(p)$? What does this say about measuring
       the entropy of a cryptographic key from samples?
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
       truth ($1.96$) in :numref:`subsec_mdl-infonce-learned-critic`.
    1. The approximation :eqref:`eq_mdl-infonce-smooth-min` replaces the
       negatives' share of the average by $1$ and $\textrm{pmi}(x, y^+)$ by
       $I$. Use Jensen's inequality to show that the first replacement
       lowers the estimate and the second raises it. Which of the two wins
       in the table of :numref:`subsec_mdl-infonce-ceiling`?
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
    1. A joint critic $T(x, y)$, a network applied to the concatenated
       pair, needs $N^2$ network evaluations for the scores of a batch,
       where the separable critic $g(x)^\top h(y)$ needs $2N$. Evaluate both
       counts for CLIP's batches of $N = 32{,}768$. A separable critic
       cannot represent every function of the pair. Show that the pmi of
       the Gaussian pair is separable with embeddings of three dimensions,
       and that two suffice for InfoNCE.
   :exlabel:`ex_mdl-mi-estimation-in-batch-negatives`

1. **Negative sampling as shifted PMI.** ● Skip-gram with negative
   sampling, the training method of word2vec, scores a center word $w$ and
   a context word $o$ by the inner product
   $s(w, o) = \mathbf{u}_o^\top \mathbf{v}_w$ of their vectors. It learns to
   tell the pairs that occur in a corpus from pairs whose context word is
   one of $K$ noise words drawn from a distribution $P$. Let $p(w, o)$ be
   the frequency of the pair among all (center, context) pairs of the
   corpus, with marginals $p(w)$ and $p(o)$. Averaged over the corpus, the
   loss is

   $$\ell = -\sum_{w, o} p(w, o) \log \sigma\big(s(w, o)\big) - K \sum_{w} p(w) \sum_{o} P(o) \log \sigma\big(-s(w, o)\big),$$

   where $\sigma(s) = 1/(1 + e^{-s})$ is the sigmoid. Assume that the
   embeddings have enough dimensions for every score $s(w, o)$ to be chosen
   independently.
    1. Show that $\ell$ is minimized at
       $s^*(w, o) = \log \frac{p(w, o)}{p(w)\, P(o)} - \log K$, which equals
       $\textrm{pmi}(w, o) - \log K$ when $P$ is the unigram distribution
       $p(o)$ :cite:`Levy.Goldberg.2014`.
    1. word2vec draws noise words with probability proportional to their
       count raised to the power $0.75$. Show that $s^*$ then differs from
       $\textrm{pmi}(w, o) - \log K$ by a term that depends on $o$ alone.
       For which context words is this term negative? How does it act on
       pairs of rare words, such as the "sri lanka" of Exercise
       :exref:`ex_mdl-mutual-information-rare-pairs` of
       :numref:`sec_mdl-mutual-information`?
    1. Compare $s^*$ with the optimal InfoNCE critic of Exercise
       :exref:`ex_mdl-mutual-information-optimal-critic`b. Why does the
       binary classification of negative sampling fix the additive term that
       InfoNCE leaves free?
   :exlabel:`ex_mdl-mutual-information-negative-sampling-pmi`

1. [extended] **Compression phase at small scale.** ●●
   :citet:`Shwartz-Ziv.Tishby.2017` studied fully connected tanh networks of
   widths 12-10-7-5-4-3-2 on the $4096$ binary input patterns of $12$ bits.
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
[Dive into Deep Learning · §37.7]{.kicker}

Estimating Mutual Information from Samples<br>**histograms & ranks · KSG nearest neighbors · McAllester–Stratos ceiling · DV & NWJ bounds · InfoNCE**
:::
:::

::: {.slide title="Why InfoNCE Reports at Most ln N Nats"}
[Motivation: Contrastive Learning vs. Measurement]{.kicker}

In self-supervised contrastive learning (SimCLR, CLIP, CPC), a network scores an anchor view $x$ against its positive partner $y^+$ and $N - 1$ negative distractors using the $N$-way cross-entropy loss $\mathcal{L}_{\textrm{NCE}} \ge 0$:

$$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}} \le \log N$$

- **The $\log N$ batch-size ceiling:** With $N = 128$ candidates, $\hat{I}_{\textrm{NCE}}$ can never exceed $\ln 128 = 4.85\text{ nats}$, even if the two views share $50\text{ nats}$ of information—identifying $1$ partner out of $128$ candidates demonstrates at most $\ln 128\text{ nats}$!
- **Two distinct paradigms for sample-based MI:**
  1. **Direct nonparametric counting (Histograms & KSG):** Counts points in bins or adaptive neighbor boxes; accurate in $1\text{--}2$ dimensions for small $I$, but breaks down in high dimensions.
  2. **Variational lower bounds (Barber–Agakov, DV, NWJ, InfoNCE):** Trains an auxiliary decoder or critic $T(x, y)$ to maximize a guaranteed population lower bound on $I(X; Y)$.
:::

::: {.slide title="Benchmark: The Gaussian Pair and Exp-PMI Tails"}
[Ground Truth & Heavy-Tailed Ratios]{.kicker}

To test estimators against an exact ground truth, we use the standard bivariate Gaussian pair $(X, Y)$ with correlation $\rho$, where $I(X; Y) = -\tfrac12\log(1 - \rho^2)$ and the pointwise mutual information is quadratic:

$$\textrm{pmi}(x, y) = \log\frac{p(x, y)}{p(x)p(y)} = -\frac{1}{2}\log(1 - \rho^2) - \frac{x^2 - 2\rho xy + y^2}{2(1 - \rho^2)} + \frac{x^2 + y^2}{2}$$

- **Two fundamental identities:** Under joint draws $(x, y) \sim P_{X,Y}$, $\mathbb{E}_{P_{X,Y}}[\textrm{pmi}(x, y)] = I(X; Y)$. Under independent draws $(x, y) \sim P_X \otimes P_Y$ (formed by rolling $y$ by one index), $\mathbb{E}_{P_X \otimes P_Y}[e^{\textrm{pmi}(x, y)}] = 1$, but its variance $\textrm{Var}_{P_X \otimes P_Y}(e^{\textrm{pmi}}) = e^{2I} - 1$ explodes with $I$!

@!mdl-mi-estimation-estimating-mutual-information-2
:::

::: {.slide title="Plug-In Histograms: Two Competing Biases"}
[Coarse Binning vs. Finite-Sample Noise]{.kicker}

Partitioning each axis into $b$ equal-width bins and applying discrete MI to the $b \times b$ empirical frequency table triggers two opposing errors plus a hard $\log b$ ceiling:

- **Downward data-processing bias (coarse bins):** By DPI, $I(X_{\textrm{bin}}; Y_{\textrm{bin}}) \le I(X; Y)$. When bin width ($\approx 9/b$) exceeds the conditional spread $\sqrt{1 - \rho^2}$ ($0.14$ at $\rho = 0.99$), binning destroys fine dependence.
- **Upward finite-sample bias (fine bins):** Empirical cell frequencies fluctuate randomly around independence, and since $D_{\textrm{KL}} \ge 0$, sampling noise always masquerades as positive mutual information.

@!mutual-information-gaussian-binned
:::

::: {.slide title="Coordinate Sensitivity and Rank-Based Binning"}
[Fixing Grid Distortion in 1D]{.kicker}

Although true $I(X; Y)$ is invariant to any bijection $X \mapsto \phi(X)$, cubing $X \mapsto X^3$ stretches the sample range to $\pm 90$ and packs $90\%$ of points into 2–3 equal-width bins. Binning sorted **ranks** restores exact invariance under monotone maps:

@!mdl-mutual-information-mutual-information-as-nonlinear-correlation-2

::: {.d2l-note .warn}
**Why Histograms Stop at $d = 2$:** For two $d$-dimensional vectors with $b = 10$ bins per axis, the joint grid has $b^{2d} = 10^{2d}$ cells ($10^{10}$ cells at $d = 5$), leaving almost every cell empty!
:::
:::

::: {.slide title="The KSG Adaptive Nearest-Neighbor Estimator"}
[Local Chebyshev Boxes & Strips]{.kicker}

The **Kraskov–Stögbauer–Grassberger (KSG)** estimator replaces fixed grids with local boxes adapted to sample density:

::: {.cols .vc}
::: {.col}
- **Joint Chebyshev ($\ell_\infty$) box:** Around point $i$, find distance $r_i$ to its $m$-th nearest neighbor under $\max(\|x - x_i\|_\infty, \|y - y_i\|_\infty)$, enclosing fraction $\approx m/n$ of joint mass.
- **Marginal strips at the exact same radius $r_i$:** Count points $n_x^{(i)}$ with $\|x_j - x_i\|_\infty < r_i$ and $n_y^{(i)}$ with $\|y_j - y_i\|_\infty < r_i$. Side lengths $(2r_i)$ cancel in the ratio $\frac{m/n}{(n_x/n)(n_y/n)}$!
- **Digamma bias correction ($\psi = \Gamma'/\Gamma$):**
  $$\hat{I}_{\textrm{KSG}} = \psi(m) + \psi(n) - \overline{\psi(n_x + 1) + \psi(n_y + 1)}$$
:::

::: {.col .fig}
@fig:mdl-it-ksg-box
:::
:::
:::

::: {.slide title="Implementing KSG and Proving Its Sample Ceiling"}
[Exact Digamma Upper Bound]{.kicker}

Because every point strictly inside the joint $\ell_\infty$ box of half-width $r_i$ satisfies both $\|x_j - x_i\|_\infty < r_i$ and $\|y_j - y_i\|_\infty < r_i$, both marginal strips hold at least $m - 1$ points ($n_x^{(i)}, n_y^{(i)} \ge m - 1$). Since $\psi$ is increasing:

$$\hat{I}_{\textrm{KSG}} \le \psi(m) + \psi(n) - 2\psi(m) = \psi(n) - \psi(m) \approx \log\frac{n}{m} \quad (\text{attained when } Y = X)$$

@mdl-mutual-information-the-curse-of-estimation
:::

::: {.slide title="Why KSG Breaks Down in Higher Dimensions"}
[The Curse of Dimensionality]{.kicker}

On $d$ independent copies of the $\rho = 0.9$ Gaussian pair (true $I = 0.830\,d\text{ nats}$), KSG ($n = 2{,}000, m = 5$) is accurate at $d \le 2$ but stalls below $3\text{ nats}$ for $d \ge 8$—well below its $6.09$-nat sample ceiling:

@!mdl-mi-estimation-ksg-in-dimension

- **Geometric failure of local constancy:** Capturing fraction $m/n = 0.0025$ of samples in $2d$ dimensions requires a box spanning $(m/n)^{1/(2d)}$ of each axis ($5\%$ at $d = 1$, jumping to $83\%$ at $d = 16$), smearing out the $7\%$-wide conditional band!
:::

::: {.slide title="The McAllester–Stratos Distribution-Free Limit"}
[Four-Step Collision Proof]{.kicker}

::: {.cols .vc}
::: {.col}
- **Theorem (McAllester & Stratos, 2020):** Any distribution-free estimator $B(S_n)$ satisfying $\Pr(B(S_n) \le I(X; Y)) \ge 0.95$ for all distributions is capped at $\mathcal{O}(\log n)$ ($\approx 2\log n$).
- **Birthday collision proof:** On $M = n^2$ uniform symbols ($H = 2\log n$), all $n$ draws are distinct with probability $> 1/2$, forcing any label-invariant bound to output $\le 2\log n$ on *every* collision-free sample!
:::

::: {.col .fig .big}
@fig:ch37-info-mcallester-stratos-wall
:::
:::
:::

::: {.slide title="Resolving the Deep Network Compression Debate"}
[What Binned Activations Actually Measure]{.kicker}

::: {.cols .vc}
::: {.col}
In a deterministic network, binned activations $Z_{\textrm{bin}} = f_{\textrm{bin}}(X)$ have $H(Z_{\textrm{bin}} \mid X) = 0$, so:

$$I(X; Z_{\textrm{bin}}) = H(Z_{\textrm{bin}}) - H(Z_{\textrm{bin}} \mid X) = H(Z_{\textrm{bin}})$$

- **Saturation vs. true MI:** As weights grow, $\tanh(z)$ saturates near $\pm 1$, collapsing inputs into the two extreme histogram bins and lowering $H(Z_{\textrm{bin}})$, whereas ReLU does not saturate and true unbinned $I(X; Z)$ never drops!
:::

::: {.col .fig}
@fig:ch37-info-tishby-tanh-binning
:::
:::
:::

::: {.slide title="Variational Lower Bounds: Estimation as Optimization"}
[Poole et al. Unified Framework]{.kicker}

::: {.cols .vc}
::: {.col}
- **Barber–Agakov (BA):** Uses a normalized decoder $q(x \mid y)$ to certify $I(X; Y) \ge H(X) + \mathbb{E}[\log q(x \mid y)]$.
- **Donsker–Varadhan (DV / MINE) & NWJ:** Use an unconstrained scalar critic $T(x, y)$ ($T^* = \textrm{pmi} + c$), but suffer from $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I} - 1$.
- **InfoNCE (Contrastive):** Normalizes $e^{T(x,y)}$ across $N$ in-batch candidates, trading a smooth $\log N$ ceiling for low variance.
:::

::: {.col .fig .big}
@fig:ch37-info-variational-mi-bounds
:::
:::
:::

::: {.slide title="The Barber–Agakov Variational Decoder Bound"}
[Reconstruction Certifies Information]{.kicker}

Replacing the unknown true posterior $p(x \mid y)$ in $I(X; Y) = H(X) - H(X \mid Y) = H(X) + \mathbb{E}[\log p(x \mid y)]$ with any tractable decoder $q(x \mid y)$ yields the **Barber–Agakov bound** (2003):

$$I(X; Y) \ge H(X) + \mathbb{E}_{(x, y) \sim P_{X,Y}}\big[\log q(x \mid y)\big] = \mathbb{E}_{(x, y) \sim P_{X,Y}}\!\left[\log\frac{q(x \mid y)}{p(x)}\right]$$

- **Exact slack:** Subtracting the bound from $I(X; Y)$ leaves $\mathbb{E}_{y \sim P_Y}\big[D_{\textrm{KL}}(p(\cdot \mid y) \,\|\, q(\cdot \mid y))\big] \ge 0$, which vanishes if and only if $q(x \mid y) = p(x \mid y)$.
- **Penalty of a fixed-variance MSE decoder:** On our unit-variance Gaussian pair ($h(X) = \tfrac12\log(2\pi e)$), a fixed-variance Gaussian decoder $q(x \mid y) = \mathcal{N}(x; ay, 1)$ certifies $\tfrac12 - \tfrac12\mathbb{E}[(X - aY)^2] = \rho^2/2$ at $a^* = \rho$—capturing only $0.405$ of $0.830\text{ nats}$ at $\rho = 0.9$ and never exceeding $0.5\text{ nats}$ as $\rho \to 1$!
:::

::: {.slide title="The Donsker–Varadhan (DV) and MINE Critic Bound"}
[Exponential Tilting Without H(X)]{.kicker}

::: {.cols .vc}
::: {.col}
With tilted density $\tilde{q}(x) = q(x)e^{T(x)} / Z$ ($Z = \mathbb{E}_Q[e^T]$), the DV gap equals $D_{\textrm{KL}}(P \| \tilde{Q}) \ge 0$:

$$D_{\textrm{KL}}(P \| Q) \ge \mathbb{E}_{P}[T] - \log \mathbb{E}_{Q}\big[e^T\big]$$

- **MINE specialization:** Setting $P = P_{X,Y}$ and $Q = P_X \otimes P_Y$ attains $I(X; Y)$ at $T^*(x, y) = \textrm{pmi}(x, y) + c$.
- **NWJ linearization:** Applying $\log Z \le Z/e$ gives $\mathbb{E}_P[T] - e^{-1}\mathbb{E}_Q[e^T]$ (unbiased on minibatches, tight at $T^* = 1 + \textrm{pmi}$).
:::

::: {.col .fig}
@fig:ch37-info-dv-nwj-tilting
:::
:::
:::

::: {.slide title="The NWJ Bound: Unbiased Batch Estimates"}
[Linearizing the Logarithm]{.kicker}

In Donsker–Varadhan, the outer logarithm in $-\log \mathbb{E}_Q[e^T]$ makes minibatch estimates $\overline{T} - \log \overline{e^T}$ **biased upward** (by Jensen's inequality, $\mathbb{E}[\log \overline{e^T}] < \log \mathbb{E}[e^T]$). Applying the tangent inequality $\log Z \le Z/e$ (tight at $Z = e$) linearizes the second term into the **NWJ bound** (Nguyen, Wainwright & Jordan, 2010):

$$I(X; Y) \ge \mathbb{E}_{P_{X,Y}}\big[T(x, y)\big] - e^{-1}\mathbb{E}_{P_X \otimes P_Y}\big[e^{T(x, y)}\big]$$

- **Connection to §{sec_mdl-divergences-distances} $f$-GAN duality:** Writing $e^{-1}\mathbb{E}_Q[e^T] = \mathbb{E}_Q[e^{T-1}]$ shows that NWJ is identically the Fenchel–Young critic bound $\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ for $f(u) = u\log u$ ($f^*(t) = e^{t-1}$).
- **Pinned additive constant:** Unlike DV, shifting $T$ changes $\log Z - Z/e$; NWJ attains equality $I(X; Y)$ uniquely at $T^*(x, y) = 1 + \textrm{pmi}(x, y)$ (where $Z = \mathbb{E}_Q[e^{1+\textrm{pmi}}] = e$).
- **Unbiased but heavy-tailed:** Because no $\log$ wraps the expectation, minibatch averages of NWJ are strictly unbiased, yet suffer from the explosive variance $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I} - 1$.
:::

::: {.slide title="InfoNCE: Contrastive Retrieval as an MI Bound"}
[In-Batch Softmax Classification]{.kicker}

Instead of averaging an unnormalized exponential $e^{T(x, y)}$, **InfoNCE** (van den Oord et al., 2018) normalizes $e^{T}$ *within* a bag of $N$ candidates—one positive partner $y_1 = y^+$ drawn from $p(y \mid x)$ and $N - 1$ negatives $y_{2:N}$ drawn i.i.d. from $P_Y$:

::: {.cols .vc}
::: {.col}
- **Contrastive cross-entropy loss:**
  $$\mathcal{L}_{\textrm{NCE}} = -\mathbb{E}\!\left[\log\frac{e^{T(x, y^+)}}{e^{T(x, y^+)} + \sum_{j=2}^N e^{T(x, y_j)}}\right]$$
- **Mutual information lower bound:**
  $$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}} \le I(X; Y)$$
- Self-normalized softmax keeps every batch term bounded in $(0, \log N]$!
:::

::: {.col .fig}
@fig:mdl-it-infonce-pos-neg
:::
:::
:::

::: {.slide title="Three-Step Proof of the InfoNCE Lower Bound"}
[Barber–Agakov on a Bag of N Candidates]{.kicker}

::: {.cols .vc}
::: {.col}
For a bag $V = (Y_1, \ldots, Y_N)$ with $Y_1 = Y^+$ and $Y_{2:N} \sim P_Y^{\otimes(N-1)}$:

$$I(X; Y) = I(X; V) \ge \mathbb{E}\!\left[\log\frac{q(V \mid X)}{p(V)}\right] = \log N - \mathcal{L}_{\textrm{NCE}}$$

- **Why it works:** Independent negatives add zero MI ($I(X; V) = I(X; Y_1)$), and by permutation symmetry of $p(v) = \prod_j p_Y(y_j)$, $q(v \mid x) = p(v)\frac{N e^{T(x,y_1)}}{\sum_j e^{T(x,y_j)}}$ integrates to $1$.
:::

::: {.col .fig}
@fig:ch37-info-infonce-bag-proof
:::
:::
:::

::: {.slide title="Why InfoNCE Bends Smoothly Below Its ln N Ceiling"}
[Exact Critic Slack Analysis]{.kicker}

InfoNCE is minimized by any critic $T^*(x, y) = \textrm{pmi}(x, y) + c(x)$. Substituting $T = \textrm{pmi}$ into $\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}$ reveals its exact population slack:

$$\hat{I}_{\textrm{NCE}} = I(X; Y) - \mathbb{E}\!\left[\log\!\left(\frac{1}{N}e^{\textrm{pmi}(x, y^+)} + \frac{1}{N}\sum_{j=2}^N e^{\textrm{pmi}(x, y_j)}\right)\right]$$

- **Smooth-minimum approximation:** In the denominator average, the $N - 1$ independent negatives each have expectation $\mathbb{E}[e^{\textrm{pmi}(x, y_j)}] = 1$, while a typical positive pair contributes $\frac{1}{N}e^{\textrm{pmi}(x, y^+)} \approx \frac{e^I}{N}$:
  $$\hat{I}_{\textrm{NCE}} \approx I - \log\!\left(1 + \frac{e^I}{N}\right) = -\log\!\left(e^{-I} + \frac{1}{N}\right)$$
- **Why InfoNCE bends early:** Right at $I = \log N$ (where $e^I/N = 1$), the positive and the $N-1$ negatives contribute equally to the denominator, leaving $\hat{I}_{\textrm{NCE}} \approx \log N - \ln 2$ (falling $0.69\text{ nats}$ below the ceiling)!
:::

::: {.slide title="Simulating the Smooth InfoNCE Ceiling Across N"}
[Batch Sizes N = 16, 128, 1024]{.kicker}

Sweeping true $I(X; Y) \in [0.25, 7]$ with the exact critic $T^* = \textrm{pmi}$ for $N \in \{16, 128, 1024\}$ confirms that every curve follows the smooth minimum $-\log(e^{-I} + 1/N)$ within $0.1\text{ nat}$:

@!mutual-information-log-n-ceiling
:::

::: {.slide title="Starved vs. Hit Batches in DV and NWJ Bounds"}
[Why Unnormalized Critics Explode Above ln N]{.kicker}

::: {.cols .vc}
::: {.col}
- **Rare-pair lottery under $P_X \otimes P_Y$:** A plausible pair ($\textrm{pmi} \approx I$, weight $e^I$) occurs by chance with probability $\sim e^{-I}$, so a batch of $N$ negatives sees $N e^{-I}$ hits on average.
- **Starved vs. hit batches ($I > \log N \implies N e^{-I} < 1$):** Most batches are *starved* ($\overline{e^{\textrm{pmi}}} \ll 1$), causing DV and NWJ to overshoot $I$ wildly, while rare *hit* batches explode with standard deviation $\sqrt{(e^{2I}-1)/N}$!
:::

::: {.col .fig .big}
@fig:ch37-info-starved-vs-hit-batches
:::
:::
:::

::: {.slide title="Comparing DV, NWJ, and InfoNCE on Exact Critics"}
[200-Batch Quartile Experiment]{.kicker}

Evaluating $200$ batches of size $N = 128$ with exact critics shows DV and NWJ destabilizing once $I > 3$, while InfoNCE maintains an interquartile range under $0.1\text{ nat}$:

@!mutual-information-perfect-critic-bounds[out=2]
:::

::: {.slide title="Comparing DV, NWJ, and InfoNCE on Exact Critics (cont.)"}
[200-Batch Quartile Experiment]{.kicker}

The medians and interquartile ranges behind the plot; the last two columns give the share of batches whose mismatched pairs have mean $e^{\textrm{pmi}} < 0.1$, and the share holding a mismatched pair with $e^{\textrm{pmi}} \ge N$:

@!mutual-information-perfect-critic-bounds[out=1]
:::

::: {.slide title="Training a Dual-Encoder Contrastive Critic"}
[Separable Inner-Product Architecture]{.kicker}

In practice $\textrm{pmi}(x, y)$ is unknown. Systems like SimCLR and CLIP train a **separable dual-encoder critic** $T(x, y) = g(x)^\top h(y)$ with embeddings $g, h : \mathcal{X} \to \mathbb{R}^d$, computing all $N \times N$ in-batch scores via one matrix product ($2N$ encoder passes instead of $N^2$):
:::

::: {.slide title="Training a Dual-Encoder Contrastive Critic (cont.)"}
[Separable Inner-Product Architecture]{.kicker}

@mdl-mi-estimation-training-a-critic-2
:::

::: {.slide title="Training the Dual-Encoder Critic With InfoNCE"}
[Separable Inner-Product Architecture]{.kicker}

@mutual-information-infonce-train
:::

::: {.slide title="Disentangling Critic Error from Bound Slack"}
[Learned vs. Exact Critic Across Batch Sizes]{.kicker}

Evaluating our trained dual-encoder critic alongside the exact $\textrm{pmi}$ critic on held-out batches at $\rho = 0.99$ ($I = 1.96\text{ nats}$) separates network approximation error from bound slack:

@!mdl-mi-estimation-training-a-critic-3

- **Bound slack vs. temperature ceiling:** The learned critic matches the exact critic within $0.02\text{ nats}$ at every $N$, proving that the $\approx 1.4\text{ nat}$ reading at $N = 8$ ($\ln 8 = 2.08 > 1.96$) is pure smooth-minimum slack. Moreover, for cosine similarity $T(x, y) = \mathrm{sim}(z_x, z_y)/\tau \in [-1/\tau, 1/\tau]$, the score gap is at most $2/\tau$, imposing a second ceiling $\hat{I}_{\textrm{NCE}} < 2/\tau$!
:::

::: {.slide title="Practical Guidelines for MI Estimation & Learning"}
[When to Count vs. When to Bound]{.kicker}

How should you choose and interpret a mutual information estimator in research and engineering?

- **Low dimension ($d \le 2$) and moderate dependence ($I \lesssim 4\text{ nats}$):** Use **KSG nearest neighbors** (or rank histograms in 1D) for direct, tuning-free nonparametric measurement within $\sim 0.1\text{ nat}$.
- **High-dimensional representations ($d \gg 2$):** Direct estimation is impossible due to the $\mathcal{O}(\log n)$ McAllester–Stratos barrier and exponential box growth. Use **InfoNCE** with large negative pools $N$ and interpret $\log N - \mathcal{L}_{\textrm{NCE}}$ strictly as a certified lower bound.
- **Avoid unnormalized exponential critics (DV / NWJ) when $I \gtrsim \log N - 2$:** Their apparent ability to report values above $\log N$ on minibatches is an artifact of starved-batch upward bias and $\mathcal{O}(e^I/\sqrt{N})$ variance.
- **More MI is not always a better representation (Tschannen et al., 2020):** Downstream linear-probe accuracy depends heavily on encoder inductive biases and critic temperature $\tau$, often peaking under looser MI bounds.
:::

::: {.slide title="Concept Check: KSG, Temperature, and Bound Slack"}
[Check Your Understanding]{.kicker}

Test your grasp of mutual information estimation and contrastive bounds:

1. **KSG on deterministic copies:** You run KSG with $m = 5$ neighbors on $n = 2{,}000$ samples of a continuous pair where $Y = X \sim \mathcal{N}(0, 1)$ (true $I(X; Y) = \infty$). What numerical value does KSG return and why?
2. **Cosine temperature ceiling:** In SimCLR with batch size $N = 256$ and cosine critic $T(x, y) = \mathrm{sim}(z_x, z_y)/\tau$ where $\mathrm{sim} \in [-1, 1]$, what is the maximum possible $\hat{I}_{\textrm{NCE}}$ with an unconstrained critic versus at temperature $\tau = 0.5$?
3. **Diagnosing a contrastive shortfall:** On a dataset where true $I(X; Y) = 1.96\text{ nats}$, your trained InfoNCE critic at $N = 8$ ($\ln 8 = 2.08\text{ nats}$) reports $\hat{I}_{\textrm{NCE}} = 1.40\text{ nats}$. Why does it fall $0.56\text{ nats}$ short even though $\ln 8 > 1.96$?
:::

::: {.slide title="Concept Check: KSG, Temperature, and Bound Slack (cont.)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. **KSG ceiling:** When $Y = X$, the joint Chebyshev distance equals the marginal distance, so every marginal strip holds only the $m - 1 = 4$ points inside the box, returning $\psi(2000) - \psi(5) = 6.09\text{ nats} \approx \ln(400)$.
2. **Temperature vs. batch ceiling:** An unconstrained critic is capped at $\ln 256 = 5.55\text{ nats}$. With $\mathrm{sim} \in [-1, 1]$ and $\tau = 0.5$, scores differ by at most $2/\tau = 4$, so $\mathcal{L}_{\textrm{NCE}} \ge \ln(1 + 255 e^{-4}) = 1.74$ and $\hat{I}_{\textrm{NCE}} \le 5.55 - 1.74 = 3.81 < 2/\tau = 4\text{ nats}$ (temperature binds before batch size!).
3. **Smooth-minimum slack:** Even with the exact critic $T^* = \textrm{pmi}$, the positive sits inside its own denominator alongside $N-1$ negatives averaging $1$, giving $-\ln(e^{-1.96} + 1/8) = 1.33\text{ nats}$—the shortfall is intrinsic bound slack, not critic failure.
:::
:::

::: {.slide title="Summary: Mutual Information Estimators & Bounds"}
[Key Takeaways]{.kicker}

| Estimator / Bound | Mathematical Formula | Ceiling / Variance Scaling | Best Regime & Primary Pitfall |
|:--|:---|:---|:---|
| **Histogram / Ranks** | $\sum_{i,j} \hat{p}_{ij}\log\frac{\hat{p}_{ij}}{\hat{p}_i\hat{p}_j}$ on $b \times b$ grid | $\le \log b$; $b^{2d}$ cells in $\mathbb{R}^{2d}$ | 1D rank histograms are coordinate-invariant; fails for $d > 2$ |
| **KSG Neighbors** | $\psi(m) + \psi(n) - \overline{\psi(n_x+1) + \psi(n_y+1)}$ | $\le \psi(n) - \psi(m) \approx \log(n/m)$ | Gold standard for $d \le 2$; boxes span $(m/n)^{1/(2d)}$ and fail in high-$d$ |
| **McAllester–Stratos** | Any distribution-free 95% lower bound $B(S_n)$ | $B(S_n) \le \mathcal{O}(\log n)$ ($\approx 2\log n$) | Proves high-$I$ certification is impossible without model assumptions |
| **Barber–Agakov** | $H(X) + \mathbb{E}_{P_{X,Y}}[\log q(x \mid y)]$ | Slack $\mathbb{E}_y[D_{\textrm{KL}}(p(\cdot\mid y) \| q(\cdot\mid y))]$ | Certifies $I \ge \log k - L$ for classifiers; requires known $H(X)$ |
| **DV (MINE) & NWJ** | $\mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T]$; $\mathbb{E}_P[T] - e^{-1}\mathbb{E}_Q[e^T]$ | No ceiling, but $\textrm{Var}_Q(e^{\textrm{pmi}}) = e^{2I}-1$ | Starved minibatches overshoot wildly whenever $I \gtrsim \log N$ |
| **InfoNCE** | $\log N + \mathbb{E}\big[\log\frac{e^{T(x,y^+)}}{\sum_{j=1}^N e^{T(x,y_j)}}\big]$ | $\le \min(\log N, 2/\tau)$; $\approx -\log(e^{-I}+1/N)$ | Low-variance workhorse of SimCLR/CLIP; saturates smoothly at $\log N$ |
:::

