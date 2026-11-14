# Test Functions and Transport
:label:`sec_mdl-test-functions-transport`

<!-- WRITER: this file was cut from mdl-divergences-distances.md by the code step (text moved unchanged, headings promoted one level). Opening, summary, exercises to the band, cross-page :eqref:s and slides are the worklist items of 37.4. -->

An f-divergence needs the density ratio $p/q$, which samples do not give and
which disjoint supports destroy (:numref:`sec_mdl-f-divergences`). Here we
build comparisons that ask only how differently $P$ and $Q$ *average* a class
of functions, and then how far mass has to move. As before, $P$ is the data and
$Q$ the model. The worked examples are on the line; in $\mathbb{R}^n$ a squared
difference $(x - x')^2$ becomes a squared norm and nothing else changes.

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
```

## Integral Probability Metrics and MMD
:label:`sec_mdl-ipm-mmd`

Total variation takes a supremum over events, that is, over indicator
functions. Other classes of functions give other distances, and for some of
them the supremum can be computed. For a class $\mathcal{G}$ of test functions
$g$, the **integral probability metric** :cite:`Muller.1997` is

$$
\textrm{IPM}_{\mathcal{G}}(P, Q)
= \sup_{g \in \mathcal{G}} \,\big( \mathbb{E}_{x \sim P}[g(x)] - \mathbb{E}_{x \sim Q}[g(x)] \big).
$$
:eqlabel:`eq_mdl-ipm-def`

The class $\mathcal{G}$ is a panel of auditors, and the IPM reports the largest
gap in averages that any auditor in the panel can find. The critic of the f-GAN
bound :eqref:`eq_mdl-f-gan-bound` was such a function too; there the second
expectation is of $f^*(T)$, here of $g$ itself. Three panels give three
distances. Functions bounded by $\|g\|_\infty \leq \tfrac{1}{2}$ give total
variation, with $g = \tfrac12\,\textrm{sign}(p - q)$ attaining the supremum.
The 1-Lipschitz functions give the Wasserstein distance of
:numref:`sec_mdl-optimal-transport`. The unit ball of a kernel's feature space
gives the **maximum mean discrepancy** (MMD)
:cite:`Gretton.Borgwardt.Rasch.ea.2012`.

An IPM compares expectations, which samples estimate directly, where
:eqref:`eq_mdl-f-div-def` needs a density ratio. But the panel must not be too
large. Two samples of $1{,}000$ points from the same continuous distribution
share no point, so the indicator of the first sample has mean $1$ on it and $0$
on the other: the empirical TV is $1$ whatever the distributions are. A class
of test functions must be small or smooth enough that sample averages track
population averages.

We build one from the smallest panel up. First, take linear test functions
$g(x) = w\, x$ with $|w| \leq 1$. Then
$\mathbb{E}_P g - \mathbb{E}_Q g = w\,(\mathbb{E}_P X - \mathbb{E}_Q X)$, and
the supremum is $|\mathbb{E}_P X - \mathbb{E}_Q X|$: the IPM over linear
functions is the difference of the means. It cannot see a change of spread; for
$\mathcal{N}(0, 1)$ against $\mathcal{N}(0, 4)$ it reports $0$.

Second, take a feature vector $\boldsymbol{\varphi}(x) = (x, x^2)$ and test
functions $g(x) = \mathbf{w}^\top \boldsymbol{\varphi}(x)$ with
$\|\mathbf{w}\| \leq 1$. Write
$\mathbf{m}_P = \mathbb{E}_P[\boldsymbol{\varphi}(x)]$ for the *mean feature*
of $P$. Then
$\mathbb{E}_P g - \mathbb{E}_Q g = \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q)$,
and by the Cauchy--Schwarz inequality the supremum of a linear function over
the unit ball is the norm of its coefficient vector:

$$
\textrm{MMD}(P, Q) = \sup_{\|\mathbf{w}\| \leq 1} \mathbf{w}^\top(\mathbf{m}_P - \mathbf{m}_Q)
= \|\mathbf{m}_P - \mathbf{m}_Q\|.
$$
:eqlabel:`eq_mdl-mmd-features`

MMD is the distance between the mean features, and the supremum has gone. For
$\mathcal{N}(0, 1)$ against $\mathcal{N}(0, 4)$ the mean features are $(0, 1)$
and $(0, 4)$, so $\textrm{MMD}^2 = 0^2 + 3^2 = 9$: means and second moments are
now both compared.

Third, we expand the squared norm. With $x, x'$ independent draws from $P$ and
$y, y'$ from $Q$, $\|\mathbf{m}_P - \mathbf{m}_Q\|^2 =
\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle +
\mathbb{E}\langle\boldsymbol{\varphi}(y), \boldsymbol{\varphi}(y')\rangle -
2\,\mathbb{E}\langle\boldsymbol{\varphi}(x), \boldsymbol{\varphi}(y)\rangle$.
Only inner products of features appear. We call
$k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x')\rangle$
the **kernel** and obtain

$$
\textrm{MMD}^2(P, Q)
= \mathbb{E}_{x, x' \sim P}[k(x, x')] + \mathbb{E}_{y, y' \sim Q}[k(y, y')]
- 2\, \mathbb{E}_{x \sim P, y \sim Q}[k(x, y)].
$$
:eqlabel:`eq_mdl-mmd2`

The first two terms measure similarity within each sample and the third
similarity across the two. If the samples interleave, the three terms cancel;
if they form separate clumps, the within-sample terms exceed the across-sample
term.

Fourth, we choose the kernel directly instead of the features. The RBF kernel
$k(x, x') = e^{-(x - x')^2/(2\ell^2)}$ with length scale $\ell$ factors as
$e^{-x^2/(2\ell^2)}\, e^{-x'^2/(2\ell^2)}\, e^{x x'/\ell^2}$, and the power
series of the last factor writes it as an inner product of infinitely many
features $\varphi_m(x) = x^m e^{-x^2/(2\ell^2)} / (\ell^m \sqrt{m!})$,
$m = 0, 1, 2, \ldots$ So the RBF kernel compares damped monomials of every
order, with weights that decrease quickly. A kernel for which
$\textrm{MMD}(P, Q) = 0$ only if $P = Q$ is called *characteristic*. The RBF
kernel is characteristic :cite:`Gretton.Borgwardt.Rasch.ea.2012`, so its MMD is
a metric; the linear and quadratic kernels above are not. (The functions
spanned by the features form the reproducing kernel Hilbert space of $k$
:cite:`Scholkopf.Smola.2002`; we need nothing more from it.)

For Gaussians the RBF expectations have a closed form. For
$Z \sim \mathcal{N}(m, s^2)$, a Gaussian integral gives

$$
\mathbb{E}\big[e^{-Z^2/(2\ell^2)}\big]
= \Big(1 + \frac{s^2}{\ell^2}\Big)^{-1/2} \exp\!\Big(-\frac{m^2}{2(\ell^2 + s^2)}\Big),
$$
:eqlabel:`eq_mdl-rbf-gaussian-integral`

and the difference of two independent Gaussians is Gaussian with the variances
added. For $\mathcal{N}(0, 1)$ against $\mathcal{N}(0, 4)$ at $\ell = 1$ the
three differences $x - x'$, $y - y'$ and $x - y$ have variances $2$, $8$ and
$5$, so
$\textrm{MMD}^2 = 3^{-1/2} + 9^{-1/2} - 2 \cdot 6^{-1/2} \approx 0.094$.

From samples, each expectation in :eqref:`eq_mdl-mmd2` becomes an average over
pairs. The unbiased estimator :cite:`Gretton.Borgwardt.Rasch.ea.2012` leaves
out the pairs of a point with itself, whose kernel value is $1$ and would bias
the within-sample terms upward. We apply it to $1{,}000$ draws from
$\mathcal{N}(0, 1)$ and $1{,}000$ draws from $\mathcal{N}(0, 4)$ under the
linear, quadratic and RBF kernels, whose population values are $0$, $9$ and
$0.094$, and then, with the RBF kernel, to a second sample from
$\mathcal{N}(0, 1)$ and to a sample from $\mathcal{N}(0.5, 1)$, for which
problem :exref:`ex_mdl-divergences-distances-q-mathcaln-delta`b derives the
population value $0.047$. What the mathematics does not supply is the size of
the sampling error.

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

The linear kernel reports a few hundredths at most, because the means agree.
The quadratic kernel reports about $9$, give or take $3$; its estimate involves
fourth powers and is noisy. The RBF kernel reports about $0.094$, give or take
$0.03$. On two samples from the same distribution the RBF estimate is within a
few thousandths of zero and may be slightly negative, which an unbiased
estimator of zero must sometimes be. The half-standard-deviation shift gives
about $0.047$, typically within $0.02$ of it. No critic was trained and no
density was evaluated.

Where do the two samples differ? The supremum in :eqref:`eq_mdl-mmd-features`
is attained by $\mathbf{w}^\star \propto \mathbf{m}_P - \mathbf{m}_Q$, and the
corresponding test function, the **witness**,
$g^\star(x) \propto \langle \mathbf{m}_P - \mathbf{m}_Q, \boldsymbol{\varphi}(x)\rangle = \mathbb{E}_P[k(x, X)] - \mathbb{E}_Q[k(x, Y)]$,
needs only kernel evaluations. We estimate it by sample means on a grid for the
shifted pair and plot it with the two densities. The cell also prints
$\mathbb{E}_P g - \mathbb{E}_Q g$ for the unscaled witness $g$, which is
$\textrm{MMD}^2$ once more, this time with the self-pairs included.

```{.python .input #mdl-divergences-distances-integral-probability-metrics-and-mmd-1}
g = lambda s: (rbf(s[:, None], x_p[None, :]).mean(axis=1)    # the unscaled
               - rbf(s[:, None], y_shift[None, :]).mean(axis=1))  # witness
grid = np.linspace(-4.0, 4.5, 400)
witness = g(grid)
print(f'witness changes sign at x = {grid[np.argmax(witness < 0)]:.2f}')
print(f'E_P g - E_Q g = {g(x_p).mean() - g(y_shift).mean():.4f}')
pdf = lambda t, mu: np.exp(-(t - mu) ** 2 / 2) / np.sqrt(2 * np.pi)
d2l.plot(grid, [witness / np.abs(witness).max(), pdf(grid, 0.0),
                pdf(grid, 0.5)], 'x', '',
         legend=['witness (scaled)', 'p = N(0, 1)', 'q = N(0.5, 1)'],
         figsize=(5, 3))
```

The witness is positive where $P$ has more mass and negative where $Q$ has
more, and it changes sign near the midpoint $x = 0.25$ of the two means (within
about $0.2$ of it, depending on the draw). The printed gap agrees with the
$\textrm{MMD}^2$ of the previous cell to about $0.001$; the difference is the
self-pairs. The witness plays the role of $A^\star$ in
:numref:`fig_mdl-tv-area`: it is the one test function that sees the whole gap.

Because MMD with a fixed kernel needs only samples and no training, it serves
as a kernel two-sample test and as an adversary-free training loss, as in
generative moment matching networks :cite:`Li.Swersky.Zemel.2015`. MMD-GANs
instead learn the kernel's features adversarially
:cite:`Li.Chang.Cheng.ea.2017`; :numref:`sec_gan_objectives` compares both with
the other critics. The kernel inception distance by which generative models are
commonly scored is the unbiased estimate of this subsection, applied to image
features with a cubic kernel (:eqref:`eq_gan_kid`).

## Optimal Transport and the Wasserstein Distance
:label:`sec_mdl-optimal-transport`

On disjoint supports every f-divergence is constant
(:numref:`sec_mdl-f-divergences`), so it gives a model no gradient, and a
better optimizer cannot repair that :cite:`Arjovsky.Chintala.Bottou.2017`. MMD
does move, but only while the supports are within a few length scales of each
other. Geometry distinguishes such distributions at every distance:
$\delta_{0.01}$ is near $\delta_0$ because its mass need only move $0.01$. The
**Wasserstein-1 distance**, also called the earth mover's distance, makes this
precise. A **coupling** $\gamma$ of $P$ and $Q$ is a joint distribution of
pairs $(x, y)$ whose marginals are $P$ and $Q$. It is a transport plan: it says
how much mass travels from each source $x$ to each destination $y$, as in
:numref:`fig_mdl-ot-transport-plan`. Writing $\Pi(P, Q)$ for the set of
couplings, for $P$ and $Q$ on $\mathbb{R}^n$ with finite first moments,

$$
W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}\big[ \|x - y\| \big],
$$
:eqlabel:`eq_mdl-w1-primal`

the smallest total of mass times distance over all plans. Between the point
masses $\delta_0$ and $\delta_d$ the only plan moves all mass from $0$ to $d$,
so $W_1(\delta_0, \delta_d) = |d|$: continuous and 1-Lipschitz in $d$, with
gradient $\pm 1$ for $d \neq 0$. This is the training signal that the
f-divergences lack.

![Optimal transport between two distributions on the line. Top left: the monotone transport plan moves each parcel of mass from $P$ (top) to $Q$ (bottom), ribbon width proportional to the mass moved; this plan is optimal for every convex cost, and for the cost $|x-y|$ it is one of several optimal plans. Bottom left: the shaded area between the two CDFs equals the Wasserstein-1 distance, $1.7$. Right: for two point masses at separation $d$, the Jensen--Shannon divergence (in nats) is frozen at $\log 2$ for every nonzero $d$ while the Wasserstein distance (in units of $x$) grows linearly; only transport gives a gradient. The vertical axis shows the value of each.](../img/mdl-it-ot-transport-plan.svg)
:label:`fig_mdl-ot-transport-plan`

The primal problem :eqref:`eq_mdl-w1-primal` optimizes over joint
distributions, which are hard even to parameterize from samples. Its dual does
not, and the dual is the reason $W_1$ matters to deep learning.

**Proposition (Kantorovich--Rubinstein duality).** *For distributions $P$ and
$Q$ on $\mathbb{R}^n$ (or any complete separable metric space) with finite
first moments,*

$$
W_1(P, Q) = \sup_{\|g\|_{\textrm{Lip}} \leq 1}
\big( \mathbb{E}_{x \sim P}[g(x)] - \mathbb{E}_{y \sim Q}[g(y)] \big),
$$
:eqlabel:`eq_mdl-kr-dual`

*where the supremum runs over all 1-Lipschitz functions $g$, those with
$|g(x) - g(y)| \leq \|x - y\|$ for all $x, y$.*

An infimum over joint distributions has become a supremum over one function:
$W_1$ is the IPM :eqref:`eq_mdl-ipm-def` over test functions with slope at most
$1$, and a network can represent such a function. The **WGAN**
:cite:`Arjovsky.Chintala.Bottou.2017` trains this dual, with a neural critic in
the role of $g$ that is constrained to be approximately 1-Lipschitz.
:numref:`sec_gan_objectives` trains such a critic, and
:numref:`sec_gan_convergence` treats the gradient penalties that enforce the
constraint. What an approximately Lipschitz critic reports is no longer exactly
$W_1$.

We prove the duality only on the line, where the proof also yields a closed
form and transport needs no optimization at all; :citet:`Peyre.Cuturi.2019`
give the general proof. The idea is to count crossings. At a point $t$, $P$ has
a fraction $F_P(t)$ of its mass to the left of $t$ and $Q$ has $F_Q(t)$, so any
plan must carry at least $|F_P(t) - F_Q(t)|$ of mass across $t$. A parcel moved
from $x$ to $y$ crosses every $t$ between them, so its distance $|x - y|$ is
the length of the set of points it crosses, and the total cost is the mass
crossing each $t$, integrated over $t$.

**Proposition ($W_1$ on the line).** *For distributions $P$ and $Q$ on
$\mathbb{R}$ with CDFs $F_P$, $F_Q$ and finite first moments,*

$$
W_1(P, Q) = \int_{-\infty}^{\infty} \lvert F_P(t) - F_Q(t) \rvert \, dt
= \sup_{\|g\|_{\textrm{Lip}} \leq 1}
\big( \mathbb{E}_P[g] - \mathbb{E}_Q[g] \big),
$$
:eqlabel:`eq_mdl-w1-cdf`

*so Kantorovich--Rubinstein duality holds on the line.*

**Proof.** *Step 1: every plan costs at least the area.* For
$(X, Y) \sim \gamma$,
$|X - Y| = \int \mathbf{1}(t \textrm{ lies between } X \textrm{ and } Y)\, dt$.
The integrand is non-negative, so expectation and integral may be exchanged:

$$
\mathbb{E}_\gamma|X - Y|
= \int \Big( \gamma(X \leq t < Y) + \gamma(Y \leq t < X) \Big)\, dt
\geq \int \big| \gamma(X \leq t < Y) - \gamma(Y \leq t < X) \big|\, dt.
$$

The step that is not routine is the difference inside the last integral. Add
and subtract $\gamma(X \leq t, Y \leq t)$: then
$\gamma(X \leq t < Y) - \gamma(Y \leq t < X) = \gamma(X \leq t) - \gamma(Y \leq t) = F_P(t) - F_Q(t)$,
because the marginals of $\gamma$ are $P$ and $Q$. So every plan costs at least
$\int |F_P - F_Q|$.

*Step 2: one plan costs exactly the area.* Let $U$ be uniform on $(0, 1)$ and
use the *monotone coupling* $X = F_P^{-1}(U)$, $Y = F_Q^{-1}(U)$, with the
quantile function $F^{-1}(\lambda) = \inf\{x : F(x) \geq \lambda\}$ for
$\lambda \in (0, 1)$, which satisfies $F^{-1}(\lambda) \leq t$ exactly when
$\lambda \leq F(t)$. Then
$\gamma(X \leq t < Y) = \Pr\big(F_Q(t) < U \leq F_P(t)\big)$ and
$\gamma(Y \leq t < X) = \Pr\big(F_P(t) < U \leq F_Q(t)\big)$. At least one of
the two is zero, so the inequality of Step 1 is an equality for this plan.
Hence the primal $W_1$ equals the area.

*Step 3: no test function beats any plan.* This is *weak duality*, and it holds
in any dimension. For a 1-Lipschitz $g$ and a coupling $\gamma$,
$g(x) - g(y) \leq \|x - y\|$ at every pair, and taking the expectation under
$\gamma$ gives
$\mathbb{E}_P[g] - \mathbb{E}_Q[g] \leq \mathbb{E}_\gamma\|x - y\|$ (compare
the weak duality of :numref:`subsec_mdl-dual-problem`).

*Step 4: one test function attains the area.* It suffices to consider
$g(x) = g(0) + \int_0^x g'(t)\, dt$ with $|g'| \leq 1$: every such $g$ is
1-Lipschitz, and Step 3 already bounds all the others. The constant $g(0)$
cancels in $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$. Writing
$\int_0^X g'(t)\,dt = \int_0^{\infty} g'(t)\, \mathbf{1}(X > t)\, dt - \int_{-\infty}^0 g'(t)\, \mathbf{1}(X \leq t)\, dt$
and swapping expectation and integral (Fubini's theorem from
:numref:`sec_mdl-integral_calculus`, which the finite first moments justify),

$$
\mathbb{E}_P[g] - \mathbb{E}_Q[g] = \int_{-\infty}^{\infty} g'(t)\, \big( F_Q(t) - F_P(t) \big)\, dt.
$$

The choice $g'(t) = \textrm{sign}\big(F_Q(t) - F_P(t)\big)$ makes the right
side equal to the area. Together, the dual is at least the area (Step 4), at
most the primal (Step 3), and the primal equals the area (Steps 1 and 2).
$\blacksquare$

On the line, then, $W_1$ is the area between the two CDFs, the shaded region of
:numref:`fig_mdl-ot-transport-plan`, and for two samples of equal size it is
the mean absolute difference of the sorted values (problem
:exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`a). It is rather
surprising that a linear program should collapse this far. We watch it happen
once, for two distributions on the six points $0, 1, \ldots, 5$, with $P$
mostly on the left and $Q$ mostly on the right: by the CDF formula, and by
solving the primal :eqref:`eq_mdl-w1-primal` as a linear program over plans (a
plan is a $6 \times 6$ matrix with row sums $p$ and column sums $q$).

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
print(f'W1 via the CDF formula : {w1_cdf:.10f}')
print(f'W1 via the primal LP   : {res.fun:.10f}')
```

Both routes give $W_1 = 1.7$: one sum over five gaps replaces a $36$-variable
linear program.

The optimal plan is not unique here. $F_P \geq F_Q$ at every atom, so every
plan that moves mass only rightward costs
$\sum_j x_j q_j - \sum_i x_i p_i = 3.35 - 1.65 = 1.7$, and the linear program
returns one of them, not necessarily the monotone plan of the figure. A
strictly convex cost removes the ambiguity. The *Wasserstein-$p$ distance*
replaces the cost by $\|x - y\|^p$ and takes the $p$-th root,
$W_p(P, Q) = \big(\inf_{\gamma} \mathbb{E}_\gamma \|x - y\|^p\big)^{1/p}$, and
for $p > 1$ on the line the monotone plan is the unique optimum (problem
:exref:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`d). The case $p = 2$ is
the one that papers on generative models report: the Fréchet inception distance
is the squared $W_2$ between two Gaussians fitted to image features
(:eqref:`eq_gan_fid`).

## Entropic Regularization and Sinkhorn Iterations
:label:`subsec_mdl-sinkhorn`

In more than one dimension there is in general no CDF formula, and the linear
program has $N^2$ variables for $N$ points on each side. The standard remedy is
**entropic regularization** :cite:`Cuturi.2013`: we add the term
$\varepsilon \sum_{ij} \gamma_{ij} (\log \gamma_{ij} - 1)$ to the transport
cost $\sum_{ij} \gamma_{ij} C_{ij}$, for a cost matrix $\mathbf{C}$ and a
regularizer $\varepsilon > 0$. On plans, $\sum_{ij}\gamma_{ij} = 1$, so the
added term is $-\varepsilon$ times the entropy of $\gamma$ plus a constant; it
penalizes plans that concentrate on few routes. With it, transport became
usable as a loss in deep learning: Sinkhorn divergences are trained on directly
:cite:`Genevay.Peyre.Cuturi.2018`, and the minibatch couplings of
:numref:`sec_mdl-flow-matching-transport` solve small transport problems of
this kind on every batch. The subsection can be skipped on a first reading.

The regularized problem can be solved almost in closed form. Its Lagrangian, as
in :numref:`subsec_mdl-dual-problem`, adds a multiplier $\xi_i$ for each
row-sum constraint $\sum_j \gamma_{ij} = p_i$ and $\zeta_j$ for each column-sum
constraint $\sum_i \gamma_{ij} = q_j$. Setting its derivative in $\gamma_{ij}$
to zero gives $C_{ij} + \varepsilon \log \gamma_{ij} - \xi_i - \zeta_j = 0$, so
$\gamma_{ij} = e^{\xi_i/\varepsilon}\, e^{-C_{ij}/\varepsilon}\, e^{\zeta_j/\varepsilon}$.
In matrix form,

$$
\boldsymbol{\gamma}_\varepsilon = \textrm{diag}(\mathbf{a})\, \mathbf{K}\, \textrm{diag}(\mathbf{b}),
\qquad \mathbf{K} = e^{-\mathbf{C}/\varepsilon} \textrm{ (entrywise)},
$$
:eqlabel:`eq_mdl-sinkhorn-form`

with positive scalings $\mathbf{a} = e^{\boldsymbol{\xi}/\varepsilon}$ and
$\mathbf{b} = e^{\boldsymbol{\zeta}/\varepsilon}$. The regularized plan is the
*Gibbs kernel* $\mathbf{K}$, rescaled row by row and column by column. Only the
scalings are unknown, and each half of the constraints determines one of them.
With $\mathbf{b}$ fixed, $\mathbf{a} = \mathbf{p} / (\mathbf{K}\mathbf{b})$
(entrywise) makes the row sums equal to $\mathbf{p}$. With $\mathbf{a}$ fixed,
$\mathbf{b} = \mathbf{q} / (\mathbf{K}^\top \mathbf{a})$ makes the column sums
equal to $\mathbf{q}$.

The **Sinkhorn iterations** alternate these two updates. Each iteration costs
two matrix-vector products, so the iterations run well on a GPU and can be
differentiated through. They converge linearly, at a rate that worsens as
$\varepsilon \to 0$. For small $\varepsilon$ the entries of $\mathbf{K}$
underflow, so practical code iterates on $\log \mathbf{a}$ and
$\log \mathbf{b}$ (problem
:exref:`ex_mdl-divergences-distances-sinkhorn-divergences`b). We report the
*entropic cost*, the transport cost
$\sum_{ij} (\gamma_\varepsilon)_{ij} C_{ij}$ of the regularized plan without
the entropy term. As $\varepsilon \to 0$ it approaches the unregularized
optimum :cite:`Peyre.Cuturi.2019`.

How small must $\varepsilon$ be? We take the squared cost
$C_{ij} = (x_i - x_j)^2$ on the six-point example, whose optimal value is
$W_2^2$ and whose optimal plan is unique, solve the linear program, run
Sinkhorn at three values of $\varepsilon$, and count the routes that carry more
than $10^{-3}$ of mass.

```{.python .input #divergences-distances-sinkhorn}
def sinkhorn(p, q, C, eps, iters=5000):
    """Entropic OT by Sinkhorn iterations; returns the plan and its cost."""
    K = np.exp(-C / eps)                         # the Gibbs kernel
    a = np.ones_like(p)
    for _ in range(iters):
        b = q / (K.T @ a)                        # match the column sums
        a = p / (K @ b)                          # match the row sums
    plan = a[:, None] * K * b[None, :]
    return plan, np.sum(plan * C)

C2 = (atoms[:, None] - atoms[None, :]) ** 2      # squared cost (x_i - y_j)^2
res2 = optimize.linprog(C2.ravel(), A_eq=A_eq,
                        b_eq=np.concatenate([p_w, q_w]), method='highs')
print(f'squared cost, LP optimum : {res2.fun:.4f}')
plans, eps_values = [], (2.0, 0.5, 0.1)
for eps in eps_values:
    plan, cost = sinkhorn(p_w, q_w, C2, eps)
    plans.append(plan)
    print(f'epsilon = {eps}: entropic cost = {cost:.4f}, '
          f'routes above 1e-3 = {(plan > 1e-3).sum()}')
d2l.show_heatmaps(d2l.tensor(np.stack(plans)[None]), 'destination j',
                  'source i', titles=[f'epsilon = {e}' for e in eps_values],
                  figsize=(9, 2.8), cmap='Blues')
```

The linear program gives $W_2^2 = 3.5$. At $\varepsilon = 2$ the plan spreads
each source's mass over several destinations ($25$ of the $36$ routes carry
more than $10^{-3}$), and its cost, $4.06$, is well above the optimum. At
$\varepsilon = 0.5$ the plan has $14$ routes and costs $3.57$. At
$\varepsilon = 0.1$ only the ten routes of the monotone plan remain, and the
cost, $3.4995$, is within $0.001$ of the linear program: an $\varepsilon$ of
about $3\%$ of the optimal cost suffices here.

The squared cost has one more use. $W_2$ has a dynamical form, as the least
kinetic energy needed to move one distribution into the other, and
:numref:`sec_mdl-flow-matching-transport` builds generative models on that
form.

## Summary

Test-function comparisons look at averages. MMD is the distance between mean
features, computed from samples in closed form. Transport looks at geometry:
$W_1$ keeps a gradient on disjoint supports, and its dual is a supremum over
1-Lipschitz critics.

## Exercises

1. [code] **MMD in closed form, and what data processing does not
   preserve.** Use the Gaussian integral
   :eqref:`eq_mdl-rbf-gaussian-integral` for the RBF kernel with length
   scale $\ell$.
    1. Apply the map $x \mapsto 2x$ to both $P$ and $Q$. Show from
       :eqref:`eq_mdl-f-div-def` that no f-divergence changes under an
       invertible map, and that $W_1$ doubles. Show that the RBF
       $\textrm{MMD}^2$ at $\ell = 1$ falls from $0.047$ to $0.036$ for
       $\mathcal{N}(0, 1)$ against $\mathcal{N}(0.5, 1)$, but rises from
       $0.235$ to $0.787$ for $\delta_0$ against $\delta_{0.5}$. Why does the
       data-processing proposition :eqref:`eq_mdl-f-div-dpi` say nothing
       about MMD and $W_1$?
    1. For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(\delta, 1)$, show
       that $\textrm{MMD}^2 = \frac{2\ell}{\sqrt{\ell^2 + 2}}
       \big(1 - e^{-\delta^2/(2(\ell^2 + 2))}\big)$. Which length scale
       maximizes $\textrm{MMD}^2$ for small shifts $\delta$? What happens
       as $\ell \to 0$ and as $\ell \to \infty$?
    1. Evaluate the formula of (b) at $\ell = 1$ and $\delta = 0.5$.
       Recompute `mmd2_unbiased` with the RBF kernel on $200$ fresh pairs of
       samples drawn like `x_p` and `y_shift`, at $n = 250$ and at
       $n = 1000$, and compare the mean and the standard deviation of the
       estimates with the closed form. How does the standard deviation scale
       with $n$? Is the value that your run of the MMD experiment printed
       within the spread?
   :exlabel:`ex_mdl-divergences-distances-q-mathcaln-delta`

1. [code] **One-dimensional transport.** ● On the real line, $W_1$ is the
   area between the CDFs :eqref:`eq_mdl-w1-cdf`.
    1. For two empirical distributions on samples $x_1, \ldots, x_n$ and
       $y_1, \ldots, y_n$ of equal size, use the monotone coupling of the
       proof to show that $W_1 = \tfrac{1}{n} \sum_i |x_{(i)} - y_{(i)}|$,
       the mean absolute difference of the sorted values. Check this against
       the linear program of the six-point example on two
       random samples of size $7$.
    1. Compute the sorted-sample $W_1$ between `x_p` and `y_shift` and
       between `x_p` and `y_same` of the MMD experiment.
       Then average it over $20$ pairs of independent samples of size $n$
       from $\mathcal{N}(0, 1)$, for $n \in \{10^2, 10^3, 10^4\}$, and find
       the rate at which it decreases. The estimate is never negative. Why
       does this rule out unbiasedness when $P = Q$?
    1. Change `q_w` to $(0.40, 0.05, 0.05, 0.10, 0.10, 0.30)$, so that the
       two CDFs cross, and rerun the cell. Build the test function $g$ with
       $g(0) = 0$ and slope $\textrm{sign}\big(F_Q(t) - F_P(t)\big)$
       between neighboring atoms, as in Step 4 of the proof, and check that
       $\mathbb{E}_P[g] - \mathbb{E}_Q[g]$ equals the LP value. For every
       route $(i, j)$ that carries mass in the LP solution `res.x`, compare
       $g(x_i) - g(x_j)$ with $|x_i - x_j|$. Why must the two agree on every
       such route for any optimal plan and any optimal test function?
    1. *Non-uniqueness.* With the original `p_w` and `q_w`, show that
       $F_P \geq F_Q$ at every atom and that every plan that moves mass only
       rightward costs $\sum_j x_j q_j - \sum_i x_i p_i = 1.7$. Run
       `sinkhorn` with the absolute cost `C` at $\varepsilon = 0.1$ and
       $0.02$. How many routes carry mass? Show that the limit plan has the
       form $\gamma_{ij} = a_i b_j$ for $j \geq i$ and $0$ otherwise, and
       explain why it is the plan of largest entropy among the optimal plans
       (use part (a) of problem
       :exref:`ex_mdl-divergences-distances-sinkhorn-divergences`). Why is
       the optimal plan unique for the squared cost?
   :exlabel:`ex_mdl-divergences-distances-eqref-eq-mdl-w1`

1. [code] **Sinkhorn divergences.** ●● Regularize the transport LP of the
   six-point example by the divergence from the independent
   coupling,

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
       unless $P$ is a point mass. With $C_{ij} = |x_i - x_j|$, which
       distribution $P$ on the atoms minimizes
       $\textrm{OT}_\varepsilon(P, Q)$ for large $\varepsilon$?
    1. The Sinkhorn divergence
       $S_\varepsilon(P, Q) = \textrm{OT}_\varepsilon(P, Q)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(P, P)
       - \tfrac{1}{2} \textrm{OT}_\varepsilon(Q, Q)$ removes this bias
       :cite:`Genevay.Peyre.Cuturi.2018,Feydy.Sejourne.Vialard.ea.2019`.
       Compute $S_\varepsilon(P, Q)$ for `p_w` and `q_w` with the absolute
       cost `C` at $\varepsilon \in \{10^{-2}, 10^{-1}, \ldots, 10^3\}$. For
       small $\varepsilon$, carry out the Sinkhorn updates on $\log \mathbf{a}$
       and $\log \mathbf{b}$ with `scipy.special.logsumexp`. Check that
       $S_\varepsilon$ approaches $W_1$ as $\varepsilon \to 0$ and
       $\tfrac{1}{2} \mathcal{E}(P, Q)$ as $\varepsilon \to \infty$, where
       $\mathcal{E}(P, Q) = 2\mathbb{E}|X - Y| - \mathbb{E}|X - X'| - \mathbb{E}|Y - Y'|$
       is the energy distance :cite:`Szekely.Rizzo.2013`, the expression
       :eqref:`eq_mdl-mmd2` with $-|x - y|$ in place of the kernel. Derive
       the second limit from (a).
    1. Discretize the mixture $P$ of :numref:`sec_mdl-fwd-vs-rev-kl`
       on $61$ equally spaced points of $[-6, 6]$, and model it by a
       Gaussian $Q_{\boldsymbol{\theta}}$ discretized on the same points,
       with log-weights normalized by `logsumexp`. At $\varepsilon = 1$,
       minimize $\textrm{OT}_\varepsilon(Q_{\boldsymbol{\theta}}, P)$ and
       $S_\varepsilon(Q_{\boldsymbol{\theta}}, P)$ over
       $\boldsymbol{\theta} = (\mu, \log\sigma)$, and compare both fits with
       the Gaussian that minimizes $W_1$ by :eqref:`eq_mdl-w1-cdf`. How does
       the bias of (a) show in the fitted $\sigma$?
   :exlabel:`ex_mdl-divergences-distances-sinkhorn-divergences`

<!-- Discussions: forum topic to be created at deploy -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.4]{.kicker}

Test Functions and Transport<br>**averages and moving mass see what ratios cannot**
:::
:::

::: {.slide title="MMD Is the Distance Between Mean Features"}
[Integral probability metrics]{.kicker}

::: {.cols .vc}
::: {.col}
- $\sup_{\|\mathbf w\|\le1}\mathbb{E}_P[\mathbf w^\top\varphi]-\mathbb{E}_Q[\mathbf w^\top\varphi]=\|\mathbf m_P-\mathbf m_Q\|$
- $\textrm{MMD}^2=\mathbb{E}k(x,x')+\mathbb{E}k(y,y')-2\mathbb{E}k(x,y)$
- $\mathcal N(0,1)$ vs $\mathcal N(0,4)$: features $x$: $0$; $(x,x^2)$: $9$; RBF: $0.094$.
- The witness $\mathbb{E}_Pk(x,X)-\mathbb{E}_Qk(x,Y)$ shows where:
:::

::: {.col}
@!mdl-divergences-distances-integral-probability-metrics-and-mmd-1
:::
:::
:::

::: {.slide title="Transport Sees Distance: W₁ Is the Cheapest Mass × Distance"}
[Optimal transport]{.kicker}

::: {.cols .vc}
::: {.col}
- Coupling $\gamma$: a joint law with marginals $P$, $Q$.
- $W_1=\inf_\gamma\mathbb{E}_\gamma|x-y|$; $W_1(\delta_0,\delta_d)=d$, gradient $\pm1$.
- Dual: $W_1=\sup_{\|g\|_{\mathrm{Lip}}\le1}\mathbb{E}_P g-\mathbb{E}_Q g$, the WGAN critic.
:::

::: {.col .fig}
@fig:mdl-it-ot-transport-plan
:::
:::
:::

::: {.slide title="Check: Two Point Masses 3 Apart"}
[Check]{.kicker}

$P=\delta_0$, $Q=\delta_3$. What are KL, JS, TV and $W_1$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\infty$, $\log 2$, $1$, $3$. The RBF $\textrm{MMD}^2$ at $\ell=1$ is
$2-2e^{-4.5}\approx1.98$: it still moves, but barely, three length scales out.
:::
:::
:::

::: {.slide title="On the Line, W₁ Is the Area Between the CDFs"}
[Wasserstein on the line]{.kicker}

- **I.** At each $t$ any plan carries at least $|F_P(t)-F_Q(t)|$ of mass across $t$.
- **II.** A parcel moved $x\to y$ crosses every $t$ between: cost $=\int(\text{mass crossing } t)\,dt$.
- **III.** The monotone plan $F_P^{-1}(U)\to F_Q^{-1}(U)$ never crosses back, so it attains it.

$$W_1=\int|F_P(t)-F_Q(t)|\,dt$$

Six points $0,\dots,5$, $P$ heavy on the left, $Q$ on the right; CDF area against a 36-variable LP:

@!divergences-distances-w1
:::

::: {.slide title="Sinkhorn Sharpens Toward the Monotone Plan as ε → 0"}
[Entropic transport]{.kicker}

$\gamma_\varepsilon=\mathrm{diag}(\mathbf a)\,e^{-\mathbf C/\varepsilon}\,\mathrm{diag}(\mathbf b)$: alternately match rows and columns. Squared cost:

@!divergences-distances-sinkhorn
:::
