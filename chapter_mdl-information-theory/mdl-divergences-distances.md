# f-Divergences
:label:`sec_mdl-divergences-distances`

:numref:`sec_mdl-information_theory` ended with a difficulty: the KL divergence
becomes infinite as soon as a model gives zero probability to an outcome that
occurs. A classifier rarely meets it, because a softmax never outputs an exact
zero. A generative network that turns a low-dimensional noise vector into an
image meets it often. Its samples lie on a thin surface in image space that can
miss the data altogether. This section asks how else two distributions can be
compared, and which comparisons notice when a model is close but not yet right.

The smallest case already shows what can go wrong. Let the data sit at a single
point, $P = \delta_0$, and let the model's samples sit at a single point a
distance $d$ away, $Q = \delta_d$. A model at $d = 0.01$ is nearly right and a
model at $d = 100$ is far off, so a useful loss should tell them apart.
:numref:`tab_mdl-point-mass-table` shows what five common comparisons report.
Most of them are defined later, and for now only the pattern matters.

:Five comparisons of the point masses $\delta_0$ and $\delta_d$. The MMD row uses the RBF kernel with length scale $1$.
:label:`tab_mdl-point-mass-table`

| comparison | $d = 0.01$ | $d = 1$ | $d = 100$ |
|:----|----:|----:|----:|
| KL divergence $D_{\textrm{KL}}(P\|Q)$ | $\infty$ | $\infty$ | $\infty$ |
| Jensen--Shannon divergence | $\log 2$ | $\log 2$ | $\log 2$ |
| total variation distance | $1$ | $1$ | $1$ |
| maximum mean discrepancy $\textrm{MMD}^2$ | $0.0001$ | $0.79$ | $2.00$ |
| Wasserstein distance $W_1$ | $0.01$ | $1$ | $100$ |

The KL divergence is infinite in every column. The next two rows are bounded
relatives of KL, and the last two are of a different kind. Three of the five
rows do not change with $d$ at all. A model trained on any of these three would
receive the same loss at $d = 0.01$ as at $d = 100$, hence no gradient in $d$.
This is the situation of the original generative adversarial network (GAN),
whose loss is the Jensen--Shannon row up to constants
(:numref:`sec_mdl-f-gan-dual`).

The point masses are extreme, but the moral is general. To compare two
distributions we must choose *what* to compare, and that choice fixes what we
must be able to compute and what the comparison can detect. There are four
choices. Ratio methods compare the two densities point by point through $p/q$,
and they need both densities. Test-function methods ask how differently the two
distributions average a class of functions, and they need only samples.
Transport methods measure the cost of the cheapest way to move one
distribution's mass onto the other, and they see the geometry of the space.
Score methods compare the slopes of $\log p$ and $\log q$, and they never see a
normalizing constant.

KL is the first ratio method, and minimizing it is maximum likelihood. Of the
results on the other families, two carry most of the weight, because
adversarial training rests on them: the variational bound that estimates a
ratio-based divergence from samples (:numref:`sec_mdl-f-gan-dual`) and the
Kantorovich--Rubinstein duality for $W_1$
(:numref:`sec_mdl-optimal-transport`). Pinsker's inequality
(:numref:`sec_mdl-tv-pinsker`) and the data-processing inequality
(:numref:`subsec_mdl-f-div-data-processing`) are the two that later proofs
reach for. The $\alpha$-divergences, the Sinkhorn iterations and the kernel
Stein discrepancy can be skipped on a first reading.

All logarithms are natural, as in :numref:`sec_mdl-information_theory`: KL,
reverse KL and Jensen--Shannon are in nats, the other ratio-based divergences
have no unit, and transport distances are in the units of $x$. Throughout, $P$
is the data or the target and $Q$ is the model; $p$ and $q$ are their densities
or probability mass functions, and $u = p/q$ is their *density ratio*. A sample
has $N$ points, and $\mathbf{x}$ has $n$ coordinates.

```{.python .input #divergences-distances-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #divergences-distances-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #divergences-distances-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #divergences-distances-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-divergences-distances-divergences-and-distances-between-distributions-2}
from scipy import integrate, optimize
```

The table is computed, not quoted: the cell treats $\delta_0$ and $\delta_d$ as
probability vectors $(1, 0)$ and $(0, 1)$ on the two points $\{0, d\}$ and
evaluates each comparison from its definition. For now, read only the output.

```{.python .input #mdl-divergences-distances-divergences-and-distances-between-distributions-1}
def kl_div(p, q):
    """D_KL(P||Q) on a finite support: 0 log 0 = 0, and inf if q = 0 < p."""
    s = p > 0
    with np.errstate(divide='ignore'):
        return np.sum(p[s] * np.log(p[s] / q[s]))

def five_comparisons(atoms, p, q, ell=1.0):
    """KL, JS, TV, RBF MMD^2 and W1 between pmfs p, q on sorted atoms."""
    m = (p + q) / 2                                   # the mixture for JS
    K = np.exp(-(atoms[:, None] - atoms[None, :]) ** 2 / (2 * ell ** 2))
    F_gap = np.abs(np.cumsum(p) - np.cumsum(q))[:-1]  # |F_P - F_Q| between atoms
    return {'KL': kl_div(p, q), 'JS': (kl_div(p, m) + kl_div(q, m)) / 2,
            'TV': np.abs(p - q).sum() / 2, 'MMD^2': (p - q) @ K @ (p - q),
            'W1': np.sum(F_gap * np.diff(atoms))}

delta_0, delta_d = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # on atoms (0, d)
for d in (0.01, 1.0, 100.0):
    vals = five_comparisons(np.array([0.0, d]), delta_0, delta_d)
    print(f'd = {d:6}: ' + '  '.join(f'{k} {v:.4g}' for k, v in vals.items()))
```

Three comparisons are the same for every $d \neq 0$, and two move with $d$. We
begin with the three that do not move.

## f-Divergences

KL compares $P$ and $Q$ through the ratio $p/q$. Here we keep the ratio and
change the function applied to it. The result is a family with one proof of
non-negativity and one variational form that can be estimated from samples.

### Divergences and Metrics
:label:`sec_mdl-what-is-a-divergence`

First, the least that a "distance between distributions" must satisfy. A
**divergence** is a function of two distributions that is non-negative and
vanishes only when they agree:

$$
D(P, Q) \geq 0, \qquad D(P, Q) = 0 \quad \textrm{if and only if} \quad P = Q.
$$

Nothing else is required. A **metric**, like the norms of
:numref:`sec_linear-algebra`, must in addition be symmetric,
$D(P, Q) = D(Q, P)$, and satisfy the triangle inequality,
$D(P, R) \leq D(P, Q) + D(Q, R)$. The KL divergence is a divergence and fails
both extra conditions. Its value between two one-dimensional Gaussians
(:eqref:`eq_mdl-gaussian_kl`) shows the failures with numbers:

$$
D_{\textrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2, \sigma_2^2)\big)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-dd-gaussian-kl`

For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(0, 4)$ the formula gives
$D_{\textrm{KL}}(P\|Q) = \log 2 + \tfrac18 - \tfrac12 \approx 0.318$ nats but
$D_{\textrm{KL}}(Q\|P) = -\log 2 + 2 - \tfrac12 \approx 0.807$ nats, so KL is
not symmetric. The triangle inequality fails for a different reason. For
unit-variance Gaussians centered at $0$, $1$ and $2$ the formula gives
$D_{\textrm{KL}} = \tfrac{1}{2}$ for each adjacent pair but $2$ for the outer
pair, and $2 > \tfrac{1}{2} + \tfrac{1}{2}$ in either argument order. KL grows
like the *square* of the distance between the means, and squared distances do
not satisfy the triangle inequality.

The asymmetry has a use: the *direction* of KL is a modeling decision
(:numref:`sec_mdl-fwd-vs-rev-kl`). Total variation, the Hellinger distance
$d_{\mathrm{H}}$ and the Wasserstein distance $W_1$ are metrics, and arguments
that need the triangle inequality use those.

### The f-Divergence Template
:label:`sec_mdl-f-divergences`

Gibbs' inequality, $D_{\textrm{KL}}(P\|Q) \geq 0$, is the reason KL is a
divergence. Which properties of KL did its proof use, and what else has them?
We answer in three steps.

First, we write KL as an average under $Q$ of a function of the ratio
$u(x) = p(x)/q(x)$:

$$
D_{\textrm{KL}}(P\|Q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
= \sum_x q(x)\, \frac{p(x)}{q(x)} \log\frac{p(x)}{q(x)}
= \mathbb{E}_{x \sim Q}\big[u(x) \log u(x)\big].
$$

KL weighs each outcome by $Q$ and charges it $u \log u$. On the running pair of
:numref:`sec_mdl-information_theory`, $P = (0.6, 0.3, 0.1)$ and
$Q = (0.2, 0.5, 0.3)$, the ratios are $u = 3$, $0.6$ and $\tfrac13$, with
weights $0.2$, $0.5$ and $0.3$:

$$
0.2\,(3 \log 3) + 0.5\,(0.6 \log 0.6) + 0.3\,\big(\tfrac13 \log \tfrac13\big)
= 0.659 - 0.153 - 0.110 = 0.396 \textrm{ nats},
$$

the KL divergence computed in that section.

Second, Gibbs' proof used only two facts about $u \log u$: it is convex, and it
is zero at $u = 1$. Third, we replace $u \log u$ by any function with these two
properties. Let $f : (0, \infty) \to \mathbb{R}$ be convex with $f(1) = 0$, and
set $f(0) = \lim_{u \downarrow 0} f(u)$, which may be $+\infty$. Let $p(x) = 0$
wherever $q(x) = 0$ (we write $P \ll Q$). The **f-divergence** of $P$ from $Q$
is :cite:`Csiszar.1967`

$$
D_f(P\|Q) = \mathbb{E}_{x \sim Q}\!\left[ f\!\left( \frac{p(x)}{q(x)} \right) \right].
$$
:eqlabel:`eq_mdl-f-div-def`

The ratio says, point by point, how much $P$ over- or under-represents $x$
relative to $Q$; $f$ sets the penalty, which is zero at $u = 1$. We call $f$
the *divergence function*. The f-GAN literature calls it the generator, and the
code below keeps the functions in a dictionary of that name; like
:numref:`sec_gan_objectives`, we reserve "generator" for a network that
produces samples.

We call $f$ *strictly convex at $1$* if some line through $(1, 0)$ lies
strictly below $f$ away from $1$: $f(u) > c\,(u - 1)$ for all $u \neq 1$ and
some slope $c$.

**Proposition (non-negativity).** *For any convex $f$ with $f(1) = 0$ and
$P \ll Q$,*

$$
D_f(P\|Q) \geq 0,
$$

*and if $f$ is strictly convex at $1$, then $D_f(P\|Q) = 0$ if and only if
$P = Q$.*

**Proof.** Jensen's inequality (:numref:`subsec_mdl-jensen`) says that
$\mathbb{E}[f(U)] \geq f(\mathbb{E}[U])$ for a convex $f$ and a random variable
$U$. Apply it to $U = u(x)$ with $x \sim Q$. Everything rests on one fact: the
ratio averages to one under $Q$,
$\mathbb{E}_Q[u] = \sum_x q(x)\, p(x)/q(x) = \sum_x p(x) = 1$. Hence

$$
D_f(P\|Q) = \mathbb{E}_Q[f(u)] \geq f(\mathbb{E}_Q[u]) = f(1) = 0.
$$

For the equality case, take the line $c\,(u-1)$ below $f$. Then
$D_f(P\|Q) \geq \mathbb{E}_Q[c\,(u - 1)] = 0$, and the gap is the average of
$f(u) - c\,(u - 1)$, which is positive wherever $u \neq 1$. So $D_f = 0$ forces
$u = 1$ for $Q$-almost every $x$, that is, $P = Q$. $\blacksquare$

The definition assumed $P \ll Q$, and the opening's point masses violate it.
Generative models begin training in just this regime, with the model's samples
and the data in disjoint regions, so we must say what $D_f$ means there.
Outcomes with $p(x) = q(x) = 0$ contribute nothing. Where $p(x) = 0 < q(x)$,
the term is $q(x)\, f(0)$, as the definition already says. Where
$q(x) = 0 < p(x)$ the ratio is infinite, and the term is $p(x)\, f'(\infty)$
with $f'(\infty) := \lim_{t \to \infty} f(t)/t$, which is the limit of
$q\, f(p/q)$ as $q \downarrow 0$. For a convex $f$ the ratio $f(t)/t$ and the
slope $f'(t)$ have the same limit, hence the notation. Both $f(0)$ and
$f'(\infty)$ may be $+\infty$, the first for reverse KL and the second for KL.

On the point masses $\delta_0$ and $\delta_d$ with $d \neq 0$, the outcome $0$
has $q = 0 < p = 1$ and the outcome $d$ has $p = 0 < q = 1$, so

$$
D_f(\delta_0 \| \delta_d) = f(0) + f'(\infty) \qquad \textrm{for every } d \neq 0.
$$

For total variation, whose divergence function is $\tfrac12|u - 1|$, this is
$\tfrac12 + \tfrac12 = 1$; for Jensen--Shannon it is
$\tfrac12 \log 2 + \tfrac12 \log 2 = \log 2$; for squared Hellinger,
$1 + 1 = 2$; for $\chi^2$ and KL, $f'(\infty) = \infty$. One line has explained
the first three rows of :numref:`tab_mdl-point-mass-table`: every f-divergence
is constant on disjoint supports. It cannot tell a near miss from a distant
one, and a model trained on it receives no gradient from the distance between
the supports.

### Common Divergence Functions

Each row of :numref:`tab_mdl-f-div-generators` is one choice of $f$, and
:numref:`fig_mdl-f-div-generators` plots the curves. The column $f''(1)$, the
curvature at $u = 1$, matters in the next subsection.

:Common divergence functions. Each row gives the function $f$ of one divergence, its curvature at $u = 1$, and whether the divergence is symmetric and whether it, or its square root, is a metric.
:label:`tab_mdl-f-div-generators`

| Divergence | $f(u)$ | $f''(1)$ | Symmetric? | Metric? |
|:----|:----|:----|:----|:----|
| Kullback--Leibler (forward) | $u \log u$ | $1$ | no | no |
| reverse KL | $-\log u$ | $1$ | no | no |
| Pearson $\chi^2$ | $(u - 1)^2$ | $2$ | no | no |
| squared Hellinger $d_{\mathrm{H}}^2$ | $(\sqrt{u} - 1)^2$ | $\tfrac12$ | yes | $d_{\mathrm{H}}$ is a metric |
| total variation | $\tfrac{1}{2}\lvert u - 1 \rvert$ | none (corner) | yes | yes |
| Jensen--Shannon | $\tfrac{u}{2} \log u - \tfrac{u+1}{2} \log \tfrac{u+1}{2}$ | $\tfrac14$ | yes | $\sqrt{\textrm{JS}}$ is a metric |
| $\alpha$-divergence ($\alpha \neq 0, 1$) | $\dfrac{u^\alpha-\alpha u+\alpha-1}{\alpha(\alpha-1)}$ | $1$ | only at $\alpha = \tfrac{1}{2}$ | square root at $\alpha = \tfrac{1}{2}$ |

![Six divergence functions, each shifted by the multiple $f'(1)(u-1)$, which changes no divergence. After the shift every curve is non-negative and tangent to zero at $u=1$, and the curvatures there, $1, 1, 2, \tfrac12, \tfrac14$ (total variation has a corner), are the constants of the local quadratic. Far from $u=1$ the functions penalize over- and under-representation very differently: reverse KL diverges as $u \to 0$, KL and $\chi^2$ grow superlinearly, and total variation grows linearly.](../img/mdl-it-f-div-generators.svg)
:label:`fig_mdl-f-div-generators`

*Reverse KL* uses $f(u) = -\log u$, and
$\mathbb{E}_Q[-\log(p/q)] = \mathbb{E}_Q[\log(q/p)] = D_{\textrm{KL}}(Q\|P)$.
So the asymmetry of KL is a *choice of $f$*: the two directions are two rows of
the table (problem :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`a
gives the general rule for swapping arguments).

*Total variation* uses $f(u) = \tfrac{1}{2}|u - 1|$, which reduces to
$\sum_x q(x)\, \tfrac12 |p(x)/q(x) - 1| = \tfrac{1}{2}\sum_x |p(x) - q(x)|$,
half the $\ell_1$ distance between the probability vectors. On the running
pair, $\textrm{TV} = \tfrac12(0.4 + 0.2 + 0.2) = 0.4$. The same number is the
largest disagreement between $P$ and $Q$ about the probability of any event,
the form in which :numref:`sec_mdl-tv-pinsker` uses it. Pearson's divergence is
$\chi^2(P\|Q) = \sum_x (p(x) - q(x))^2 / q(x)$, which here is
$0.16/0.2 + 0.04/0.5 + 0.04/0.3 = 1.0133$.

The **Jensen--Shannon divergence** :cite:`Lin.1991` symmetrizes KL through the
even mixture $M = \tfrac{1}{2}(P + Q)$:

$$
\textrm{JS}(P, Q)
= \frac{1}{2} D_{\textrm{KL}}(P \,\|\, M)
+ \frac{1}{2} D_{\textrm{KL}}(Q \,\|\, M).
$$
:eqlabel:`eq_mdl-js-def`

Each distribution is compared with the mixture, which has mass wherever either
one does, so neither KL term can be infinite. JS is therefore symmetric and
bounded, $0 \leq \textrm{JS}(P, Q) \leq \log 2$, with the upper bound attained
when $P$ and $Q$ have disjoint supports (then $p/m = 2$ wherever $p > 0$). Its
square root is a metric :cite:`Endres.Schindelin.2003`. Expanding
:eqref:`eq_mdl-js-def` in the ratio $u$ gives the function in the table
(problem :exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`c).

The squared Hellinger divergence is
$d_{\mathrm{H}}^2(P, Q) = \sum_x \big(\sqrt{p(x)} - \sqrt{q(x)}\big)^2$. So
$d_{\mathrm{H}}$ is the Euclidean distance between the vectors $\sqrt{p}$ and
$\sqrt{q}$, which is why it is a metric (problem
:exref:`ex_mdl-divergences-distances-prove-that-total-variation`b).

The table's last row is a family within the family, the *$\alpha$-divergences*
:cite:`Amari.2016`, with

$$
f_\alpha(u)=\frac{u^\alpha-\alpha u+\alpha-1}{\alpha(\alpha-1)},
$$
:eqlabel:`eq_mdl-alpha-generator`

which is convex with $f_\alpha(1) = 0$ and curvature $1$ at $u = 1$ for every
$\alpha \neq 0, 1$. As $\alpha \to 1$ it tends to $u\log u - u + 1$ and as
$\alpha \to 0$ to $-\log u + u - 1$. A multiple of $u - 1$ averages to zero
under $Q$, so the two ends are forward and reverse KL. Halfway, at
$\alpha = \tfrac12$, the function is $2(\sqrt{u} - 1)^2$, twice squared
Hellinger, and $\alpha = 2$ gives half of $\chi^2$. The family is used where a
continuous passage between the two directions of KL is wanted: variational
inference with Rényi's divergence :cite:`Li.Turner.2016` tunes $\alpha$ between
the two behaviors of :numref:`sec_mdl-fwd-vs-rev-kl`, and Rényi's divergence,
an increasing function of $D_{f_\alpha}$ (problem
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d), is the quantity
that differential privacy bounds :cite:`Mironov.2017`.

We evaluate the first six rows on the running pair, in both argument orders.
<!-- WRITER: the cell no longer recomputes JS from the mixture formula (the line repeated the JS row, 0.0911); the next sentence is now false and goes. -->
The last line of the output recomputes JS from the mixture formula.

```{.python .input #divergences-distances-f-table}
def f_divergence(f, p, q):
    """D_f(P||Q) = sum_x q(x) f(p(x)/q(x)); nats for KL/JS, else unitless."""
    return np.sum(q * f(p / q))

generators = {
    'KL':          lambda u: u * np.log(u),
    'reverse KL':  lambda u: -np.log(u),
    'chi^2':       lambda u: (u - 1) ** 2,
    'Hellinger^2': lambda u: (np.sqrt(u) - 1) ** 2,
    'TV':          lambda u: 0.5 * np.abs(u - 1),
    'JS':          lambda u: u / 2 * np.log(u) - (u + 1) / 2 * np.log((u + 1) / 2),
}

p = np.array([0.6, 0.3, 0.1])
q = np.array([0.2, 0.5, 0.3])
for name, f in generators.items():
    print(f'{name:12s} D_f(P||Q) = {f_divergence(f, p, q):.4f}   '
          f'D_f(Q||P) = {f_divergence(f, q, p):.4f}')
```

The KL row is asymmetric ($0.3961$ against $0.3653$ nats, the two numbers of
:numref:`sec_mdl-information_theory`), and the reverse-KL row is the KL row
with its columns swapped. The $\chi^2$ row is asymmetric too ($1.0133$ against
$0.8000$). Squared Hellinger ($0.1862$), TV ($0.4$) and JS are symmetric, and
JS, at $0.0911$ nats, sits well under its ceiling of $\log 2 \approx 0.693$.
The six divergences disagree by a factor of ten about one pair of
distributions. Is there any order in these numbers?

### The Local Quadratic

The functions in :numref:`fig_mdl-f-div-generators` differ greatly away from
$u = 1$. Near $u = 1$ they turn out to be one function up to a constant, which
is perhaps the most surprising fact about the family.

Adding $c\,(u - 1)$ to $f$ changes no divergence, because
$\mathbb{E}_Q[c\,(u - 1)] = c\,(1 - 1) = 0$ (problem
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`b). So the value
$f(1) = 0$ and the slope $f'(1)$ carry no information, and the first property
that distinguishes divergence functions is the curvature $f''(1)$. Near $u = 1$
every smooth $f$ looks like the parabola $\tfrac12 f''(1)(u - 1)^2$, and the
average of $(u-1)^2$ under $Q$ is the $\chi^2$ divergence,
$\mathbb{E}_Q[(u - 1)^2] = \sum_x (p(x) - q(x))^2/q(x)$. Let
$r = \max_x |u(x) - 1|$ measure how far the ratio strays from $1$.

**Proposition (local quadratic).** *Let $P \ll Q$, let $0 < r < 1$, let $f$ be
three times differentiable on $[1 - r, 1 + r]$ with $|f'''| \leq B$ there, and
let $|u(x) - 1| \leq r$ for every $x$ with $q(x) > 0$. Then*

$$
\left| D_f(P\|Q) - \frac{f''(1)}{2}\, \chi^2(P\|Q) \right|
\leq \frac{B r}{6}\, \chi^2(P\|Q).
$$
:eqlabel:`eq_mdl-f-div-local`

**Proof.** Taylor's theorem with remainder gives
$f(u) = f'(1)(u - 1) + \tfrac12 f''(1)(u - 1)^2 + R(u)$ with
$|R(u)| \leq B |u - 1|^3 / 6$, since $f(1) = 0$. Take $\mathbb{E}_Q$ term by
term. The linear term vanishes because $\mathbb{E}_Q[u - 1] = 0$, and with it
the only first-order difference between one $f$ and another. The quadratic term
is $\tfrac12 f''(1)\, \chi^2(P\|Q)$. The remainder obeys
$\mathbb{E}_Q|R| \leq \tfrac{B}{6}\,\mathbb{E}_Q|u - 1|^3 \leq \tfrac{B r}{6}\, \mathbb{E}_Q(u - 1)^2 = \tfrac{B r}{6}\,\chi^2$.
$\blacksquare$

So $D_f \approx \tfrac12 f''(1)\, \chi^2$, with a relative error proportional
to $r$. How close must $P$ be to $Q$ before this is useful? We move the running
pair's $P$ toward $Q$ along the segment $P_t = Q + t\,(P - Q)$, which
multiplies every $u - 1$ by $t$, and print
$D_f(P_t\|Q) / \big(\tfrac12 f''(1)\, \chi^2(P_t\|Q)\big)$ for four
divergences.

```{.python .input #mdl-divergences-distances-the-local-quadratic}
curvature = {'KL': 1.0, 'reverse KL': 1.0, 'Hellinger^2': 0.5, 'JS': 0.25}
for t in (0.5, 0.1, 0.01):
    p_t = q + t * (p - q)                     # moves P_t toward Q as t -> 0
    chi2 = f_divergence(generators['chi^2'], p_t, q)
    ratios = [f_divergence(generators[name], p_t, q) / (c / 2 * chi2)
              for name, c in curvature.items()]
    print(f't = {t:4}: ' + '  '.join(f'{name} {r:.3f}'
                                     for name, r in zip(curvature, ratios)))
```

At $t = 0.5$ the four divergences are about $75\%$ to $85\%$ of their
quadratics, at $t = 0.1$ more than $91\%$, and at $t = 0.01$ more than $99\%$.
Once $t$ is small, the shortfall shrinks in proportion to $t$, as the
proposition's factor $r$ predicts. Smooth f-divergences disagree only about
distant distributions, which is the regime early in training, when the model is
far from the data.

On a parametric family the common quadratic is the Fisher information
$\mathbf{I}(\boldsymbol{\theta}) = \mathbb{E}\big[\nabla_{\boldsymbol{\theta}}
\log q_{\boldsymbol{\theta}}\, \nabla_{\boldsymbol{\theta}} \log
q_{\boldsymbol{\theta}}^\top\big]$ of :numref:`sec_mdl-maximum_likelihood`. For
a small step $\mathbf{h}$ the ratio of $q_{\boldsymbol{\theta} + \mathbf{h}}$
to $q_{\boldsymbol{\theta}}$ is
$u \approx 1 + \mathbf{h}^\top \nabla_{\boldsymbol{\theta}} \log q_{\boldsymbol{\theta}}$,
so $\chi^2(q_{\boldsymbol{\theta} + \mathbf{h}}\| q_{\boldsymbol{\theta}}) =
\mathbb{E}[(u - 1)^2] \approx \mathbf{h}^\top \mathbf{I}(\boldsymbol{\theta})\,
\mathbf{h}$, and every smooth $D_f$ has Hessian
$f''(1)\, \mathbf{I}(\boldsymbol{\theta})$ at $\mathbf{h} = \mathbf{0}$. The
natural gradient :cite:`Amari.1998`, which measures steps by KL, would
therefore be the same method under any other smooth f-divergence. Total
variation is the exception: its divergence function has a corner at $u = 1$ and
no curvature, and problem
:exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`b shows
what that changes when $P$ approaches $Q$.

### Forward and Reverse KL
:label:`sec_mdl-fwd-vs-rev-kl`

The two KL rows of the table are two different fitting criteria. When we fit a
model $Q_\theta$ to a target $P$, the two directions ask for different things:

* **Forward KL**, $D_{\textrm{KL}}(P \,\|\, Q_\theta) = \mathbb{E}_P[\log(p/q_\theta)]$,
  averages over the *truth*. If $q_\theta(x) \to 0$ while $p(x) > 0$, the
  divergence blows up. Forward KL is **zero-avoiding**, or *mode-covering*:
  the model must cover every mode of the data, even if that smears mass over
  regions $P$ never visits.
* **Reverse KL**, $D_{\textrm{KL}}(Q_\theta \,\|\, P) = \mathbb{E}_{Q_\theta}[\log(q_\theta/p)]$,
  averages over the *model*. The model is punished for putting mass where
  the truth has none, while modes it never visits leave the objective
  untouched. Reverse KL is **zero-forcing**, or *mode-seeking*: the model
  concentrates on the mass it can explain.

Maximum likelihood *is* forward KL minimization:
:numref:`subsec_mdl-nll-crossentropy` shows that the average negative
log-likelihood is the cross-entropy from the empirical distribution to the
model, which differs from forward KL by a constant. Variational inference and
the ELBO (:numref:`sec_mdl-latent-em-elbo`) minimize the *reverse* KL from the
approximate posterior to the true one. The choice returns wherever one
distribution is fitted to another: the distillation of language models
(:numref:`sec_lm_distillation`) and the KL-regularized policies of
:numref:`sec_regularized` each have to pick a direction.

When the model family contains the target, both directions agree on the answer.
They differ for a *misspecified* family, and the simplest instance is a single
Gaussian fitted to a bimodal target. The forward fit needs no computation.
Forward KL is a constant minus $\mathbb{E}_P[\log q(X)]$, so its minimizer is
the maximum-likelihood Gaussian with $P$ in place of the empirical
distribution, and maximum likelihood in an exponential family matches the
expected sufficient statistics (:numref:`subsec_mdl-moment-matching`):
$\mu^* = \mathbb{E}_P[X]$ and $(\sigma^*)^2 = \textrm{Var}_P(X)$, provided the
variance is finite. Information geometry calls this fit the *M-projection*
:cite:`Amari.2016`. The reverse direction has no such closed form, and it can
have one local optimum per mode.

For the mixture $P = 0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$
we can therefore predict the forward fit: mean $0.7(-2) + 0.3(2) = -0.8$ and
variance $0.36 + 0.7 \cdot 0.3 \cdot 4^2 = 3.72$. The reverse fit we leave to a
derivative-free optimizer over $(\mu, \log\sigma)$, started once at the origin
and once near the minor mode, with each KL evaluated by quadrature on a grid.

```{.python .input #divergences-distances-fwd-rev-kl}
w, mus, sigmas = np.array([0.7, 0.3]), np.array([-2.0, 2.0]), np.array([0.6, 0.6])
x = np.linspace(-8.0, 8.0, 4001)

def log_components(x, w, mus, sigmas):
    """log(w_k N(x; mu_k, sigma_k^2)) for every component k; shape (k, len(x))."""
    return (np.log(w)[:, None] - 0.5 * np.log(2 * np.pi * sigmas[:, None] ** 2)
            - (x - mus[:, None]) ** 2 / (2 * sigmas[:, None] ** 2))

def log_mix(x, w, mus, sigmas):
    """Log-density of the mixture sum_k w_k N(mu_k, sigma_k^2), computed stably."""
    c = log_components(x, w, mus, sigmas)
    c_max = c.max(axis=0)
    return c_max + np.log(np.exp(c - c_max).sum(axis=0))

log_p = lambda x: log_mix(x, w, mus, sigmas)

def log_q(x, theta):
    mu, log_sigma = theta
    return (-0.5 * np.log(2 * np.pi) - log_sigma
            - (x - mu) ** 2 / (2 * np.exp(2 * log_sigma)))

def kl(log_a, log_b, x):
    """D_KL(A||B) by quadrature on the grid x; in nats."""
    return integrate.trapezoid(np.exp(log_a) * (log_a - log_b), x)

fwd = lambda th: kl(log_p(x), log_q(x, th), x)     # D_KL(P || Q_theta)
rev = lambda th: kl(log_q(x, th), log_p(x), x)     # D_KL(Q_theta || P)

th_fwd = optimize.minimize(fwd, x0=[0.0, 0.0], method='Nelder-Mead').x
th_rev = optimize.minimize(rev, x0=[0.0, 0.0], method='Nelder-Mead').x
th_rev2 = optimize.minimize(rev, x0=[2.5, -0.5], method='Nelder-Mead').x

for name, th, val in [('forward KL', th_fwd, fwd(th_fwd)),
                      ('reverse KL', th_rev, rev(th_rev)),
                      ('reverse KL, 2nd start', th_rev2, rev(th_rev2))]:
    print(f'{name:22s} mu = {th[0]:+.3f}, sigma = {np.exp(th[1]):.3f}, '
          f'KL = {val:.3f} nats')

d2l.plot(x, [np.exp(log_p(x)), np.exp(log_q(x, th_fwd)),
             np.exp(log_q(x, th_rev))], 'x', 'density',
         legend=['target P (mixture)', 'forward-KL fit', 'reverse-KL fit'],
         figsize=(6, 3))
```

The forward fit lands on the prediction, $\mu = -0.800$ and
$\sigma = 1.929 \approx \sqrt{3.72}$. It is a broad Gaussian draped across both
modes, with substantial mass in the valley between them where $P$ has almost
none. The reverse fit lands at $\mu = -1.998$, $\sigma = 0.603$: it matches the
dominant component and assigns little mass to the minor one. Its divergence,
$0.356$ nats, is close to $\log(1/0.7) \approx 0.357$, the divergence of
treating the $70\%$ component as the whole distribution. Started near the minor
mode, the optimizer converges instead to $\mu = +1.995$ with KL
$1.202 \approx \log(1/0.3)$ nats, so which optimum a variational method finds
depends on its initialization. The width of the reverse fit, $0.60$ against the
target's $1.93$, is the reason variational posteriors tend to be too narrow.

Maximum likelihood penalizes a missed mode heavily, and reverse-KL objectives
do not (problem :exref:`ex_mdl-divergences-distances-mode-dropping`). The two
divergence functions say so in the conventions for vanishing densities:
$u \log u$ has $f(0) = 0$ and $f'(\infty) = \infty$, and $-\log u$ has them the
other way round.

### Total Variation and Pinsker's Inequality
:label:`sec_mdl-tv-pinsker`

The simplest functions to average are indicators of events, and averaging an
indicator gives a probability. So the first question is whether some event has
very different probabilities under $P$ and under $Q$. The *total variation
distance* is the largest such disagreement over all events $A$:

$$
\textrm{TV}(P, Q) = \sup_{A} \,\lvert P(A) - Q(A) \rvert.
$$
:eqlabel:`eq_mdl-tv-def`

If $\textrm{TV}(P, Q) = 0.03$, then no event distinguishes the two
distributions by more than three percentage points. TV is a metric (symmetry is
visible in :eqref:`eq_mdl-tv-def`, and the triangle inequality is problem
:exref:`ex_mdl-divergences-distances-prove-that-total-variation`a). It is also
the f-divergence with $f(u) = \tfrac12|u - 1|$ of
:numref:`tab_mdl-f-div-generators`: for discrete $P$ and $Q$,

$$
\textrm{TV}(P, Q) = \frac{1}{2} \sum_x \lvert p(x) - q(x) \rvert,
$$

and the supremum in :eqref:`eq_mdl-tv-def` is attained at
$A^\star = \{x : p(x) > q(x)\}$. To see this, take any event $A$. Then
$P(A) - Q(A) = \sum_{x \in A} (p(x) - q(x)) \leq \sum_{x \in A^\star} (p(x) - q(x))$,
since passing from $A$ to $A^\star$ adds only non-negative terms and removes
only non-positive ones. Because $\sum_x (p(x) - q(x)) = 0$, the positive part
of $p - q$ equals the negative part in magnitude, so the right side is
$\tfrac12 \sum_x |p(x) - q(x)|$, and the same bound holds for $Q(A) - P(A)$. On
the running pair $P = (0.6, 0.3, 0.1)$, $Q = (0.2, 0.5, 0.3)$ of
:numref:`sec_mdl-information_theory`, $A^\star$ is the first outcome and
$\textrm{TV} = 0.6 - 0.2 = 0.4$. :numref:`fig_mdl-tv-area` shows the same for
two densities.

![Two equal-variance densities and the regions where they disagree. The lobe where $p > q$ and the lobe where $q > p$ have the same area, and each equals the total variation distance. The event that attains the supremum is $A^\star = \{p > q\}$, here the half-line left of the crossing: betting on $P$ exactly there is the best single-sample test.](../img/mdl-it-tv-area.svg)
:label:`fig_mdl-tv-area`

This gives TV its operational meaning. Hand a tester one sample, drawn from $P$
or from $Q$ with probability $\tfrac12$ each, and ask which distribution
produced it. A test that answers "$P$" when the sample lands in an event $A$,
and "$Q$" otherwise, is right with probability

$$
\tfrac12 P(A) + \tfrac12 \big(1 - Q(A)\big)
= \tfrac12 \big(1 + P(A) - Q(A)\big) \leq \tfrac12 \big(1 + \textrm{TV}(P, Q)\big),
$$

with equality at $A = A^\star$. Every test is of this form for some $A$ (a
randomized test averages such tests and does no better), so the best possible
test beats coin flipping by $\textrm{TV}/2$. TV is a limit on what any test can
detect, and KL in turn limits TV: a small KL divergence certifies that no test
can tell the distributions apart well. This is the inequality that later proofs
use most, because KL is usually the quantity we can compute and TV the one we
want to control.

**Proposition (Pinsker's inequality).** :cite:`Pinsker.1964` *For distributions
$P$ and $Q$ on a finite or countable set,*

$$
\textrm{TV}(P, Q) \;\leq\; \sqrt{ \tfrac{1}{2}\, D_{\textrm{KL}}(P\|Q) }.
$$
:eqlabel:`eq_mdl-pinsker`

**Proof.** *Step 1: two outcomes.* For coins with heads-probabilities
$a \in [0, 1]$ and $b \in (0, 1)$ the KL divergence is
$D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) = a \log\frac{a}{b} + (1-a) \log\frac{1-a}{1-b}$
and the TV distance is $|a - b|$, so we must show that
$\rho(a) = D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) - 2(a - b)^2$ is
non-negative. Fix $b$. Then $\rho(b) = 0$, $\rho'(b) = 0$, and for $0 < a < 1$

$$
\rho''(a) = \frac{1}{a(1-a)} - 4 \geq 0,
$$

since $a(1-a) \leq \tfrac{1}{4}$. A convex function with value and slope zero
at $a = b$ is non-negative, and by continuity (with $0 \log 0 = 0$) also at the
endpoints $a = 0$ and $a = 1$. If $b$ is $0$ or $1$, the divergence is infinite
unless $a = b$, and there is nothing to prove.

*Step 2: reduction to two outcomes.* Let $A^\star = \{p > q\}$, and set
$a = P(A^\star)$, $b = Q(A^\star)$, so that $\textrm{TV}(P, Q) = a - b$.
Merging the outcomes inside $A^\star$ and inside its complement can only
*decrease* KL, by the log-sum inequality: for non-negative numbers $v_i$ and
$w_i$,

$$
\sum_i v_i \log\frac{v_i}{w_i} \geq \big(\sum_i v_i\big) \log \frac{\sum_i v_i}{\sum_i w_i},
$$

itself one application of Jensen to $t \mapsto t\log t$. Applied separately to
the terms in $A^\star$ and in its complement,

$$
D_{\textrm{KL}}(P\|Q) \;\geq\; D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) \;\geq\; 2(a - b)^2
= 2\,\textrm{TV}(P, Q)^2,
$$

which rearranges to :eqref:`eq_mdl-pinsker`. $\blacksquare$

Pinsker's inequality has no useful converse: a model that assigns $q = 0$ to a
rare event has infinite KL at tiny TV. The bound also goes slack for distant
pairs, since TV is at most $1$. Two unit-variance Gaussians $50$ apart have
$\textrm{TV} \approx 1$ but, by :eqref:`eq_mdl-dd-gaussian-kl`, KL
$= 50^2/2 = 1250$ nats.

The proof leaves two questions open: how tight is the bound for typical pairs,
and can the constant $\tfrac12$ be improved? For the first we draw $10{,}000$
random pairs of distributions on five outcomes. For the second we let two coins
approach each other; Step 1 lost nothing to second order at $a = b = \tfrac12$,
so the ratio $\textrm{TV}/\sqrt{\textrm{KL}/2}$ should approach $1$ there.

```{.python .input #divergences-distances-pinsker}
rng = np.random.default_rng()
pr, qr = rng.dirichlet(np.ones(5), 10000), rng.dirichlet(np.ones(5), 10000)
tv = 0.5 * np.abs(pr - qr).sum(axis=1)
kl_pq = np.sum(pr * np.log(pr / qr), axis=1)
ratio = tv / np.sqrt(kl_pq / 2)
print(f'TV / sqrt(KL/2) over 10,000 random pairs: median {np.median(ratio):.2f}, '
      f'max {ratio.max():.3f}')
for delta in (0.1, 0.01, 0.001):
    a, b = np.array([0.5, 0.5]), np.array([0.5 + delta, 0.5 - delta])
    print(f'coins 1/2 vs 1/2 + {delta:5}: TV / sqrt(KL/2) = '
          f'{delta / np.sqrt(kl_div(a, b) / 2):.6f}')
grid = np.linspace(0, 3, 100)
d2l.plot([kl_pq[:2000], grid], [tv[:2000], np.sqrt(grid / 2)], 'KL (nats)',
         'TV', legend=['random pairs', 'sqrt(KL/2)'], fmts=('.', '-'),
         xlim=[0, 3], figsize=(4.5, 3))
```

<!-- WRITER: the cell now prints the median ratio TV/sqrt(KL/2) over the 10,000 pairs: 0.78 in all four tabs (run-to-run 0.78-0.79), slightly above "three quarters"; read the printed median. Max 0.98-0.99 (unchanged, "within a few hundredths of 1" holds). -->
In the scatter plot every pair lies below the curve $\sqrt{\textrm{KL}/2}$,
typically at about three quarters of it, and the gap grows with KL; the largest
ratio among the $10{,}000$ pairs comes within a few hundredths of $1$. For
nearly fair coins the ratio climbs to $0.9899$, $0.9999$ and $0.999999$, so the
constant $\tfrac{1}{2}$ cannot be improved (problem
:exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`c).

### Data Processing
:label:`subsec_mdl-f-div-data-processing`

Can a feature extractor, a quantizer, or any other processing make two
distributions easier to tell apart? Pinsker's proof merged outcomes and found
that KL can only fall. The same holds for every f-divergence and every kind of
processing. A *channel* $\kappa(y \mid x)$ is any rule, deterministic or
random, that turns an outcome $x$ into an outcome $y$; it maps $P$ to the
output distribution $(P\kappa)(y) = \sum_x \kappa(y \mid x)\, p(x)$, and $Q$
likewise.

**Proposition (data processing for f-divergences).** *For any channel $\kappa$,
convex $f$ with $f(1) = 0$, and $P \ll Q$ on a finite or countable set,*

$$
D_f(P\kappa \,\|\, Q\kappa) \;\leq\; D_f(P\|Q).
$$
:eqlabel:`eq_mdl-f-div-dpi`

**Proof.** Since $\sum_y \kappa(y \mid x) = 1$,
$D_f(P\|Q) = \sum_y \sum_x \kappa(y \mid x)\, q(x)\, f\big(p(x)/q(x)\big)$. Fix
$y$ with $(Q\kappa)(y) > 0$. The weights
$\kappa(y \mid x)\, q(x) / (Q\kappa)(y)$ sum to one over $x$, and averaging the
ratios $p(x)/q(x)$ with these weights gives $(P\kappa)(y)/(Q\kappa)(y)$.
Jensen's inequality therefore bounds the inner sum below by
$(Q\kappa)(y)\, f\big((P\kappa)(y)/(Q\kappa)(y)\big)$, which is the $y$-th term
of $D_f(P\kappa\|Q\kappa)$. An outcome $y$ with $(Q\kappa)(y) = 0$ has
$\kappa(y \mid x) = 0$ wherever $q(x) > 0$, hence also $(P\kappa)(y) = 0$, and
contributes nothing to either side. Summing over $y$ finishes the proof.
$\blacksquare$

No processing of the data, whatever the divergence function, makes two
distributions easier to tell apart. In deep learning the inequality is mostly
read from right to left: a divergence measured on features, on quantized values
or on any other function of the data is a lower bound on the divergence between
the inputs. Merging outcomes is the deterministic channel
$x \mapsto \mathbf{1}(x \in A^\star)$ of Pinsker's proof, and the
data-processing inequality for mutual information (:numref:`subsec_mdl-dpi`)
follows from the case of KL. The inequality belongs to the f-divergences alone:
MMD and $W_1$ can grow under a map (problem
:exref:`ex_mdl-divergences-distances-q-mathcaln-delta`a).

### The Fenchel Conjugate and the f-GAN Bound
:label:`sec_mdl-f-gan-dual`

A network that generates samples gives us draws from its distribution $Q$, and
we hold draws from the data distribution $P$, but neither tells us the ratio
$p(x)/q(x)$ that :eqref:`eq_mdl-f-div-def` needs. Can $D_f$ be computed from
samples alone? It can be bounded from below, and the bound turns a divergence
into a training objective.

The tool is the convex conjugate of :numref:`subsec_mdl-convex-conjugate`: for
a convex $f$,

$$
f^*(t) = \sup_{u}\, \big( ut - f(u) \big),
$$

so that $-f^*(t)$ is the intercept of the tangent line to $f$ with slope $t$
(:numref:`fig_mdl-f-gan-tangent-bound`). A convex function lies above each of
its tangents, which is the Fenchel--Young inequality

$$
f(u) \geq ut - f^*(t) \quad \textrm{for all } t,
$$
:eqlabel:`eq_mdl-fenchel-young`

with equality when $t$ is the slope of $f$ at $u$, that is, $t = f'(u)$.

![A convex divergence function, here $f(u)=u\log u$, is the upper envelope of its tangent lines. Each tangent of slope $t$ has intercept given by the negative conjugate $-f^*(t)$ (here $-e^{t-1}$), so keeping one tangent yields the Fenchel--Young lower bound and taking the supremum over slopes recovers the function exactly. Applied pointwise to the density ratio, this picture is the $f$-GAN variational bound.](../img/mdl-it-f-gan-tangent-bound.svg)
:label:`fig_mdl-f-gan-tangent-bound`

Now we apply the inequality *pointwise inside the divergence* and let the slope
vary with $x$. We choose a function $T(x)$, the **critic**, and apply
:eqref:`eq_mdl-fenchel-young` at $u = p(x)/q(x)$ and $t = T(x)$.

**Proposition (f-GAN variational bound).** *Let $f$ be convex and
differentiable with $f(1) = 0$, let $P \ll Q$, and let $T$ be any critic with
$\mathbb{E}_P|T| < \infty$ and $\mathbb{E}_Q|f^*(T)| < \infty$. Then*

$$
D_f(P\|Q) \;\geq\; \mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[f^*(T(x))].
$$
:eqlabel:`eq_mdl-f-gan-bound`

*The supremum of the right side over critics equals $D_f(P\|Q)$, and it is
attained by $T^\star(x) = f'\big(p(x)/q(x)\big)$ wherever $T^\star$ is finite*
:cite:`Nowozin.Cseke.Tomioka.2016`.

**Proof.** Multiply :eqref:`eq_mdl-fenchel-young` at $u = p(x)/q(x)$,
$t = T(x)$ by $q(x) \geq 0$:

$$
q(x)\, f\!\left(\frac{p(x)}{q(x)}\right)
\;\geq\; p(x)\, T(x) - q(x)\, f^*(T(x)).
$$

Summing (or integrating) over $x$ gives :eqref:`eq_mdl-f-gan-bound`. Since
Fenchel--Young holds with equality at $t = f'(u)$, the choice
$T^\star(x) = f'(p(x)/q(x))$ makes the inequality an equality at every $x$,
hence in expectation. $\blacksquare$

Both expectations on the right of :eqref:`eq_mdl-f-gan-bound` are averages over
samples, of $T$ under $P$ and of $f^*(T)$ under $Q$, and neither needs a
density. A neural network $T$ can therefore be trained on minibatches to
maximize the bound. In the **f-GAN** objective the critic maximizes the bound,
which equals the divergence at the optimal critic, while the network that
produces the samples of $Q$, the generator of the GAN literature, minimizes it;
:numref:`sec_gan_objectives` trains such pairs. For KL, $f(u) = u \log u$, the
conjugate is $f^*(t) = e^{t-1}$, so
$D_{\textrm{KL}}(P\|Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T - 1}]$ for
every critic.

The original GAN of :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` is the case
$f(u) = u\log u - (u+1)\log(u+1)$. Its conjugate is $f^*(t) = -\log(1 - e^t)$
for $t < 0$, and writing the critic as the logarithm of a discriminator with
values in $(0, 1)$, $T = \log D$, turns the bound into the familiar objective

$$
\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]
= \mathbb{E}_{x \sim P}[\log D(x)] + \mathbb{E}_{x \sim Q}[\log(1 - D(x))].
$$

The optimal critic $f'(p/q) = \log\big(p/(p+q)\big)$ is the discriminator
$D^\star = p/(p + q)$. This $f$ equals
$2 f_{\textrm{JS}}(u) - (u - 1)\log 2 - \log 4$: twice the Jensen--Shannon
function, a multiple of $u - 1$, which changes nothing, and the constant
$f(1) = -\log 4$, which only shifts the value. At $D^\star$ the game is
therefore worth $2\,\textrm{JS}(P, Q) - \log 4$, on the running pair
$2(0.0911) - 1.386 = -1.204$.

Two caveats limit the bound in practice. First, it is tight only at the optimal
critic, so a critic that is not fully trained *underestimates* the divergence:
adversarial estimates are biased low. Second, the bound inherits the constancy
on disjoint supports. If we extend $T^\star = f'(p/q)$ formally beyond the
proposition's hypothesis $P \ll Q$, the ratio is $\infty$ on the data and $0$
on the model's samples, so $T^\star$ is the constant $f'(\infty)$ on one
support and $f'(0)$ on the other. A critic that is constant on the model's
samples gives the generator no gradient.

<!-- WRITER: the cell is redesigned. It no longer evaluates the parabola chi^2 (2c - c^2) at six scales: it trains the critic, three numbers T(x) on the running pair, by gradient ascent on the bound E_P[T] - E_Q[f*(T)] for chi^2 (f*(t) = t + t^2/4), from T = 0 with gradient p - q (1 + T/2) and step size 1, and prints the bound after 0, 1, 2, 5, 10, 20, 50 steps beside the exact 1.0133, plus the overconfident critic 2T* (bound 0). The parabola derivation below (from 'Consider the critics T = c T*' to 'evaluates it at six scales') goes; keep the conjugate and T* sentences. The function fgan_bound now takes f_star as an argument. -->
The first caveat can be measured on the running pair, where expectations are
sums. Take $\chi^2$, with $f(u) = (u-1)^2$, conjugate $f^*(t) = t + t^2/4$ and
optimal critic $T^\star = 2(p/q - 1)$. (The conjugate takes the supremum over
all real $u$; problem
:exref:`ex_mdl-divergences-distances-derive-convex-conjugate`a derives it and
shows that this loses nothing at $T^\star$.) Consider the critics
$T = c\,T^\star$ for a scale $c$, from untrained ($c = 0$) to overconfident
($c = 2$). Three sums give the bound in closed form:
$\sum_x p\, T^\star = 2\chi^2$, $\sum_x q\, T^\star = 0$ and
$\sum_x q\, (T^\star)^2/4 = \chi^2$, so

$$
\mathbb{E}_P[c\,T^\star] - \mathbb{E}_Q[f^*(c\,T^\star)]
= 2c\,\chi^2 - c^2 \chi^2 = \chi^2(P\|Q)\,(2c - c^2),
$$

a concave parabola in $c$ with its maximum $\chi^2$ at $c = 1$. The cell
evaluates it at six scales.

```{.python .input #divergences-distances-f-gan-bound}
def fgan_bound(T, p, q, f_star):
    """E_P[T] - E_Q[f*(T)]: a lower bound on D_f(P||Q) for any critic T."""
    return np.sum(p * T) - np.sum(q * f_star(T))

chi2_star = lambda t: t + t ** 2 / 4    # conjugate of f(u) = (u - 1)^2
exact = f_divergence(generators['chi^2'], p, q)
T, bounds = np.zeros(3), []             # untrained critic: one number per outcome
for step in range(51):
    bounds.append(fgan_bound(T, p, q, chi2_star))
    T = T + (p - q * (1 + T / 2))       # ascent step 1 along p - q f*'(T)
shown = (0, 1, 2, 5, 10, 20, 50)
print(f'exact chi^2(P||Q) = {exact:.4f}')
print('steps ' + ''.join(f'{s:8d}' for s in shown))
print('bound ' + ''.join(f'{bounds[s]:8.4f}' for s in shown))
T_star = 2 * (p / q - 1)                # the optimal critic f'(p/q)
print(f'overconfident critic 2 T*: bound = '
      f'{fgan_bound(2 * T_star, p, q, chi2_star):.4f}')
d2l.plot(np.arange(51), [bounds, [exact] * 51], 'ascent steps', 'value',
         legend=['bound', 'chi^2(P||Q)'], figsize=(4.5, 2.8))
```

<!-- WRITER: measured (deterministic, identical in all four tabs): bound 0.0000, 0.2240, 0.3935, 0.7036, 0.9107, 1.0013, 1.0133 after 0, 1, 2, 5, 10, 20, 50 steps, rising monotonically and always below the exact 1.0133 (gap 1e-5 at 50 steps); the overconfident critic 2T* gives 0.0000. 'c = 1/2 reports 0.76' is no longer printed (T*/2 still gives 0.760 by the formula); read 'half-trained' off the steps instead (about 70% of the divergence after 5 steps, 90% after 10). The plot is now the bound against ascent steps with the exact value as a line. -->
The optimal critic reproduces the exact divergence, $1.0133$. A half-trained
critic ($c = \tfrac12$) reports $0.76$, which is $75\%$ of the divergence, and
an overconfident one ($c = 2$) reports zero. Every critic other than $T^\star$
reports less than the divergence.

The f-divergences thus leave two things wanting. Estimated through a critic
they are biased low, and on disjoint supports they are constant whatever the
critic. Both defects come from the ratio $p/q$. The integral probability
metrics of :numref:`sec_mdl-ipm-mmd` and the transport distances of
:numref:`sec_mdl-optimal-transport` do without it.

## Summary

What we choose to compare decides what we must compute and what the comparison
can detect. f-divergences compare the density ratio point by point. They share
one proof of non-negativity, agree near $P = Q$ up to the curvature $f''(1)$,
are constant on disjoint supports, and have a variational form, the f-GAN
bound, that a critic estimates from samples with a downward bias. Forward KL,
which maximum likelihood minimizes, covers every mode, while reverse KL, which
variational inference minimizes, can settle on one.

Total variation is the best advantage of a test that sees one sample. It
is bounded by $\sqrt{D_{\textrm{KL}}/2}$, and no f-divergence grows when
the data are processed.

## Exercises

1. **Divergence functions and argument order.** Each row of
   :numref:`tab_mdl-f-div-generators` is one choice of $f$ in the
   template :eqref:`eq_mdl-f-div-def`.
    1. For a divergence function $f$, define $f^\diamond(u) = u\, f(1/u)$. Show that
       $f^\diamond$ is convex with $f^\diamond(1) = 0$ and that
       $D_f(Q\|P) = D_{f^\diamond}(P\|Q)$. Compute $f^\diamond$ for the KL
       function $u \log u$ and for the $\chi^2$ function $(u - 1)^2$, and
       write the divergence that the second one defines as a sum over $x$.
    1. Adding $c\,(u - 1)$ to $f$ does not change the divergence.
       Show that $D_f$ is symmetric whenever
       $f^\diamond(u) = f(u) + c\,(u - 1)$ for some constant $c$. Check this
       condition for the rows of :numref:`tab_mdl-f-div-generators` that are
       marked symmetric, and show that
       $D_{f_\alpha}(Q\|P) = D_{f_{1-\alpha}}(P\|Q)$ for the
       function :eqref:`eq_mdl-alpha-generator`. Use this identity to
       confirm the entry of the $\alpha$-divergence row in the "Symmetric?"
       column.
    1. Expand :eqref:`eq_mdl-js-def` in terms of the ratio $u = p/q$ and
       recover the Jensen--Shannon
       function of :numref:`tab_mdl-f-div-generators`.
    1. *Rényi divergence.* For $\alpha > 0$, $\alpha \neq 1$, define
       $D_\alpha(P\|Q) = \frac{1}{\alpha - 1} \log \sum_x p(x)^\alpha\,
       q(x)^{1-\alpha}$ :cite:`Renyi.1961`. Show that
       $\sum_x p^\alpha q^{1-\alpha} = 1 + \alpha(\alpha - 1)\,
       D_{f_\alpha}(P\|Q)$ and that $D_\alpha$ is an increasing function of
       $D_{f_\alpha}$. Deduce that $D_\alpha \geq 0$ and that $D_\alpha$
       satisfies the data-processing inequality :eqref:`eq_mdl-f-div-dpi`.
       Find $\lim_{\alpha \to 0} D_\alpha(P\|Q)$, and explain why it is $0$
       whenever $p$ and $q$ are positive on the same outcomes. Why does the
       Rényi family therefore *not* reach reverse KL as $\alpha \to 0$, while
       $D_{f_\alpha}$ does?
   :exlabel:`ex_mdl-divergences-distances-recover-kl-reverse-kl`

1. **Local quadratic regime.** The proposition
   :eqref:`eq_mdl-f-div-local` bounds the gap between $D_f$ and
   $\tfrac12 f''(1)\, \chi^2$.
    1. Verify $f''(1) = \tfrac{1}{4}$ for the Jensen--Shannon function and
       $f''(1) = 1$ for both KL functions. Problem
       :exref:`ex_mdl-information-theory-gaussian-kl`b of
       :numref:`sec_mdl-information_theory` finds the order at which the two
       directions of the Gaussian KL divergence first differ. Which property
       of the KL and reverse-KL functions explains that order? Rerun the
       local-quadratic cell with $t = 0.001$ added, and explain from the
       proposition's error term the rate at which each ratio approaches $1$.
    1. The total-variation function $\tfrac{1}{2}|u - 1|$ has no second
       derivative at $1$. Let $P_t = Q + t\,(P - Q)$ for
       $t \in (0, 1]$. How do $\textrm{TV}(P_t, Q)$ and
       $\chi^2(P_t\|Q)$ scale with $t$? Explain from this why Pinsker's
       inequality :eqref:`eq_mdl-pinsker` bounds TV by the square root of
       KL and not by a multiple of KL.
    1. Let $P$ and $Q$ be coins with heads probabilities $b$ and
       $b + \delta$. Use the proposition to show that
       $D_{\textrm{KL}}(P\|Q) = \delta^2 / \big(2 b (1 - b)\big) + O(\delta^3)$.
       Problem :exref:`ex_mdl-information-theory-two-part-coin`a of
       :numref:`sec_mdl-coding-compression` obtains the same expansion
       directly from the Bernoulli KL divergence. The factor
       $1/\big(b(1-b)\big)$ is the Fisher information of the
       coin (:numref:`sec_mdl-maximum_likelihood`). Find the limit of
       $\textrm{TV}/\sqrt{D_{\textrm{KL}}/2}$ as $\delta \to 0$, and the
       values of $b$ for which Pinsker's inequality is tight in this limit.
       The coin experiment that follows Pinsker's inequality takes
       $b = \tfrac{1}{2}$ and prints a ratio of $0.999999$ at
       $\delta = 10^{-3}$. Why does the ratio approach $1$ at the rate
       $\delta^2$ and not $\delta$?
   :exlabel:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`

1. **Conjugates and critics.** The f-GAN bound :eqref:`eq_mdl-f-gan-bound`
   needs the conjugate $f^*$ of the divergence function.
    1. Derive the conjugate $f^*(t) = t + t^2/4$ of the $\chi^2$ function
       $f(u) = (u - 1)^2$, taking the supremum over $u \in \mathbb{R}$, and
       its optimal critic $T^\star = 2(p/q - 1)$. Redo the computation over
       the function's domain $u \in (0, \infty)$, and show that the
       supremum is $t + t^2/4$ for $t \geq -2$ and $-1$ for $t \leq -2$.
    1. Use the $\mathbb{R}$-conjugate. Show that for every critic $T$ the
       gap between $\chi^2(P\|Q)$ and the bound equals
       $\tfrac{1}{4} \mathbb{E}_Q\big[(T - T^\star)^2\big]$. Deduce the
       <!-- WRITER: the experiment no longer scales T* (it now trains the critic by gradient ascent); reword: e.g. derive (2c - c^2) chi^2 here, and/or show that the ascent of the cell contracts T - T* by (1 - q/2) per step, so the gap falls like a geometric series. -->
       formula $\chi^2(P\|Q)\,(2c - c^2)$ for the critics $T = c\,T^\star$
       that the experiment with scaled critics prints, and check
       the gap formula on three critics of your own.
    1. Compute the conjugate of the total-variation function
       $f(u) = \tfrac{1}{2}|u - 1|$ over $u \in (0, \infty)$. Show that
       critics with values in $[-\tfrac{1}{2}, \tfrac{1}{2}]$ suffice to
       maximize the bound, and conclude that TV is the IPM
       :eqref:`eq_mdl-ipm-def` over $\{g : \|g\|_\infty \leq \tfrac{1}{2}\}$.
       Which critic attains it?
   :exlabel:`ex_mdl-divergences-distances-derive-convex-conjugate`

1. **Cost of dropping a mode.** A model $Q$ misses part of a data
   distribution $P$.
    1. Let $P = (1 - w)\, Q + w R$ with $0 < w < 1$, where $R$ is a
       distribution on a set to which $Q$ assigns probability zero. Compute
       $D_{\textrm{KL}}(P\|Q)$, $D_{\textrm{KL}}(Q\|P)$,
       $\textrm{TV}(P, Q)$ and $d_{\mathrm{H}}^2(P, Q)$ as functions of $w$.
       Which direction of KL penalizes the missing mass? How do the finite
       divergences behave as $w \to 0$?
    1. Now let
       $P = w_1 \mathcal{N}(m_1, s^2) + w_2 \mathcal{N}(m_2, s^2)$ with
       separation $\Delta = |m_2 - m_1|$, and let the model
       $Q = \mathcal{N}(m_1, s^2)$ keep only the first component. Show that
       $D_{\textrm{KL}}(Q\|P) \leq \log(1/w_1)$ for every $\Delta$, with
       equality in the limit $\Delta \to \infty$, and relate the limit to
       (a). Use the bound to explain the two reverse-KL values, $0.356$ and
       $1.202$ nats, that the fit of a Gaussian to the mixture prints.
    1. Assume that the components are well separated, so that almost every
       $x$ has non-negligible density under at most one of them. Show that
       $D_{\textrm{KL}}(P\|Q) \approx w_2 \Delta^2 / (2 s^2)
       + w_1 \log w_1 + w_2 \log w_2$. Evaluate the approximation for
       the mixture of that experiment, check it
       with `kl(log_p(x), log_q(x, [-2.0, np.log(0.6)]), x)`, and compare it
       with the value $0.558$ nats of $D_{\textrm{KL}}(P\|Q)$ at the forward
       fit `th_fwd`.
   :exlabel:`ex_mdl-divergences-distances-mode-dropping`

1. **Metrics among divergences.** The last column of
   :numref:`tab_mdl-f-div-generators` records which divergences are
   metrics.
    1. Prove the triangle inequality for TV from its half-$\ell_1$ form.
    1. Show that the Hellinger distance $d_{\mathrm{H}} = \sqrt{d_{\mathrm{H}}^2}$
       is the Euclidean distance between the vectors
       $\big(\sqrt{p(x)}\big)_x$ and $\big(\sqrt{q(x)}\big)_x$, so that
       $d_{\mathrm{H}}$ is a metric. Find three distributions on two outcomes
       for which both $d_{\mathrm{H}}^2$ and JS violate the triangle
       inequality.
    1. Use :eqref:`eq_mdl-w1-cdf` to show that $W_1$ is a metric on the
       distributions on $\mathbb{R}$ with finite first moments.
   :exlabel:`ex_mdl-divergences-distances-prove-that-total-variation`

1. [code] **Restricted critics.** ● The bound :eqref:`eq_mdl-f-gan-bound`
   is tight only when the critic class contains the optimal critic. Take
   the $\chi^2$ function with the conjugate $f^*(t) = t + t^2/4$, and
   critics $T_{\boldsymbol{\theta}}(x) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(x)$
   that are linear in a feature vector $\boldsymbol{\phi}(x) \in \mathbb{R}^m$.
   Write $\mathbf{m}_P = \mathbb{E}_P[\boldsymbol{\phi}]$,
   $\mathbf{m}_Q = \mathbb{E}_Q[\boldsymbol{\phi}]$ and
   $\boldsymbol{\Sigma} = \mathbb{E}_Q[\boldsymbol{\phi} \boldsymbol{\phi}^\top]$,
   assumed invertible.
    1. Show that the bound is a concave quadratic in $\boldsymbol{\theta}$,
       maximized at
       $\boldsymbol{\theta}^\star = 2 \boldsymbol{\Sigma}^{-1} (\mathbf{m}_P - \mathbf{m}_Q)$,
       with the value
       $(\mathbf{m}_P - \mathbf{m}_Q)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{m}_P - \mathbf{m}_Q)$.
    1. For affine critics, $\boldsymbol{\phi}(x) = (1, x)$, show that the
       value is $(\mathbb{E}_P[X] - \mathbb{E}_Q[X])^2 / \textrm{Var}_Q(X)$.
       What does the restricted bound certify for $P = \mathcal{N}(0, 1)$
       and $Q = \mathcal{N}(0, 4)$?
    1. Use the features $\phi_k(x) = \mathrm{He}_k\big((x - \mu_Q)/\sigma_Q\big)$
       for $k = 0, \ldots, d$, where $\mathrm{He}_k$ are the probabilists'
       Hermite polynomials (`numpy.polynomial.hermite_e.hermevander`) and
       $\mu_Q$, $\sigma_Q$ are the mean and standard deviation of $Q$.
       Compute $\mathbf{m}_P$, $\mathbf{m}_Q$ and $\boldsymbol{\Sigma}$ by
       quadrature on a grid, and plot the value against $d \leq 10$ for the
       pairs $\big(\mathcal{N}(1, 1), \mathcal{N}(0, 1)\big)$ and
       $\big(\mathcal{N}(0, 1), \mathcal{N}(0, 4)\big)$, together with
       their exact divergences $\chi^2 = e - 1$ and $4/\sqrt{7} - 1$. For the
       first pair the values are partial sums of a familiar series. Explain
       why from the generating function
       $e^{xt - t^2/2} = \sum_k \mathrm{He}_k(x)\, t^k / k!$ and the
       orthogonality $\mathbb{E}[\mathrm{He}_j(Z)\, \mathrm{He}_k(Z)] = k!\, \delta_{jk}$
       for $Z \sim \mathcal{N}(0, 1)$.
    1. For the second pair, replace the expectations by averages over
       $n = 200$ samples from each distribution and fit
       $\boldsymbol{\theta}^\star$ on these samples. Evaluate the bound on
       the same samples and on $10^5$ fresh samples, averaged over $200$
       repetitions, for $d \in \{2, 4, 6, 8, 10\}$. Which of the two
       evaluations estimates a valid lower bound on $\chi^2$? Why does the
       other one exceed the restricted value of (c) and, for large $d$, even
       $\chi^2$ itself?
   :exlabel:`ex_mdl-divergences-distances-restricted-critics`

1. **Hellinger bounds and testing.** ● Pinsker's inequality is vacuous
   once $D_{\textrm{KL}} > 2$. Let
   $\textrm{BC}(P, Q) = \sum_x \sqrt{p(x) q(x)} = 1 - \tfrac{1}{2} d_{\mathrm{H}}^2(P, Q)$
   be the Bhattacharyya coefficient.
    1. Prove that $\tfrac{1}{2} d_{\mathrm{H}}^2(P, Q) \leq \textrm{TV}(P, Q)$,
       using $|p - q| = |\sqrt{p} - \sqrt{q}|\,(\sqrt{p} + \sqrt{q})$.
    1. Prove that
       $\textrm{TV}(P, Q) \leq d_{\mathrm{H}}(P, Q) \sqrt{1 - \tfrac{1}{4} d_{\mathrm{H}}^2(P, Q)}$
       with the Cauchy--Schwarz inequality of
       :numref:`sec_mdl-geometry-linear-algebraic-ops`, and rewrite the
       bound as $\textrm{TV}^2 \leq 1 - \textrm{BC}^2$.
    1. Use Jensen's inequality to show that
       $\textrm{BC}(P, Q) \geq e^{-D_{\textrm{KL}}(P\|Q)/2}$, and conclude
       the Bretagnolle--Huber inequality
       $\textrm{TV}(P, Q) \leq \sqrt{1 - e^{-D_{\textrm{KL}}(P\|Q)}}$
       :cite:`Bretagnolle.Huber.1978,Canonne.2022`. For which values of
       $D_{\textrm{KL}}$ is it tighter than :eqref:`eq_mdl-pinsker`?
    1. A test sees $n$ independent samples, all from $P$ or all from $Q$,
       each case with probability $\tfrac{1}{2}$. Use (c) and the
       additivity of KL over independent samples to show that every test
       errs with probability at least $\tfrac{1}{4} e^{-n D_{\textrm{KL}}(P\|Q)}$.
       How many samples does an error probability of at most $\eta$
       require? Why can Pinsker's inequality not give a bound of this kind?
   :exlabel:`ex_mdl-divergences-distances-hellinger-bretagnolle-huber`

[Discussions](https://d2l.discourse.group/t/36835)

<!-- slides -->

<!-- WRITER/SLIDES: deck cut at the split seams (code step). Slides on MMD, W1, Sinkhorn moved to mdl-test-functions-transport.md, slides on scores, Stein, KSD and the wrap-up to mdl-scores.md; TV and Pinsker now come before the critic slides under their own divider. Still to do: goals slide (drop the W1 and score bullets), 'Skips' line, 'Generators' title, a closer for this deck. -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.3]{.kicker}

f-Divergences<br>**one convex function of the density ratio**
:::
:::

::: {.slide title="Jensen–Shannon Cannot Tell a Near Miss from a Far Miss"}
[The problem]{.kicker}

Data at $\delta_0$, model at $\delta_d$:

| | $d=0.01$ | $d=1$ | $d=100$ |
|:---|---:|---:|---:|
| KL | $\infty$ | $\infty$ | $\infty$ |
| Jensen–Shannon | $\log 2$ | $\log 2$ | $\log 2$ |
| total variation | $1$ | $1$ | $1$ |
| $\textrm{MMD}^2$ (RBF, $\ell=1$) | $0.0001$ | $0.79$ | $2.00$ |
| Wasserstein $W_1$ | $0.01$ | $1$ | $100$ |

::: {.d2l-note .warn}
The GAN loss at the best discriminator, $2\,\textrm{JS}-\log 4$, is $0$ in all three columns, so its gradient is zero.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- pick a divergence by what you can compute and what it must detect;
- prove $D_f \ge 0$ for a whole family with one Jensen step;
- bound total variation by $\sqrt{\textrm{KL}/2}$;
- compute $W_1$ on the line as the area between two CDFs;
- write any f-divergence as a game against a critic;
- test an unnormalized model with its score.

**Running examples:** $P=(0.6,0.3,0.1)$ vs $Q=(0.2,0.5,0.3)$; point masses; a six-point pair.

**The idea:** compare ratios, averages, transport costs, or slopes.
:::

::: {.slide title="Ratios, Averages, Transport, Slopes"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Entropy, Cross-Entropy, and KL | surprise, KL, the loss floor |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| **f-Divergences** | **one convex function of the ratio** |
| Test Functions and Transport | averages and moving mass |
| Score-Based Comparisons | slopes, no normalizer |
| Mutual Information | dependence and what it limits |
| Estimating Mutual Information | lower bounds from samples |
:::

::: {.col .narrow}
- **Uses:** KL, Jensen's inequality.
- **Skips:** Rényi divergences, transport in many dimensions beyond Sinkhorn, estimation error.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[f-divergences]{.dtitle}

[one convex function of the ratio]{.dsub}
:::
:::

::: {.slide title="Any Convex f with f(1) = 0 Gives a Divergence"}
[The template]{.kicker}

$D_{\textrm{KL}}(P\|Q)=\mathbb{E}_Q[u\log u]$ with $u=p/q$; replace $u\log u$ by convex $f$, $f(1)=0$:

$$D_f(P\|Q)=\mathbb{E}_{x\sim Q}\big[f\big(p(x)/q(x)\big)\big]$$

::: {.d2l-note .rule}
$\mathbb{E}_Q[u]=\sum_x p(x)=1$, so by Jensen $D_f\ge f(\mathbb{E}_Q u)=f(1)=0$. $\blacksquare$
:::

- Disjoint supports: $D_f(\delta_0\|\delta_d)=f(0)+f'(\infty)$ for every $d\ne0$.
- TV $=1$, JS $=\log2$, KL $=\infty$: constants, whatever $d$ is.
:::

::: {.slide title="Generators Disagree Far from u = 1 and Agree Near It"}
[The gallery]{.kicker}

::: {.cols .vc}
::: {.col .fig}
@fig:mdl-it-f-div-generators
:::

::: {.col}
- Near $P=Q$: $D_f\approx\tfrac12 f''(1)\,\chi^2$, $f''(1)=1,1,2,\tfrac12,\tfrac14$.
- Running pair: KL $0.396$ / $0.365$ (both orders).
- $\chi^2$ $1.013$ / $0.800$; JS $0.091 < \log 2$.
:::
:::
:::

::: {.slide title="Check: TV and χ² of the Running Pair"}
[Check]{.kicker}

$P=(0.6,0.3,0.1)$, $Q=(0.2,0.5,0.3)$. Compute
$\textrm{TV}=\tfrac12\sum|p-q|$ and $\chi^2=\sum(p-q)^2/q$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\textrm{TV}=\tfrac12(0.4+0.2+0.2)=0.4$;
$\chi^2=\tfrac{0.16}{0.2}+\tfrac{0.04}{0.5}+\tfrac{0.04}{0.3}=1.013$.
:::
:::
:::

::: {.slide title="Forward KL Covers Both Modes; Reverse KL Picks One"}
[Forward and reverse KL]{.kicker}

Fit $\mathcal N(\mu,\sigma^2)$ to $0.7\,\mathcal N(-2,0.6^2)+0.3\,\mathcal N(2,0.6^2)$:

@!divergences-distances-fwd-rev-kl

Reverse KL charges only $\log(1/0.7)=0.357$ for the dropped mode.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Total variation and Pinsker]{.dtitle}

[tests, KL, and processing]{.dsub}
:::
:::

::: {.slide title="TV Is the Best Single-Sample Test, and KL Bounds It"}
[Total variation]{.kicker}

::: {.cols .vc}
::: {.col}
- $\textrm{TV}=\sup_A|P(A)-Q(A)|=\tfrac12\|p-q\|_1$, attained at $A^\star=\{p>q\}$.
- Guess "$P$" on $A$: success $\tfrac12(1+P(A)-Q(A))\le\tfrac12(1+\textrm{TV})$.
- Pinsker: $\textrm{TV}\le\sqrt{D_{\textrm{KL}}/2}$.
:::

::: {.col .fig}
@fig:mdl-it-tv-area
:::
:::
:::

::: {.slide title="Pinsker Is Two Steps, and Its Constant Is Sharp"}
[Pinsker's inequality]{.kicker}

- **I. Coins.** $\rho(a)=d(a\|b)-2(a-b)^2$ has $\rho(b)=\rho'(b)=0$ and $\rho''=\tfrac{1}{a(1-a)}-4\ge0$.
- **II. Merge.** Collapse $A^\star$ and its complement: data processing, $D_f(P\kappa\|Q\kappa)\le D_f(P\|Q)$.
- 10,000 random pairs: $\textrm{TV}/\sqrt{\textrm{KL}/2}$ never exceeds $1$ (max within a few hundredths of $1$).
- Coins $\tfrac12$ vs $\tfrac12+\delta$, $\delta=0.1, 0.01, 0.001$: $0.9899$, $0.9999$, $0.999999$.

::: {.d2l-note}
No converse: Gaussians 50 apart have TV $\approx1$ but KL $=1250$.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The critic bound]{.dtitle}

[a divergence from samples]{.dsub}
:::
:::

::: {.slide title="Any f-Divergence Is a Game Against a Critic"}
[Variational form]{.kicker}

::: {.cols .vc}
::: {.col}
Fenchel–Young, $f(u)\ge ut-f^*(t)$, at $u=p/q$, $t=T(x)$:

$$D_f(P\|Q)\ge\mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)]$$

- Equality at $T^\star=f'(p/q)$.
- Both terms are sample averages: no densities.
:::

::: {.col .fig}
@fig:mdl-it-f-gan-tangent-bound
:::
:::

[Assumes differentiable $f$, $P\ll Q$, finite expectations.]{.small}
:::

::: {.slide title="A Half-Trained Critic Reports 75% of the Divergence"}
<!-- WRITER/SLIDES: the cell now trains a three-number critic by gradient ascent (bound 0, 0.224, 0.394, 0.704, 0.911, 1.001, 1.013 after 0/1/2/5/10/20/50 steps; 2T* gives 0); the title and the line 'critics T = cT*: bound chi^2 (2c - c^2)' and the note 'c = 2 reports zero' need rewriting to the run. -->
$\chi^2$ generator, critics $T=c\,T^\star$: bound $\chi^2\,(2c-c^2)$.

@!divergences-distances-f-gan-bound

::: {.d2l-note}
Estimates are biased low; $c=2$ reports zero.
:::
:::
