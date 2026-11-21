# f-Divergences
:label:`sec_mdl-divergences-distances`

Throughout this book, whenever we fit a model by maximum likelihood, we measure
how well it matches the data with a single number: the KL divergence. As we saw
at the end of :numref:`sec_mdl-information_theory`, however, KL divergence has a
glaring blind spot: it jumps to infinity the moment a model assigns zero
probability to an event that actually occurs. A classifier rarely runs into this
problem, because a softmax layer never outputs an exact zero. A generative
network that maps a low-dimensional noise vector into image space runs into it
immediately, because its samples lie on a thin surface in image space that can
miss the real data altogether. How else can we compare two distributions, and
which comparisons can still distinguish a model that is nearly right from one
that is far off?

The simplest possible example already reveals what can go wrong. Suppose the
data sit at a single point, $P = \delta_0$, while the model places all its mass
at a single point a distance $d$ away, $Q = \delta_d$. Intuitively, a model at
$d = 0.01$ is nearly right and one at $d = 100$ is far off, so a useful loss
should tell them apart. :numref:`tab_mdl-point-mass-table` shows what five
standard comparisons report.

:Five comparisons of the point masses $\delta_0$ and $\delta_d$. The MMD row uses the RBF kernel with length scale $1$.
:label:`tab_mdl-point-mass-table`

| comparison | $d = 0.01$ | $d = 1$ | $d = 100$ |
|:----|----:|----:|----:|
| KL divergence $D_{\textrm{KL}}(P\|Q)$ | $\infty$ | $\infty$ | $\infty$ |
| Jensen--Shannon divergence | $\log 2$ | $\log 2$ | $\log 2$ |
| total variation distance | $1$ | $1$ | $1$ |
| maximum mean discrepancy $\textrm{MMD}^2$ | $0.0001$ | $0.79$ | $2.00$ |
| Wasserstein distance $W_1$ | $0.01$ | $1$ | $100$ |

Look closely at the table: three of the five rows do not budge as $d$ varies. A
model trained with any of those three receives the same loss at $d = 0.01$ as at
$d = 100$, and therefore zero gradient with respect to $d$. The original
generative adversarial network (GAN) of :numref:`chap_gans` sat in this
predicament, since its objective reduces to the Jensen--Shannon row up to
constants.

Those three frozen rows belong to a single family. KL divergence compares $P$
and $Q$ through the ratio $p/q$ of their densities. If we keep that density
ratio and vary the convex function applied to it, we obtain a broad family
called the *f-divergences*. All of its members share one proof of
non-negativity, one method for estimation from samples, and one structural
reason for being blind to the distance $d$. The remaining two rows, which do
move with $d$, avoid the density ratio altogether; we will study them in
:numref:`sec_mdl-test-functions-transport` (before turning to score-based
comparisons such as Fisher divergence and Stein discrepancy in
:numref:`sec_mdl-scores`, and sample-based mutual-information estimation in
:numref:`sec_mdl-mi-estimation`).

Once the family is in place, we will address three questions that recur
throughout deep learning. First, does it matter which distribution comes first
in KL, the data as in maximum likelihood or the model as in variational
inference (:numref:`sec_mdl-fwd-vs-rev-kl`)? Second, what does a small
divergence actually guarantee to someone trying to tell $P$ from $Q$
(:numref:`sec_mdl-tv-pinsker`)? Third, what can we still compute when both
distributions are known only through samples (:numref:`sec_mdl-f-gan-dual`)?
:numref:`sec_mdl-coding-compression` ended on that last question, and adversarial
training rests on its answer.

Throughout this section, all logarithms are natural, so KL and Jensen--Shannon
divergences are measured in nats. We write $P$ for the data or target
distribution and $Q$ for the model, $p$ and $q$ for their probability mass
functions or densities, and $u = p/q$ for their *density ratio*.

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

We can check every entry of :numref:`tab_mdl-point-mass-table` directly in code.
The next cell represents $\delta_0$ and $\delta_d$ as the probability vectors
$(1, 0)$ and $(0, 1)$ on the two-point support $\{0, d\}$ and evaluates each
comparison from its definition. Three of these definitions follow in this
section and the other two in the next.

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

## The f-Divergence Family

Before generalizing KL divergence, let us pin down the minimum properties that
any comparison between two distributions should satisfy. The f-divergence
family then emerges naturally when we ask which ingredients the non-negativity
proof of KL truly uses.

### Divergences and Metrics
:label:`sec_mdl-what-is-a-divergence`

At a minimum, a "distance between distributions" should vanish when the two
distributions coincide and stay strictly positive when they differ. A
**divergence** is any function of two distributions with this property:

$$
D(P, Q) \geq 0, \qquad D(P, Q) = 0 \quad \textrm{if and only if} \quad P = Q.
$$

Nothing else is required. A **metric**, like the Euclidean distance
$\|\mathbf{x} - \mathbf{y}\|$ between two vectors
(:numref:`sec_linear-algebra`), must additionally be symmetric,
$D(P, Q) = D(Q, P)$, and obey the triangle inequality,
$D(P, R) \leq D(P, Q) + D(Q, R)$. KL divergence is a divergence: Gibbs'
inequality (:numref:`sec_mdl-information_theory`) guarantees that
$D_{\textrm{KL}}(P\|Q) \geq 0$, with equality only when $P = Q$. It fails both
metric axioms, however, and a pair of univariate Gaussians shows both failures
immediately. As we computed in that section, the KL divergence between two
Gaussians is

$$
D_{\textrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\big\|\, \mathcal{N}(\mu_2, \sigma_2^2)\big)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}.
$$
:eqlabel:`eq_mdl-dd-gaussian-kl`

For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(0, 4)$, this formula gives
$D_{\textrm{KL}}(P\|Q) \approx 0.318$ nats in one direction and
$D_{\textrm{KL}}(Q\|P) \approx 0.807$ nats in the other, so KL is not
symmetric. The triangle inequality fails for a different reason. Take three
unit-variance Gaussians centered at $0$, $1$, and $2$: the formula gives
$D_{\textrm{KL}} = \tfrac{1}{2}$ for each adjacent pair, yet $2$ for the outer
pair, and $2 > \tfrac{1}{2} + \tfrac{1}{2}$. Because KL grows like the *square*
of the distance between the means, it behaves like a squared Euclidean distance
rather than a true length.

Neither failure needs repair. Minimizing a divergence uses neither symmetry nor
the triangle inequality, and the asymmetry of KL turns out to be a genuine
modeling choice (:numref:`sec_mdl-fwd-vs-rev-kl`). Still, whenever a proof
needs to chain comparisons from $P$ to $Q$ and from $Q$ to $R$, the triangle
inequality becomes essential. Fortunately, the family we are about to construct
includes true metrics as well.

### The f-Divergence Template
:label:`sec_mdl-f-divergences`

Gibbs' inequality is what makes KL a valid divergence. Let us inspect its proof
closely, because any other function that shares the same two properties will
yield a divergence too.

First, rewrite KL as an expectation under $Q$ of a function of the density
ratio $u(x) = p(x)/q(x)$:

$$
D_{\textrm{KL}}(P\|Q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
= \sum_x q(x)\, \frac{p(x)}{q(x)} \log\frac{p(x)}{q(x)}
= \mathbb{E}_{x \sim Q}\big[u(x) \log u(x)\big].
$$

Viewed this way, KL weights each outcome by $Q$ and penalizes its density ratio
through the function $u \log u$. Let us check this formula on our running
example from :numref:`sec_mdl-information_theory`, the discrete pair
$P = (0.6, 0.3, 0.1)$ and $Q = (0.2, 0.5, 0.3)$. Here the ratios are $u = 3$,
$0.6$, and $\tfrac13$, weighted by $0.2$, $0.5$, and $0.3$:

$$
0.2\,(3 \log 3) + 0.5\,(0.6 \log 0.6) + 0.3\,\big(\tfrac13 \log \tfrac13\big)
= 0.659 - 0.153 - 0.110 = 0.396 \textrm{ nats},
$$

matching the KL divergence we found earlier.

Second, look at the proof of non-negativity. In
:numref:`sec_mdl-information_theory`, we applied Jensen's inequality once to
$-\log(q/p)$ under $P$. Run instead under $Q$ on the function $u \log u$, that
same one-step argument uses only two facts: the function is convex, and it
equals zero at $u = 1$. Nothing stops us from replacing $u \log u$ by any
function with those two properties. Let $f : (0, \infty) \to \mathbb{R}$ be
convex with $f(1) = 0$, and define $f(0) = \lim_{u \downarrow 0} f(u)$, which
may be $+\infty$. Assume for now that $p(x) = 0$ wherever $q(x) = 0$, written
$P \ll Q$, so that an expectation under $Q$ sees all of $P$. The
**f-divergence** of $P$ with respect to $Q$ is :cite:`Csiszar.1967`

$$
D_f(P\|Q) = \mathbb{E}_{x \sim Q}\!\left[ f\!\left( \frac{p(x)}{q(x)} \right) \right].
$$
:eqlabel:`eq_mdl-f-div-def`

Point by point, the ratio $u = p/q$ measures how much $P$ over- or
under-represents $x$ relative to $Q$, and $f$ assigns a penalty that vanishes
when $u = 1$. We call $f$ the *divergence function*. Much of the literature
calls it the generator of the divergence, and our code below stores these
functions in a dictionary named `generators`. In the prose, however, we reserve
the word *generator* for a network that produces samples, as in
:numref:`sec_gan_objectives`.

To pin down when the divergence equals zero, we need one mild refinement of
convexity: we call $f$ *strictly convex at $1$* if some line through $(1, 0)$
lies strictly below $f$ everywhere else, meaning $f(u) > c\,(u - 1)$ for all
$u \neq 1$ and some slope $c$.

**Proposition (non-negativity).** *For any convex $f$ with $f(1) = 0$ and
$P \ll Q$,*

$$
D_f(P\|Q) \geq 0,
$$

*and if $f$ is strictly convex at $1$, then $D_f(P\|Q) = 0$ if and only if
$P = Q$.*

**Proof.** Jensen's inequality (:numref:`subsec_mdl-jensen`) states that
$\mathbb{E}[f(U)] \geq f(\mathbb{E}[U])$ for any convex $f$ and random variable
$U$. Apply it to $U = u(x)$ with $x \sim Q$. Everything rests on a single
identity: the density ratio averages to one under $Q$, since
$\mathbb{E}_Q[u] = \sum_x q(x)\, p(x)/q(x) = \sum_x p(x) = 1$. Therefore

$$
D_f(P\|Q) = \mathbb{E}_Q[f(u)] \geq f(\mathbb{E}_Q[u]) = f(1) = 0.
$$

For the equality case, take the supporting line $c\,(u - 1)$ lying strictly
below $f$ away from $1$. Then $D_f(P\|Q) \geq \mathbb{E}_Q[c\,(u - 1)] = 0$,
and the gap is the expectation of $f(u) - c\,(u - 1)$, which is strictly
positive wherever $u \neq 1$. Hence $D_f(P\|Q) = 0$ forces $u(x) = 1$ at every
$x$ with $q(x) > 0$, meaning $P = Q$. $\blacksquare$

The identity $\mathbb{E}_Q[u] = 1$ that powered this proof has a second handy
consequence we will use repeatedly: adding any multiple $c\,(u - 1)$ to $f$
leaves every divergence unchanged, because $\mathbb{E}_Q[c\,(u - 1)] = 0$. The
functions $u \log u$ and $u \log u - u + 1$, for example, define the same KL
divergence.

So far we have assumed $P \ll Q$, yet our opening point-mass example violates
that condition. A generative model starts training in the same regime, with its
samples and the data occupying disjoint regions, so we must extend $D_f$ to
handle zeros in $q$. Outcomes with $p(x) = q(x) = 0$ contribute nothing. Where
$p(x) = 0 < q(x)$, the contribution is simply $q(x)\, f(0)$, as the definition
already says. Where $q(x) = 0 < p(x)$, the ratio blows up, so we define the
contribution as the limit of $q\, f(p/q)$ as $q \downarrow 0$. Substituting
$t = p/q$ rewrites this as $p\, f(t)/t$, whose limit is $p(x)\, f'(\infty)$
with $f'(\infty) := \lim_{t \to \infty} f(t)/t$. For a convex function $f$, the
secant $f(t)/t$ and the derivative $f'(t)$ share the same limit at infinity,
which explains the notation. Either $f(0)$ or $f'(\infty)$ may be $+\infty$.

Now apply this rule to the point masses $\delta_0$ and $\delta_d$ for any
$d \neq 0$. At $x = 0$ we have $q = 0 < p = 1$, and at $x = d$ we have
$p = 0 < q = 1$, so

$$
D_f(\delta_0 \| \delta_d) = f(0) + f'(\infty) \qquad \textrm{for every } d \neq 0.
$$

Notice that $d$ has vanished from the right-hand side. This tells us something
far stronger than :numref:`tab_mdl-point-mass-table` alone showed: *every*
f-divergence is constant on disjoint supports. It cannot distinguish a near
miss from a distant one, so a model trained on it gets zero gradient from the
distance separating the supports. For KL, $f(u) = u \log u$ gives $f(0) = 0$
and $f'(\infty) = \lim_{t \to \infty} \log t = \infty$, reproducing the first
row of :numref:`tab_mdl-point-mass-table`. The other two frozen rows follow as
soon as we write down their divergence functions.

### Common Divergence Functions

Each row of :numref:`tab_mdl-f-div-generators` corresponds to one choice of
$f$, and :numref:`fig_mdl-f-div-generators` plots six of these curves side by
side. The column $f''(1)$, recording the curvature at $u = 1$, will play a
central role in the next subsection.

:Common divergence functions. Each row gives the function $f$ of one divergence, its curvature at $u = 1$, whether the divergence is symmetric, and whether it (or its square root) is a metric.
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

![Six divergence functions, with the smooth curves shifted by $f'(1)(u-1)$ so that each touches zero tangentially at $u=1$ without changing its divergence. Near $u=1$ the smooth curves are parabolas with the curvatures $1, 1, 2, \tfrac12, \tfrac14$ listed in the table, while total variation has a sharp corner instead. Away from $u=1$ they penalize disagreement very differently: reverse KL blows up as $u \to 0$, KL and $\chi^2$ grow superlinearly, and total variation grows linearly.](../img/mdl-it-f-div-generators.svg)
:label:`fig_mdl-f-div-generators`

**Reverse KL** uses $f(u) = -\log u$, because
$\mathbb{E}_Q[-\log(p/q)] = \mathbb{E}_Q[\log(q/p)] = D_{\textrm{KL}}(Q\|P)$.
Swapping the arguments of KL is therefore just another choice of $f$, so both
directions sit side by side in the same table. Exercise
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`a gives the general
recipe for swapping the arguments of any f-divergence.

**Total variation** uses $f(u) = \tfrac{1}{2}|u - 1|$, which simplifies to
$\sum_x q(x)\, \tfrac12 |p(x)/q(x) - 1| = \tfrac{1}{2}\sum_x |p(x) - q(x)|$.
We write it as $\textrm{TV}(P, Q)$. It is half the $\ell_1$ distance between
the probability vectors; on our running pair,
$\textrm{TV} = \tfrac12(0.4 + 0.2 + 0.2) = 0.4$. We will unpack what this
number means for distinguishing $P$ from $Q$ in :numref:`sec_mdl-tv-pinsker`.
On the point masses, its divergence function gives
$f(0) + f'(\infty) = \tfrac12 + \tfrac12 = 1$, accounting for the third row of
:numref:`tab_mdl-point-mass-table`.

**Pearson's $\chi^2$ divergence** uses the parabola $f(u) = (u - 1)^2$, giving
$\chi^2(P\|Q) = \sum_x q(x)\,\big(p(x)/q(x) - 1\big)^2
= \sum_x (p(x) - q(x))^2 / q(x)$. On the running pair, it evaluates to
$0.16/0.2 + 0.04/0.5 + 0.04/0.3 = 1.0133$. Notice how heavily a mismatch is
weighted where $Q$ is small: the first outcome alone, where $q = 0.2$,
contributes $0.8$ of the total.

The **Jensen--Shannon divergence** :cite:`Lin.1991` symmetrizes KL by comparing
both distributions against their even mixture $M = \tfrac{1}{2}(P + Q)$ (with
mass function $m$):

$$
\textrm{JS}(P, Q)
= \frac{1}{2} D_{\textrm{KL}}(P \,\|\, M)
+ \frac{1}{2} D_{\textrm{KL}}(Q \,\|\, M).
$$
:eqlabel:`eq_mdl-js-def`

Because $M$ puts mass wherever either $P$ or $Q$ does, the ratios $p/m$ and
$q/m$ can never blow up: $p/m = 2p/(p + q) \leq 2$, so each KL term is at most
$\log 2$. Hence JS is symmetric and bounded, $0 \leq \textrm{JS}(P, Q) \leq \log 2$,
and it hits its ceiling $\log 2$ only when $P$ and $Q$ have disjoint supports
(where $p/m = 2$ wherever $p > 0$). Remarkably, its square root is a true
metric :cite:`Endres.Schindelin.2003`. Writing :eqref:`eq_mdl-js-def` in terms
of the ratio $u$ yields the divergence function in the table (Exercise
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`c), which satisfies
$f(0) = f'(\infty) = \tfrac12 \log 2$. Their sum is $\log 2$, the second row of
:numref:`tab_mdl-point-mass-table`. Our single identity
$D_f(\delta_0\|\delta_d) = f(0) + f'(\infty)$ has now explained all three
frozen rows.

The **squared Hellinger divergence** is
$d_{\mathrm{H}}^2(P, Q) = \sum_x \big(\sqrt{p(x)} - \sqrt{q(x)}\big)^2$. Its
square root $d_{\mathrm{H}}$ is simply the Euclidean distance between the
vectors $\sqrt{p}$ and $\sqrt{q}$, which immediately makes $d_{\mathrm{H}}$ a
metric (Exercise
:exref:`ex_mdl-divergences-distances-prove-that-total-variation`b).

The final row of the table is a continuous family inside the family, the
**$\alpha$-divergences** :cite:`Amari.2016`, generated by

$$
f_\alpha(u)=\frac{u^\alpha-\alpha u+\alpha-1}{\alpha(\alpha-1)}.
$$
:eqlabel:`eq_mdl-alpha-generator`

For every $\alpha \neq 0, 1$, this function is convex with $f_\alpha(1) = 0$
and curvature $f_\alpha''(1) = 1$. As $\alpha \to 1$ it approaches
$u\log u - u + 1$, and as $\alpha \to 0$ it approaches $-\log u + u - 1$. Since
adding a multiple of $u - 1$ changes no divergence, the two limits recover
forward and reverse KL. Halfway between them, at $\alpha = \tfrac12$, we get
$2(\sqrt{u} - 1)^2$ (twice squared Hellinger), while $\alpha = 2$ gives half of
$\chi^2$. A single parameter $\alpha$ therefore interpolates smoothly from
reverse KL through Hellinger to forward KL and on to $\chi^2$ (Exercise
:exref:`ex_mdl-divergences-distances-recover-kl-reverse-kl`d connects this
family to Rényi's divergence).

Let us evaluate the first six rows on our running pair in both argument orders.
KL should print $0.396$ and $0.365$ nats, matching
:numref:`sec_mdl-information_theory`, while TV and $\chi^2(P\|Q)$ should match
the hand calculations above.

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

As expected, the KL row is asymmetric, and the reverse-KL row is simply the KL
row with its columns swapped. The $\chi^2$ row is asymmetric too ($1.0133$
against $0.8000$). By contrast, squared Hellinger ($0.1862$), TV ($0.4$), and
JS are symmetric, with JS at $0.0911$ nats sitting well below its ceiling of
$\log 2 \approx 0.693$. Six divergences have looked at the same pair of
distributions and returned values spanning an order of magnitude, from $0.09$
to $1.01$. Yet if we multiply KL by $2$, squared Hellinger by $4$, and JS by
$8$, the three rescaled values are $0.79$, $0.74$, and $0.73$. Why would three
quite different functions of the density ratio nearly agree, and where did the
factors $2$, $4$, and $8$ come from?

### The Local Quadratic

The curves in :numref:`fig_mdl-f-div-generators` behave very differently far
from $u = 1$. Near $u = 1$, however, they collapse into a single function up to
a constant factor, which is perhaps the most surprising property of the entire
family. Nothing later in this section depends on this subsection, so you can
skip it on a first reading.

Because adding a linear term $c\,(u - 1)$ leaves every divergence unchanged,
neither the value $f(1) = 0$ nor the slope $f'(1)$ affects $D_f$. The first
property that distinguishes one smooth divergence function from another is its
curvature $f''(1)$. Near $u = 1$, every smooth $f$ looks like the parabola
$\tfrac12 f''(1)(u - 1)^2$, and the expectation of $(u - 1)^2$ under $Q$ is
none other than the $\chi^2$ divergence,
$\mathbb{E}_Q[(u - 1)^2] = \sum_x (p(x) - q(x))^2/q(x)$. Let
$r = \max_x |u(x) - 1|$ measure how far the density ratio strays from $1$.

**Proposition (local quadratic).** *Let $P \ll Q$, let $0 < r < 1$, let $f$ be
three times differentiable on $[1 - r, 1 + r]$ with $|f'''| \leq B$ there, and
let $|u(x) - 1| \leq r$ for every $x$ with $q(x) > 0$. Then*

$$
\left| D_f(P\|Q) - \frac{f''(1)}{2}\, \chi^2(P\|Q) \right|
\leq \frac{B r}{6}\, \chi^2(P\|Q).
$$
:eqlabel:`eq_mdl-f-div-local`

**Proof.** Since $f(1) = 0$, Taylor's theorem with remainder gives
$f(u) = f'(1)(u - 1) + \tfrac12 f''(1)(u - 1)^2 + R(u)$ with
$|R(u)| \leq B |u - 1|^3 / 6$. Take $\mathbb{E}_Q$ term by term. The linear
term vanishes because $\mathbb{E}_Q[u - 1] = 0$, eliminating the only
first-order difference between one $f$ and another. The quadratic term becomes
$\tfrac12 f''(1)\, \chi^2(P\|Q)$, and the remainder satisfies
$\mathbb{E}_Q|R| \leq \tfrac{B}{6}\,\mathbb{E}_Q|u - 1|^3 \leq \tfrac{B r}{6}\, \mathbb{E}_Q(u - 1)^2 = \tfrac{B r}{6}\,\chi^2$.
$\blacksquare$

In short, $D_f \approx \tfrac12 f''(1)\, \chi^2$ with a relative error
proportional to $r$: whenever $P$ is close to $Q$, all smooth f-divergences are
scalar multiples of one another. The factors $2$, $4$, and $8$ we applied to
KL, squared Hellinger, and JS are simply $2/f''(1)$, and the target all three
products approximate is $\chi^2(P\|Q) = 1.013$. On our running pair they still
fall about a quarter short, which is no surprise: the largest ratio is $3$, so
$r = 2$, outside the proposition's hypothesis $r < 1$. To watch the quadratic
approximation tighten, let us slide $P$ toward $Q$ along the line segment
$P_t = Q + t\,(P - Q)$. This scales every $u - 1$ by $t$, giving $r = 2t$. For
four smooth divergences, we print the ratio
$D_f(P_t\|Q) / \big(\tfrac12 f''(1)\, \chi^2(P_t\|Q)\big)$, which should
converge to $1$ at a rate proportional to $t$.

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

At $t = 0.5$, the four divergences sit at roughly $75\%$ to $85\%$ of their
quadratic approximations; at $t = 0.1$ they exceed $91\%$, and at $t = 0.01$
they exceed $99\%$. Between the last two rows $t$ drops tenfold and each
shortfall drops nearly tenfold too, matching the linear factor $r$ in the
proposition. The bound itself is conservative: for KL at $t = 0.1$ we have
$r = 0.2$ and $|f'''(u)| = 1/u^2 \leq 1/0.8^2$, which allows a relative error
of about $10\%$, whereas the actual gap is only $4\%$. The takeaway for
optimization is practical: smooth f-divergences disagree meaningfully only on
distant distributions, which is the regime early in training when the model is
still far from the data.

Inside a parametric family, this common local quadratic is governed by the
Fisher information matrix
$\mathbf{I}(\boldsymbol{\theta}) = \mathbb{E}\big[\nabla_{\boldsymbol{\theta}}
\log q_{\boldsymbol{\theta}}\, \nabla_{\boldsymbol{\theta}} \log
q_{\boldsymbol{\theta}}^\top\big]$ of :numref:`sec_mdl-maximum_likelihood`. For
a small parameter step $\mathbf{h}$, the density ratio of
$q_{\boldsymbol{\theta} + \mathbf{h}}$ to $q_{\boldsymbol{\theta}}$ expands as
$u \approx 1 + \mathbf{h}^\top \nabla_{\boldsymbol{\theta}} \log q_{\boldsymbol{\theta}}$.
Therefore
$\chi^2(q_{\boldsymbol{\theta} + \mathbf{h}}\| q_{\boldsymbol{\theta}}) =
\mathbb{E}[(u - 1)^2] \approx \mathbf{h}^\top \mathbf{I}(\boldsymbol{\theta})\,
\mathbf{h}$, so every smooth $D_f$ has Hessian
$f''(1)\, \mathbf{I}(\boldsymbol{\theta})$ at $\mathbf{h} = \mathbf{0}$. The
natural gradient of :numref:`sec_mdl-adaptive-stochastic-methods`, though
usually motivated by KL divergence, would yield the same update under any
smooth f-divergence. Total variation is the lone exception in our table: its
divergence function has a corner at $u = 1$ and no second derivative there, and
Exercise :exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`b
explores how that changes its behavior as $P$ approaches $Q$.

## Forward and Reverse KL
:label:`sec_mdl-fwd-vs-rev-kl`

Maximum likelihood, the training criterion behind most models in this book,
minimizes KL divergence with the data in the first argument. Variational
inference minimizes KL divergence with the model in the first argument. Both
orders appear as separate rows in :numref:`tab_mdl-f-div-generators`, and when
we fit a model $Q_{\boldsymbol{\theta}}$ (with density $q_{\boldsymbol{\theta}}$)
to a target $P$ that lies outside the model family, they push the model in
opposite directions:

* **Forward KL**, $D_{\textrm{KL}}(P \,\|\, Q_{\boldsymbol{\theta}}) = \mathbb{E}_P[\log(p/q_{\boldsymbol{\theta}})]$,
  takes its expectation over the *truth*. Whenever $q_{\boldsymbol{\theta}}(x) \to 0$
  at a point where $p(x) > 0$, the divergence blows up. Forward KL is therefore
  *zero-avoiding*, or *mode-covering*: the model must stretch to cover every
  mode of the data, that is, every region where $p$ concentrates, even if that
  forces it to spill mass across empty regions $P$ never visits.
* **Reverse KL**, $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \,\|\, P) = \mathbb{E}_{Q_{\boldsymbol{\theta}}}[\log(q_{\boldsymbol{\theta}}/p)]$,
  takes its expectation over the *model*. Here the model is penalized heavily
  for placing mass where the truth has none, while any mode the model ignores
  contributes nothing to the objective. Reverse KL is therefore *zero-forcing*,
  or *mode-seeking*: the model locks onto the portion of the target it can
  explain well.

In practice, the choice of direction is often settled by what we can actually
compute. Forward KL requires samples from $P$ and the density of the model, and
minimizing it is just maximum likelihood: :numref:`subsec_mdl-nll-crossentropy`
shows that the average negative log-likelihood equals the cross-entropy from
the empirical distribution to the model, which differs from forward KL only by
a constant independent of $\boldsymbol{\theta}$. Reverse KL requires samples
from the model and the density of the target, and it works even when the target
density is known only up to a normalizing constant. Variational inference
(:numref:`sec_mdl-latent-em-elbo`) sits in this setting: its target $P$ is a
posterior with an intractable normalizer, so it minimizes
$D_{\textrm{KL}}(Q_{\boldsymbol{\theta}}\|P)$ over tractable approximations
$Q_{\boldsymbol{\theta}}$.

When samples and densities are available on both sides, however, the direction
becomes a genuine design choice. A student network distilled from a teacher
(:numref:`sec_lm_distillation`) spreads out to cover everything the teacher
might say when the teacher comes first, and concentrates on the teacher's
likeliest outputs when the student comes first. Similarly, the KL penalty in
:numref:`sec_regularized` places the learned policy first, keeping it away from
actions that the reference policy never takes.

If the model family contains $P$, both directions recover $P$. They part ways
when the model is *misspecified*, and the clearest illustration is fitting a
single Gaussian to a bimodal mixture. For forward KL, no optimization is even
needed. Because forward KL equals a constant minus $\mathbb{E}_P[\log q(X)]$,
its minimizer is the maximum-likelihood Gaussian with $P$ replacing the
empirical distribution, and maximum likelihood in an exponential family matches
the expected sufficient statistics (:numref:`subsec_mdl-moment-matching`):
$\mu^* = \mathbb{E}_P[X]$ and $(\sigma^*)^2 = \textrm{Var}_P(X)$, provided the
variance is finite. Reverse KL has no such closed-form solution, and it can
have a separate local minimum near each mode.

For the bimodal mixture
$P = 0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$, we can predict
the forward fit in advance. Its mean is $0.7(-2) + 0.3(2) = -0.8$. Its variance
is the within-component variance plus the between-component variance, which for
weights $w_1$, $w_2$ and means separated by $\Delta$ equals $w_1 w_2 \Delta^2$,
giving $0.36 + 0.7 \cdot 0.3 \cdot 4^2 = 3.72$. For the reverse fit, we run a
derivative-free optimizer over $\boldsymbol{\theta} = (\mu, \log\sigma)$ from
two starting points, once at the origin and once near the smaller mode,
evaluating each KL by numerical quadrature on a grid. For each fit, the next
cell reports the mean, standard deviation, and both divergences.

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

print(f'{"fit":26s}{"mu":>8s}{"sigma":>7s}'
      f'{"forward KL":>12s}{"reverse KL":>12s}')
for name, th in [('forward-KL fit', th_fwd), ('reverse-KL fit', th_rev),
                 ('reverse-KL fit, 2nd start', th_rev2)]:
    print(f'{name:26s}{th[0]:+8.3f}{np.exp(th[1]):7.3f}'
          f'{fwd(th):12.3f}{rev(th):12.3f}')

d2l.plot(x, [np.exp(log_p(x)), np.exp(log_q(x, th_fwd)),
             np.exp(log_q(x, th_rev))], 'x', 'density',
         legend=['target P (mixture)', 'forward-KL fit', 'reverse-KL fit'],
         figsize=(6, 3))
```

The forward fit matches our calculation: $\mu = -0.800$ and
$\sigma = 1.929 \approx \sqrt{3.72}$. It is a broad bell curve draped over both
modes, putting substantial mass in the valley between them where $P$ has almost
none. The first reverse fit lands at $\mu = -1.998$, $\sigma = 0.603$: it hugs
the dominant component and leaves almost no mass on the minor one. Its reverse
KL, $0.356$ nats, is close to $\log(1/0.7) \approx 0.357$, the cost of treating
the $70\%$ component as if it were the entire distribution. When started near
the minor mode, the optimizer settles instead on $\mu = +1.995$, with reverse
KL $1.202$ nats, close to $\log(1/0.3) \approx 1.204$. Which mode a variational
approximation captures therefore depends on its initialization.

Because reverse KL averages under the model, any mode the model fails to visit
is never inspected. The objective only notices that the chosen mode carries
weight $0.7$ (or $0.3$) instead of $1$. That is why variational posteriors
notoriously underestimate uncertainty: here the reverse fit has standard
deviation $0.60$, whereas the target's is $1.93$.

The last two columns highlight how sharply the two criteria disagree on the
same model. Forward KL, the maximum-likelihood criterion, penalizes the
dominant-mode fit at $5.98$ nats (more than ten times the $0.558$ nats of the
broad fit) and the minor-mode fit at $14.5$ nats. In both cases, the dropped
mode sits more than six of the model's standard deviations away from its mean,
where $\log q$ is deeply negative (Exercise
:exref:`ex_mdl-divergences-distances-mode-dropping`c). Reverse KL ranks the
first two fits in the opposite order, $0.356$ nats against $1.27$: what it
cannot forgive is the broad fit's mass sitting in the empty valley between the
modes.

Our boundary rules for vanishing densities capture this contrast in the limit.
For forward KL, $f(u) = u \log u$ has $f'(\infty) = \infty$, so placing zero
model mass where $P$ has positive mass costs infinity, while $f(0) = 0$ means
placing model mass where $P$ has none adds no boundary term at all. For
reverse KL, $f(u) = -\log u$, those two values swap places.

## Total Variation, Pinsker's Inequality, and Data Processing
:label:`sec_mdl-tv-pinsker`

Knowing that a divergence equals $0.4$ tells us very little on its own until we
know what that number actually guarantees. For total variation, the operational
meaning is completely transparent, and Pinsker's inequality passes much of that
meaning on to KL divergence.

### Total Variation

The divergence function $\tfrac12|u - 1|$ is the simplest entry in
:numref:`tab_mdl-f-div-generators`: it penalizes only the absolute distance of
the ratio from $1$. Its divergence, half the $\ell_1$ distance between $p$ and
$q$, admits a second, equivalent form that makes the number immediately
interpretable. Total variation is the largest gap between the probabilities
that $P$ and $Q$ assign to any event $A$:

$$
\textrm{TV}(P, Q) = \frac{1}{2} \sum_x \lvert p(x) - q(x) \rvert
= \sup_{A} \,\lvert P(A) - Q(A) \rvert.
$$
:eqlabel:`eq_mdl-tv-def`

If $\textrm{TV}(P, Q) = 0.03$, no event whatsoever can separate the two
distributions by more than three percentage points. The supremum is achieved by
the event $A^\star = \{x : p(x) > q(x)\}$ of outcomes that $P$ favors over $Q$.
To verify this, pick any event $A$. Then
$P(A) - Q(A) = \sum_{x \in A} (p(x) - q(x)) \leq \sum_{x \in A^\star} (p(x) - q(x))$,
because switching from $A$ to $A^\star$ only adds non-negative terms and drops
non-positive ones. Since $\sum_x (p(x) - q(x)) = 0$, the positive and negative
parts of $p - q$ have equal total magnitude, so the right-hand side equals
$\tfrac12 \sum_x |p(x) - q(x)|$; the same bound applies to $Q(A) - P(A)$.

On our running pair, $A^\star$ consists of the first outcome alone, giving
$\textrm{TV} = 0.6 - 0.2 = 0.4$, as we found earlier. :numref:`fig_mdl-tv-area`
illustrates the same geometry for two continuous densities. Total variation is
also a true metric: both forms are manifestly symmetric, and the triangle
inequality follows directly from the $\ell_1$ formula (Exercise
:exref:`ex_mdl-divergences-distances-prove-that-total-variation`a).

![Two equal-variance densities and the regions where they disagree. The lobe where $p > q$ and the lobe where $q > p$ have equal area, and that common area is the total variation distance. The event attaining the supremum is $A^\star = \{p > q\}$, here the half-line left of the crossing: guessing $P$ there and $Q$ elsewhere is the optimal single-sample test.](../img/mdl-it-tv-area.svg)
:label:`fig_mdl-tv-area`

The supremum form gives TV a concrete statistical meaning. Imagine handing a
tester a single sample, drawn with equal probability $\tfrac12$ from $P$ or
from $Q$, and asking which distribution generated it. Any deterministic test
answers "$P$" when the sample falls in some region $A$, and "$Q$" otherwise,
succeeding with probability

$$
\tfrac12 P(A) + \tfrac12 \big(1 - Q(A)\big)
= \tfrac12 \big(1 + P(A) - Q(A)\big) \leq \tfrac12 \big(1 + \textrm{TV}(P, Q)\big),
$$

with equality at $A = A^\star$. Every test takes this form for some $A$ (a
randomized test merely averages deterministic ones and cannot do better), so
even the optimal tester beats a fair coin flip by at most $\textrm{TV}/2$. On
our running pair, the optimal test answers "$P$" on the first outcome and is
right $70\%$ of the time. Both the discriminator of a GAN (trying to tell real
data from generated samples) and a detector of machine-generated text are
testers of this kind, and total variation sets a hard ceiling on how accurate
either can be from a single sample.

### Pinsker's Inequality

Total variation bounds every statistical test, while KL divergence is what we
can usually compute or control in practice because neural network losses are
built from log-probabilities. Pinsker's inequality bridges the two: a small KL
divergence certifies that no test can reliably tell $P$ from $Q$. Policy trust
regions in :numref:`sec_ppo` are a classic example: the monotonic improvement
guarantee is proved in terms of total variation, yet enforced through a bound
on KL.

**Proposition (Pinsker's inequality).** :cite:`Pinsker.1964,Csiszar.1967`
*For distributions $P$ and $Q$ on a finite or countable set,*

$$
\textrm{TV}(P, Q) \;\leq\; \sqrt{ \tfrac{1}{2}\, D_{\textrm{KL}}(P\|Q) }.
$$
:eqlabel:`eq_mdl-pinsker`

**Proof.** *Step 1: two outcomes.* Write $\textrm{Ber}(a)$ for a coin with
heads probability $a$. For $a \in [0, 1]$ and $b \in (0, 1)$, we have
$D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) = a \log\frac{a}{b} + (1-a) \log\frac{1-a}{1-b}$
and $\textrm{TV}(\textrm{Ber}(a), \textrm{Ber}(b)) = |a - b|$. We therefore
need to show that
$\rho(a) = D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) - 2(a - b)^2$ is
non-negative. Fix $b$. Clearly $\rho(b) = 0$, the first derivative
$\rho'(a) = \log\frac{a}{b} - \log\frac{1-a}{1-b} - 4(a - b)$ also vanishes at
$a = b$, and for $0 < a < 1$ the second derivative is

$$
\rho''(a) = \frac{1}{a(1-a)} - 4 \geq 0,
$$

because $a(1-a) \leq \tfrac{1}{4}$. A convex function with both value and slope
zero at $a = b$ is non-negative everywhere on $(0, 1)$, and by continuity (with
$0 \log 0 = 0$) at the endpoints $a = 0$ and $a = 1$ as well. If $b$ is $0$ or
$1$, the KL divergence is infinite unless $a = b$, so the inequality holds
trivially.

*Step 2: reduction to two outcomes.* Let $A^\star = \{p > q\}$, and set
$a = P(A^\star)$ and $b = Q(A^\star)$, so that $\textrm{TV}(P, Q) = a - b$.
Coarsening the space by merging all outcomes inside $A^\star$ into one bucket
and all outcomes outside $A^\star$ into another can only *decrease* KL. That
follows from the log-sum inequality: for non-negative numbers $v_i$ and $w_i$,

$$
\sum_i v_i \log\frac{v_i}{w_i} \geq \big(\sum_i v_i\big) \log \frac{\sum_i v_i}{\sum_i w_i},
$$

which is itself one application of Jensen's inequality to $t \mapsto t\log t$.
Applying it separately to $A^\star$ and its complement gives

$$
D_{\textrm{KL}}(P\|Q) \;\geq\; D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b)) \;\geq\; 2(a - b)^2
= 2\,\textrm{TV}(P, Q)^2,
$$

which rearranges to :eqref:`eq_mdl-pinsker`. $\blacksquare$

On our running pair, Pinsker's inequality reads
$0.4 \leq \sqrt{0.396/2} = 0.445$, and we can trace Step 2 by hand. Merging the
second and third outcomes collapses the pair to two coins, $(0.6, 0.4)$ and
$(0.2, 0.8)$. Their TV distance remains $0.4$, their KL divergence drops from
$0.396$ to $0.6 \log 3 + 0.4 \log \tfrac12 = 0.382$ nats, and Step 1 lower
bounds that by $2\,(0.4)^2 = 0.32$.

For a coin of bias $0.6$ compared with a fair coin, KL is
$0.6 \log 1.2 + 0.4 \log 0.8 = 0.0201$ nats, while $2\,(0.1)^2 = 0.02$: for
nearly equal coins, Pinsker's inequality is almost tight. Crucially, however,
there is no useful converse. A model that puts $q = 0$ on a single rare outcome
has infinite KL even when its TV distance is tiny. The bound also becomes
vacuous for well-separated distributions, since TV can never exceed $1$: two
unit-variance Gaussians separated by $50$ have $\textrm{TV} \approx 1$, whereas
:eqref:`eq_mdl-dd-gaussian-kl` gives KL $= 50^2/2 = 1250$ nats.

How tight is Pinsker's bound on typical distributions, and can the constant
$\tfrac12$ be improved? To check the first question, we sample $10{,}000$
random pairs of distributions on five outcomes uniformly from the simplex. To
check the second, we let two coins approach each other at $a = b = \tfrac12$:
Step 1 lost nothing to second order there, so the ratio
$\textrm{TV}/\sqrt{\textrm{KL}/2}$ should climb all the way to $1$.

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

Every point in the scatter plot falls beneath the curve $\sqrt{\textrm{KL}/2}$.
The median pair sits near $0.8$ of the bound, and the tightest of the $10{,}000$
random pairs comes within a few hundredths of $1$. Once $\textrm{KL} > 2$, the
curve rises above $1$ and the bound becomes vacuous. For nearly fair coins, the
ratio climbs through $0.9899$, $0.9999$, and $0.999999$, confirming that the
constant $\tfrac{1}{2}$ cannot be tightened further (Exercise
:exref:`ex_mdl-divergences-distances-sharpness-pinsker-coins-biases`c).
Pinsker's original proof had a weaker constant; the sharp factor $\tfrac{1}{2}$
is due to :citet:`Csiszar.1967` and, independently, to Kullback and Kemperman.

### Data Processing
:label:`subsec_mdl-f-div-data-processing`

Can a feature extractor, a quantizer, or any other post-processing step make
two distributions easier to tell apart? In Pinsker's proof, merging outcomes
could only decrease KL. The same principle holds for every f-divergence and
every form of processing. A *channel* $\kappa(y \mid x)$ is any rule,
deterministic or randomized, that transforms an input $x$ into an output $y$.
Passing $P$ through the channel produces the output distribution
$(P\kappa)(y) = \sum_x \kappa(y \mid x)\, p(x)$, and passing $Q$ produces
$Q\kappa$ analogously.

**Proposition (data processing for f-divergences).** *For any channel $\kappa$,
convex $f$ with $f(1) = 0$, and $P \ll Q$ on a finite or countable set,*

$$
D_f(P\kappa \,\|\, Q\kappa) \;\leq\; D_f(P\|Q).
$$
:eqlabel:`eq_mdl-f-div-dpi`

**Proof.** Since $\sum_y \kappa(y \mid x) = 1$, we can expand
$D_f(P\|Q) = \sum_y \sum_x \kappa(y \mid x)\, q(x)\, f\big(p(x)/q(x)\big)$. Fix
an output $y$ with $(Q\kappa)(y) > 0$. The normalized weights
$\kappa(y \mid x)\, q(x) / (Q\kappa)(y)$ sum to one over $x$, and averaging the
density ratios $p(x)/q(x)$ under these weights yields
$(P\kappa)(y)/(Q\kappa)(y)$. Applying Jensen's inequality to the inner sum
therefore bounds it below by
$(Q\kappa)(y)\, f\big((P\kappa)(y)/(Q\kappa)(y)\big)$, which is the $y$-th term
of $D_f(P\kappa\|Q\kappa)$. Any output $y$ with $(Q\kappa)(y) = 0$ must have
$\kappa(y \mid x) = 0$ wherever $q(x) > 0$, hence $(P\kappa)(y) = 0$ as well,
contributing nothing to either side. Summing over $y$ completes the proof.
$\blacksquare$

No matter which divergence function we pick, processing data can never make two
distributions more distinguishable. In deep learning, this inequality is often
read in reverse: a divergence measured on learned features, quantized outputs,
or any other summary of the data is always a lower bound on the divergence
between the raw inputs. Merging outcomes in Pinsker's proof is simply the
deterministic channel $x \mapsto \mathbf{1}(x \in A^\star)$. On our running
pair, it lowered KL from $0.396$ to $0.382$, and it lowers $\chi^2$ from
$1.013$ to $0.16/0.2 + 0.16/0.8 = 1$. The KL special case is also the engine
behind the data-processing inequality for mutual information in
:numref:`subsec_mdl-dpi`.

An invertible transformation acts as a channel in both directions, forcing
equality in :eqref:`eq_mdl-f-div-dpi`: every f-divergence is invariant to
relabeling outcomes or rescaling the axes. By contrast, the last two rows of
:numref:`tab_mdl-point-mass-table` do change when you rescale the axis. What
those distances gain in geometric sensitivity to $d$, they give up in scale
invariance.

## The Fenchel Conjugate and the Critic Bound
:label:`sec_mdl-f-gan-dual`

Suppose we have a generative network that produces samples from a distribution
$Q$, alongside a dataset of samples from the true distribution $P$. Neither
collection of samples gives us access to the density ratio $p(x)/q(x)$ required
by :eqref:`eq_mdl-f-div-def`. Can we still estimate $D_f$ from samples alone?
As it turns out, we can lower-bound any f-divergence using only expectations
under $P$ and $Q$, and that lower bound turns any divergence into an
adversarial training objective.

### Derivation of the Bound

The key tool is the convex conjugate from :numref:`subsec_mdl-convex-conjugate`.
For a convex function $f$, its conjugate is defined by

$$
f^*(t) = \sup_{u}\, \big( ut - f(u) \big),
$$

so that $-f^*(t)$ is the vertical intercept of the tangent line to $f$ with
slope $t$ (:numref:`fig_mdl-f-gan-tangent-bound`). For $f(u) = u \log u$, the
supremum occurs where $\log u + 1 = t$, giving $f^*(t) = e^{t-1}$. Because a
convex function always lies above every one of its tangent lines, we obtain the
Fenchel--Young inequality:

$$
f(u) \geq ut - f^*(t) \quad \textrm{for all } t,
$$
:eqlabel:`eq_mdl-fenchel-young`

with equality whenever $t$ matches the slope of $f$ at $u$, namely
$t = f'(u)$.

![A convex divergence function, here $f(u)=u\log u$, is the upper envelope of its tangent lines. The tangent of slope $t$ has vertical intercept $-f^*(t)$, here $-e^{t-1}$. Keeping a single tangent gives the Fenchel--Young lower bound, and taking the supremum over slopes recovers $f(u)$. Applying this inequality pointwise to the density ratio yields the critic bound.](../img/mdl-it-f-gan-tangent-bound.svg)
:label:`fig_mdl-f-gan-tangent-bound`

Now apply :eqref:`eq_mdl-fenchel-young` *pointwise inside the divergence*,
letting the slope vary with $x$. Specifically, pick any function $T(x)$, called
the **critic**, and substitute $u = p(x)/q(x)$ and $t = T(x)$.

**Proposition (critic bound).** *Let $f$ be convex and differentiable with
$f(1) = 0$, let $P \ll Q$, and let $T$ be any critic with
$\mathbb{E}_P|T| < \infty$ and $\mathbb{E}_Q|f^*(T)| < \infty$. Then*

$$
D_f(P\|Q) \;\geq\; \mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[f^*(T(x))].
$$
:eqlabel:`eq_mdl-f-gan-bound`

*Moreover, the supremum of the right-hand side over all critics equals
$D_f(P\|Q)$, and it is attained by $T^\star(x) = f'\big(p(x)/q(x)\big)$
wherever $T^\star$ is finite*
:cite:`Nguyen.Wainwright.Jordan.2010,Nowozin.Cseke.Tomioka.2016`.

**Proof.** Multiply :eqref:`eq_mdl-fenchel-young` at $u = p(x)/q(x)$ and
$t = T(x)$ by $q(x) \geq 0$:

$$
q(x)\, f\!\left(\frac{p(x)}{q(x)}\right)
\;\geq\; p(x)\, T(x) - q(x)\, f^*(T(x)).
$$

Summing (or integrating) over $x$ yields :eqref:`eq_mdl-f-gan-bound`. Because
Fenchel--Young holds with equality at $t = f'(u)$, choosing
$T^\star(x) = f'(p(x)/q(x))$ turns the inequality into an equality at every
$x$, and therefore in expectation as well. $\blacksquare$

Both terms on the right-hand side of :eqref:`eq_mdl-f-gan-bound` are simple
expectations, of $T$ over real data and of $f^*(T)$ over generated samples,
requiring no explicit densities at all. We can therefore parameterize $T$ as a
neural network and train it on minibatches to maximize the bound. For KL
divergence, the bound reads
$D_{\textrm{KL}}(P\|Q) \geq \mathbb{E}_P[T] - \mathbb{E}_Q[e^{T - 1}]$ for
every critic $T$. The **f-GAN** of :citet:`Nowozin.Cseke.Tomioka.2016` (after
which :eqref:`eq_mdl-f-gan-bound` is also called the f-GAN bound) turns this
variational form into a two-player game: the critic maximizes the bound so that
it tracks the divergence, while the generator producing $Q$ minimizes it.
:numref:`sec_gan_objectives` implements and trains such pairs.

### The Original GAN

The original GAN of :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` falls right
out of this framework when we choose $f(u) = u\log u - (u+1)\log(u+1)$. Its
convex conjugate is $f^*(t) = -\log(1 - e^t)$ for $t < 0$. If we write the
critic as the logarithm of a discriminator $D(x) \in (0, 1)$, setting
$T = \log D$, the critic bound becomes the familiar binary cross-entropy game:

$$
\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]
= \mathbb{E}_{x \sim P}[\log D(x)] + \mathbb{E}_{x \sim Q}[\log(1 - D(x))].
$$

The optimal critic $T^\star = f'(p/q) = \log\big(p/(p+q)\big)$ corresponds to
the Bayes-optimal discriminator $D^\star = p/(p + q)$. Comparing with the
Jensen--Shannon divergence function $f_{\textrm{JS}}$ in
:numref:`tab_mdl-f-div-generators`, we find that
$f(u) = 2 f_{\textrm{JS}}(u) - (u - 1)\log 2 - \log 4$: twice the
Jensen--Shannon function, plus a linear term in $u - 1$ (which changes no
divergence) and the constant $f(1) = -\log 4$ (which only shifts the value). At
$D^\star$, the minimax game therefore achieves $2\,\textrm{JS}(P, Q) - \log 4$,
which on our running pair equals $2(0.0911) - 1.386 = -1.204$.

### Limits of the Bound

In practice, the critic bound runs into two limitations, and we can measure the
first one directly. Because the bound is tight only at $T^\star$, an
undertrained critic reports less than the true divergence. Let us watch how a
critic closes that gap as it trains.

On our discrete running pair, expectations are simple sums and a critic is just
a vector of three numbers, one per outcome. Take the $\chi^2$ divergence, with
$f(u) = (u-1)^2$, conjugate $f^*(t) = t + t^2/4$, and optimal critic
$T^\star = 2(p/q - 1)$. (Here the conjugate takes the supremum over all real
$u$; Exercise :exref:`ex_mdl-divergences-distances-derive-convex-conjugate`a
derives it and checks that this loses nothing at $T^\star$.) Starting from
$T = 0$, where the bound is $0$, we run gradient ascent with step size $1$ on
the bound, whose derivative with respect to $T(x)$ is
$p(x) - q(x)\,\big(1 + T(x)/2\big)$. By the proposition, the ascent must climb
from below toward $\chi^2(P\|Q) = 1.0133$. As a sanity check, the final line
also evaluates an overconfident critic $2T^\star$ that points in the right
direction but overshoots by a factor of two.

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

After five steps, the critic certifies $0.70$ out of $1.01$ (about $70\%$ of
the true divergence); after ten steps it reaches $90\%$, and after twenty
steps $99\%$. At no point does the bound overshoot the true value. Even the
overconfident critic $2T^\star$ reports zero, no better than an untrained
critic. With exact expectations, any adversarial divergence estimate is an
underestimate, and the number itself never reveals how much is missing. When
expectations are replaced by finite-sample averages, it remains a valid lower
bound (in expectation) only on fresh samples that were not used to train the
critic.

The gradient update also reveals where the missing divergence hides. Because
$T(x)$ moves at a rate proportional to $q(x)$, the critic learns slowest where
the model's samples are rarest. On our running pair, that is the first outcome,
where the density ratio is largest and most of the divergence lives
(Exercise :exref:`ex_mdl-divergences-distances-derive-convex-conjugate`b).

The second limitation brings us back to disjoint supports, now viewed from the
critic's perspective. Extending $T^\star = f'(p/q)$ formally beyond the
hypothesis $P \ll Q$, the ratio equals $\infty$ on the data support and $0$ on
the model support, so $T^\star$ takes the constant value $f'(\infty)$ on one
and $f'(0)$ on the other. Because the generator updates its parameters using
the gradient of the critic evaluated at its own samples, a critic that is
locally constant there provides zero gradient. The generator receives no signal
telling it which way to move.

Both limitations stem from the same root: the density ratio $p/q$. Finite
samples never hand us $p/q$ directly, and whenever the supports separate, the
ratio collapses to $0$ or $\infty$ and erases all trace of how far apart the
distributions sit. In :numref:`sec_mdl-test-functions-transport`, we turn to
comparisons that do not rely on density ratios at all.

## Summary

KL divergence is one member of a broad family: an f-divergence averages a
convex function of the density ratio $p/q$ under $Q$. Jensen's inequality
guarantees non-negativity for every member, all smooth members agree up to
their curvature $f''(1)$ when $P$ and $Q$ are close, and no post-processing of
the data can increase any f-divergence. Choosing the direction of KL is a
modeling decision: forward KL (maximum likelihood) stretches to cover every
mode, whereas reverse KL (variational inference) can lock onto a single one.
Operationally, the best single-sample test distinguishing $P$ from $Q$ succeeds
with probability $\tfrac12(1 + \textrm{TV})$, and Pinsker's inequality bounds
TV by $\sqrt{D_{\textrm{KL}}/2}$. Using the convex conjugate, any critic
lower-bounds $D_f$ via expectations over $P$ and $Q$ alone, which is how
adversarial training estimates divergences from samples. Yet every member
shares the same blind spot of the density ratio: on disjoint supports, $D_f$
freezes at the constant $f(0) + f'(\infty)$, regardless of how near or far the
supports lie.

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
   :exlabel:`ex_mdl-divergences-distances-prove-that-total-variation`

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
       the mixture of that experiment, and check it
       with `kl(log_p(x), log_q(x, [-2.0, np.log(0.6)]), x)`. The experiment
       prints two values in its forward-KL column to compare with:
       $5.981$ nats at the reverse-KL fit, whose parameters
       $(-1.998, 0.603)$ are not quite $(-2, 0.6)$, and $0.558$ nats at the
       forward fit.
   :exlabel:`ex_mdl-divergences-distances-mode-dropping`

1. **Local quadratic regime.** The proposition
   :eqref:`eq_mdl-f-div-local` bounds the gap between $D_f$ and
   $\tfrac12 f''(1)\, \chi^2$.
    1. Exercise :exref:`ex_mdl-information-theory-gaussian-kl`b of
       :numref:`sec_mdl-information_theory` finds the order at which the two
       directions of the Gaussian KL divergence first differ. Which property
       of the KL and reverse-KL functions explains that order? Carry the
       Taylor expansion of the proof one term further, and show that the
       ratio which the local-quadratic experiment prints is
       $1 + c\, t\, f'''(1)/f''(1) + O(t^2)$ with
       $c = \mathbb{E}_Q[(u - 1)^3] / \big(3 \chi^2(P\|Q)\big)$, about $0.49$
       for the running pair. Compute $f'''(1)/f''(1)$ for the four functions.
       Why is the shortfall of reverse KL at $t = 0.01$ twice that of KL, and
       why do squared Hellinger and JS print the same value there?
    1. The total-variation function $\tfrac{1}{2}|u - 1|$ has no second
       derivative at $1$. Let $P_t = Q + t\,(P - Q)$ for
       $t \in (0, 1]$. How do $\textrm{TV}(P_t, Q)$ and
       $\chi^2(P_t\|Q)$ scale with $t$? Explain from this why Pinsker's
       inequality :eqref:`eq_mdl-pinsker` bounds TV by the square root of
       KL and not by a multiple of KL.
    1. Let $P$ and $Q$ be coins with heads probabilities $b$ and
       $b + \delta$. Use the proposition to show that
       $D_{\textrm{KL}}(P\|Q) = \delta^2 / \big(2 b (1 - b)\big) + O(\delta^3)$.
       Exercise :exref:`ex_mdl-information-theory-two-part-coin`a of
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

1. **Conjugates and critics.** The critic bound :eqref:`eq_mdl-f-gan-bound`
   needs the conjugate $f^*$ of the divergence function.
    1. Derive the conjugate $f^*(t) = t + t^2/4$ of the $\chi^2$ function
       $f(u) = (u - 1)^2$, taking the supremum over $u \in \mathbb{R}$, and
       its optimal critic $T^\star = 2(p/q - 1)$. Redo the computation over
       the function's domain $u \in (0, \infty)$, and show that the
       supremum is $t + t^2/4$ for $t \geq -2$ and $-1$ for $t \leq -2$.
    1. Use the $\mathbb{R}$-conjugate. Show that for every critic $T$ the
       gap between $\chi^2(P\|Q)$ and the bound equals
       $\tfrac{1}{4} \mathbb{E}_Q\big[(T - T^\star)^2\big]$. Deduce that a
       scaled critic $T = c\,T^\star$ certifies
       $\chi^2(P\|Q)\,(2c - c^2)$, which explains the value that the
       experiment prints for $2T^\star$. Show that one ascent step of the
       experiment multiplies $T(x) - T^\star(x)$ by $1 - q(x)/2$. Which
       outcome of the running pair accounts for most of the gap after ten
       steps, and why? For which step sizes does the ascent converge?
    1. Compute the conjugate of the total-variation function
       $f(u) = \tfrac{1}{2}|u - 1|$ over $u \in (0, \infty)$, corresponding to
       the total variation distance
       $D_{\textrm{TV}}(p, q) = \tfrac{1}{2}\int |p(x)-q(x)|\,dx$. Show that
       critics with values in $[-\tfrac{1}{2}, \tfrac{1}{2}]$ suffice to
       maximize the bound, and conclude that
       $\textrm{TV}(P, Q) = \sup_g \big(\mathbb{E}_P[g] - \mathbb{E}_Q[g]\big)$
       over all functions $g$ with $|g| \leq \tfrac{1}{2}$. (As we will see in
       :numref:`sec_mdl-ipm-mmd`, replacing the boundedness condition
       $|g| \leq \tfrac{1}{2}$ by the $1$-Lipschitz condition
       $\|f\|_{\textrm{Lip}} \le 1$ yields the 1-Wasserstein or Earth Mover's
       distance
       $W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(X,Y)\sim\gamma}[\|X - Y\|] = \sup_{\|f\|_{\textrm{Lip}} \le 1} (\mathbb{E}_p[f] - \mathbb{E}_q[f])$.)
       Which $g$ attains the total-variation supremum?
   :exlabel:`ex_mdl-divergences-distances-derive-convex-conjugate`

[Discussions](https://d2l.discourse.group/t/36835)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.3]{.kicker}

f-Divergences and Variational Duals<br>**density ratios · forward vs. reverse KL · total variation · Pinsker's inequality · Fenchel duality · f-GAN**
:::
:::

::: {.slide title="Why Disjoint Supports Break Density-Ratio Losses"}
[Motivation: The Point-Mass Diagnostic]{.kicker}

A generative network mapping low-dimensional noise into image space places its samples on a thin manifold that initially fails to overlap real data. Compare two point masses $P = \delta_0$ and $Q = \delta_d$ separated by distance $d$:

| Distribution Comparison | Near Miss ($d = 0.01$) | Moderate ($d = 1$) | Far Off ($d = 100$) | Gradient w.r.t. $d$ |
|:--|---:|---:|---:|:---|
| **KL divergence** $D_{\textrm{KL}}(P \| Q)$ | $\infty$ | $\infty$ | $\infty$ | Undefined ($\infty$) |
| **Jensen–Shannon divergence** $\textrm{JS}(P, Q)$ | $\log 2 \approx 0.693$ | $\log 2 \approx 0.693$ | $\log 2 \approx 0.693$ | $0$ (completely flat) |
| **Total variation distance** $\textrm{TV}(P, Q)$ | $1.000$ | $1.000$ | $1.000$ | $0$ (completely flat) |
| **RBF Kernel MMD** $\textrm{MMD}^2(P, Q)$ ($\ell = 1$) | $0.0001$ | $0.787$ | $2.000$ | Nonzero locally |
| **Wasserstein-1 distance** $W_1(P, Q)$ | $0.01$ | $1.00$ | $100.00$ | $\pm 1$ everywhere |

::: {.d2l-note .warn}
**The Frozen-Row Problem:** The first three rows belong to one family—the **$f$-divergences**—built from the density ratio $u(x) = p(x)/q(x)$. When supports do not overlap, all of them freeze at a constant independent of $d$!
:::
:::

::: {.slide title="Checking the Point-Mass Diagnostic in Code"}
[Five Comparisons on Disjoint Atoms]{.kicker}

Representing $\delta_0$ and $\delta_d$ as probability vectors $(1, 0)$ and $(0, 1)$ on atoms $\{0, d\}$ reproduces the table directly:

@!mdl-divergences-distances-divergences-and-distances-between-distributions-1

- This section studies the density-ratio family (**$f$-divergences**) to answer three questions: (1) how forward and reverse KL differ on misspecified models, (2) what total variation and Pinsker's inequality guarantee for statistical detectors, and (3) how convex duality estimates any $D_f$ from samples via a critic network.
:::

::: {.slide title="Divergences Versus True Probability Metrics"}
[Mathematical Foundations]{.kicker}

What mathematical axioms distinguish a general divergence from a true distance metric?

- **Divergence axioms:** A function $D(P, Q)$ is a *divergence* if it satisfies non-negativity and identity of indiscernibles:
  $$D(P, Q) \ge 0, \qquad D(P, Q) = 0 \iff P = Q$$
- **Metric axioms:** A *metric* must additionally be symmetric ($D(P, Q) = D(Q, P)$) and satisfy the triangle inequality ($D(P, R) \le D(P, Q) + D(Q, R)$).
- **Why KL divergence fails both metric axioms:**
  - **Asymmetry:** For $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(0, 4)$, $D_{\textrm{KL}}(P \| Q) = 0.318\text{ nats} \neq D_{\textrm{KL}}(Q \| P) = 0.807\text{ nats}$.
  - **Triangle violation:** For unit-variance Gaussians centered at $0, 1, 2$, $D_{\textrm{KL}}(\mathcal{N}(\mu_1, 1) \| \mathcal{N}(\mu_2, 1)) = \tfrac12(\mu_1 - \mu_2)^2$, so $D_{\textrm{KL}}(0 \| 2) = 2 > \tfrac12 + \tfrac12$ (KL behaves like a *squared* Euclidean distance).
:::

::: {.slide title="The f-Divergence Template and Non-Negativity"}
[Unifying Density-Ratio Divergences]{.kicker}

Rewriting KL divergence as $\mathbb{E}_{x \sim Q}[u(x)\log u(x)]$ where $u(x) = p(x)/q(x)$ is the density ratio reveals a general template: replace $u \log u$ with any convex function $f : (0, \infty) \to \mathbb{R}$ satisfying $f(1) = 0$ (assuming $P \ll Q$):

$$D_f(P \| Q) = \mathbb{E}_{x \sim Q}\!\left[f\!\left(\frac{p(x)}{q(x)}\right)\right]$$

- **One-line proof of non-negativity (Jensen's inequality):** Because the density ratio averages to $1$ under $Q$ ($\mathbb{E}_Q[u] = \sum_x q(x)\frac{p(x)}{q(x)} = \sum_x p(x) = 1$), convexity of $f$ immediately yields:
  $$D_f(P \| Q) = \mathbb{E}_Q[f(u)] \ge f(\mathbb{E}_Q[u]) = f(1) = 0$$
- **Strict positivity and affine invariance:** If $f$ is strictly convex at $u = 1$ ($f(u) > c(u-1)$ for all $u \neq 1$), then $D_f(P \| Q) = 0 \iff u(x) = 1$ everywhere ($P = Q$). Moreover, adding $c(u - 1)$ to $f(u)$ changes no divergence because $\mathbb{E}_Q[c(u - 1)] = 0$.
:::

::: {.slide title="Common Divergence Functions and Their Properties"}
[The f-Divergence Catalog]{.kicker}

Every major density-ratio comparison in statistics and machine learning corresponds to one convex generator $f(u)$:

| Divergence Name | Generator $f(u)$ (with $f(1) = 0$) | Explicit Sum / Integral Formula | Curvature $f''(1)$ | Symmetric / Metric? |
|:--|:---|:---|---:|:---|
| **Forward KL** $D_{\textrm{KL}}(P \| Q)$ | $u \log u$ | $\sum_x p(x) \log(p(x)/q(x))$ | $1$ | No / No |
| **Reverse KL** $D_{\textrm{KL}}(Q \| P)$ | $-\log u$ | $\sum_x q(x) \log(q(x)/p(x))$ | $1$ | No / No |
| **Pearson $\chi^2$** $\chi^2(P \| Q)$ | $(u - 1)^2$ | $\sum_x (p(x) - q(x))^2 / q(x)$ | $2$ | No / No |
| **Squared Hellinger** $d_{\mathrm{H}}^2(P, Q)$ | $(\sqrt{u} - 1)^2$ | $\sum_x (\sqrt{p(x)} - \sqrt{q(x)})^2$ | $1/2$ | Yes / $d_{\mathrm{H}}$ is a metric |
| **Total Variation** $\textrm{TV}(P, Q)$ | $\tfrac12 |u - 1|$ | $\tfrac12 \sum_x |p(x) - q(x)|$ | Corner | Yes / True metric |
| **Jensen–Shannon** $\textrm{JS}(P, Q)$ | $\tfrac{u}{2}\log u - \tfrac{u+1}{2}\log\tfrac{u+1}{2}$ | $\tfrac12 D_{\textrm{KL}}(P \| M) + \tfrac12 D_{\textrm{KL}}(Q \| M)$ | $1/4$ | Yes / $\sqrt{\textrm{JS}}$ is a metric |
| **$\alpha$-Divergence** ($\alpha \neq 0, 1$) | $\frac{u^\alpha - \alpha u + \alpha - 1}{\alpha(\alpha - 1)}$ | Interpolates rev-KL ($\alpha\to 0$) to KL ($\alpha\to 1$) | $1$ | At $\alpha = 1/2$ |
:::

::: {.slide title="Geometry of Divergence Generators Near u = 1"}
[Comparing Penalty Shapes]{.kicker}

Shifting each smooth generator by $f'(1)(u - 1)$ so all curves touch zero tangentially at $u = 1$ reveals how they agree locally yet diverge in the tails:

::: {.cols .vc}
::: {.col}
- **Near $u = 1$ (local agreement):** Every smooth curve is a parabola $\tfrac12 f''(1)(u - 1)^2$, while Total Variation has a V-shaped corner.
- **As $u \to 0$ ($p \ll q$, model places mass in empty regions):** Reverse KL ($-\log u$) blows up to $+\infty$, heavily penalizing spurious model modes.
- **As $u \to \infty$ ($p \gg q$, model misses data):** Forward KL ($u \log u$) and $\chi^2$ ($(u-1)^2$) grow superlinearly, whereas TV and JS grow only linearly.
:::

::: {.col .fig}
@fig:mdl-it-f-div-generators
:::
:::
:::

::: {.slide title="Boundary Rules: Why Disjoint Supports Freeze D_f"}
[Singular Measures & Disjoint Supports]{.kicker}

What happens to $D_f(P \| Q)$ when $P$ and $Q$ place mass on regions where the other vanishes?

- **Boundary limits at $p = 0$ and $q = 0$:**
  - Where $p(x) = 0 < q(x)$ ($u = 0$), the contribution is $q(x)\,f(0)$ with $f(0) = \lim_{u \downarrow 0} f(u)$.
  - Where $q(x) = 0 < p(x)$ ($u \to \infty$), taking $\lim_{q \downarrow 0} q\,f(p/q) = p \lim_{t \to \infty} f(t)/t$ gives $p(x)\,f'(\infty)$ where $f'(\infty) = \lim_{t \to \infty} f(t)/t$.
- **Universal constant on disjoint point masses ($d \neq 0$):** At $x = 0$ ($p=1, q=0$) and $x = d$ ($p=0, q=1$):
  $$D_f(\delta_0 \| \delta_d) = f(0) + f'(\infty) \quad \text{for every } d \neq 0$$
  - **Forward KL ($u\log u$):** $f(0) = 0$, $f'(\infty) = \infty$ $\implies$ $D_{\textrm{KL}}(\delta_0 \| \delta_d) = \infty$.
  - **Jensen–Shannon:** $f(0) = \tfrac12\log 2$, $f'(\infty) = \tfrac12\log 2$ $\implies$ $\textrm{JS}(\delta_0, \delta_d) = \log 2$.
  - **Total Variation ($\tfrac12|u-1|$):** $f(0) = \tfrac12$, $f'(\infty) = \tfrac12$ $\implies$ $\textrm{TV}(\delta_0, \delta_d) = 1$.
:::

::: {.slide title="Evaluating Six f-Divergences on the Running Pair"}
[Numerical Comparison]{.kicker}

On $P = (0.6, 0.3, 0.1)$ and $Q = (0.2, 0.5, 0.3)$, the six divergences span an order of magnitude ($0.091$ to $1.013$), yet rescaling by $2/f''(1)$ makes $2\,\textrm{KL} = 0.79$, $4\,d_{\mathrm{H}}^2 = 0.74$, and $8\,\textrm{JS} = 0.73$ nearly coincide:

@!divergences-distances-f-table
:::

::: {.slide title="Local Quadratic Theorem: All Smooth D_f Agree"}
[Second-Order Taylor Expansion]{.kicker}

Why do all smooth $f$-divergences become scalar multiples of Pearson's $\chi^2(P \| Q) = \mathbb{E}_Q[(u - 1)^2]$ when $P$ is close to $Q$?

- **Taylor expansion around $u = 1$:** For three-times differentiable $f$ with $|f'''| \le B$ on $[1-r, 1+r]$ where $r = \max_x |u(x) - 1| < 1$:
  $$f(u) = \underbrace{f(1)}_{=0} + f'(1)(u - 1) + \frac{f''(1)}{2}(u - 1)^2 + R(u), \qquad |R(u)| \le \frac{B}{6}|u - 1|^3$$
- **Expectation under $Q$ kills the first-order term:** Because $\mathbb{E}_Q[u - 1] = 0$, the slope $f'(1)$ drops out completely:
  $$\left|D_f(P \| Q) - \frac{f''(1)}{2}\chi^2(P \| Q)\right| \le \frac{Br}{6}\chi^2(P \| Q)$$
- **Connection to Fisher information and natural gradients:** For a small parameter step $q_{\boldsymbol{\theta}+\mathbf{h}}$ vs. $q_{\boldsymbol{\theta}}$, $u - 1 \approx \mathbf{h}^\top \nabla_{\boldsymbol{\theta}} \log q_{\boldsymbol{\theta}}$, so $\chi^2(q_{\boldsymbol{\theta}+\mathbf{h}} \| q_{\boldsymbol{\theta}}) \approx \mathbf{h}^\top \mathbf{I}(\boldsymbol{\theta})\mathbf{h}$ and *every* smooth $D_f$ has Hessian $f''(1)\mathbf{I}(\boldsymbol{\theta})$.
:::

::: {.slide title="Verifying Local Quadratic Convergence in Code"}
[Asymptotic Agreement]{.kicker}

Sliding $P_t = Q + t(P - Q)$ toward $Q$ scales $r = 2t$; as $t$ drops tenfold from $0.1$ to $0.01$, the relative gap between $D_f(P_t \| Q)$ and $\tfrac12 f''(1)\chi^2(P_t \| Q)$ drops tenfold to $< 1\%$:

@!mdl-divergences-distances-the-local-quadratic
:::

::: {.slide title="Forward vs. Reverse KL: Mode-Covering vs. Seeking"}
[Asymmetry as a Modeling Choice]{.kicker}

When fitting a parametric model $Q_{\boldsymbol{\theta}}$ to a target $P$ outside the model family, argument order reverses the geometry of the fit:

| Property | Forward KL: $D_{\textrm{KL}}(P \| Q_{\boldsymbol{\theta}}) = \mathbb{E}_P[\log(p/q_{\boldsymbol{\theta}})]$ | Reverse KL: $D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P) = \mathbb{E}_{Q_{\boldsymbol{\theta}}}[\log(q_{\boldsymbol{\theta}}/p)]$ |
|:--|:---|:---|
| **Expectation taken over** | True data distribution $P$ | Model distribution $Q_{\boldsymbol{\theta}}$ |
| **Blows up to $+\infty$ when** | $q_{\boldsymbol{\theta}}(x) \to 0$ where $p(x) > 0$ ($f'(\infty) = \infty$) | $p(x) \to 0$ where $q_{\boldsymbol{\theta}}(x) > 0$ ($f(0) = \infty$) |
| **Geometric behavior** | **Zero-avoiding / mode-covering:** stretches over all data modes, spilling mass into valleys | **Zero-forcing / mode-seeking:** locks onto a single mode and ignores distant modes |
| **What it requires** | Data samples $x \sim P$ and tractable density $q_{\boldsymbol{\theta}}(x)$ | Model samples $x \sim Q_{\boldsymbol{\theta}}$ and target $\tilde{p}(x)$ up to $Z$ |
| **Primary ML applications** | Maximum likelihood (classifiers, autoregressive LLMs) | Variational inference (ELBO), RLHF policy KL penalty |
:::

::: {.slide title="Fitting a Single Gaussian to a Bimodal Mixture"}
[Misspecified Model Behavior]{.kicker}

Consider fitting $Q_{\boldsymbol{\theta}} = \mathcal{N}(\mu, \sigma^2)$ to the asymmetric bimodal mixture $P = 0.7\,\mathcal{N}(-2, 0.6^2) + 0.3\,\mathcal{N}(2, 0.6^2)$:

- **Exact forward-KL fit (moment matching):** Minimizing $D_{\textrm{KL}}(P \| \mathcal{N}(\mu, \sigma^2))$ matches the mean and variance of $P$:
  $$\mu^* = 0.7(-2) + 0.3(2) = -0.800, \qquad (\sigma^*)^2 = 0.6^2 + 0.7(0.3)(-2 - 2)^2 = 3.72 \implies \sigma^* = 1.929$$
- **Reverse-KL local minima (mode dropping):** If $Q_{\boldsymbol{\theta}}$ matches a single component $\mathcal{N}(\mu_k, 0.6^2)$ carrying mixture weight $w_k$, then $p(x) \approx w_k q_{\boldsymbol{\theta}}(x)$ across that mode, so:
  $$D_{\textrm{KL}}(Q_{\boldsymbol{\theta}} \| P) \approx \mathbb{E}_{Q_{\boldsymbol{\theta}}}\!\left[\log\frac{q_{\boldsymbol{\theta}}(x)}{w_k q_{\boldsymbol{\theta}}(x)}\right] = \log\frac{1}{w_k}$$
  - Locking onto the $70\%$ mode ($\mu \approx -2, \sigma \approx 0.6$) costs $\ln(1/0.7) \approx 0.357\text{ nats}$; locking onto the $30\%$ mode ($\mu \approx +2$) costs $\ln(1/0.3) \approx 1.204\text{ nats}$.
:::

::: {.slide title="Visualizing Mode-Covering and Mode-Seeking Fits"}
[Numerical Optimization Demo]{.kicker}

Notice how reverse KL underestimates the target standard deviation by $>3\times$ ($\sigma = 0.603$ vs. $1.929$) and converges to whichever mode lies closer to initialization:

@!divergences-distances-fwd-rev-kl
:::

::: {.slide title="Total Variation and Optimal Single-Sample Tests"}
[Operational Meaning of TV]{.kicker}

What does a Total Variation distance of $\textrm{TV}(P, Q) = 0.4$ physically guarantee?

::: {.cols .vc}
::: {.col}
- **Worst-case event discrepancy:**
  $$\textrm{TV}(P, Q) = \frac{1}{2}\sum_x |p(x) - q(x)| = \sup_{A} |P(A) - Q(A)|$$
- **Optimal separating event:** The supremum is attained by $A^\star = \{x : p(x) > q(x)\}$ because positive and negative lobes of $p - q$ have equal area.
- **Hypothesis testing ceiling:** Given one draw equally likely from $P$ or $Q$, *any* detector (GAN discriminator or AI-text detector) succeeds with probability at most $\tfrac12(1 + \textrm{TV}(P, Q))$.
:::

::: {.col .fig}
@fig:mdl-it-tv-area
:::
:::
:::

::: {.slide title="Pinsker's Inequality: Bounding TV via KL"}
[Two-Step Proof via Log-Sum]{.kicker}

Because neural losses control log-probabilities ($D_{\textrm{KL}}$) while safety and trust-region guarantees require bounded event shifts ($\textrm{TV}$), **Pinsker's inequality** bridges the two:

::: {.d2l-note .rule}
**Proposition (Pinsker's Inequality):** For any two distributions $P$ and $Q$, $\textrm{TV}(P, Q) \le \sqrt{\tfrac12 D_{\textrm{KL}}(P \| Q)}$.
:::

- **Step 1 (Two-outcome Bernoulli case):** For coins $\textrm{Ber}(a)$ and $\textrm{Ber}(b)$, define $\rho(a) = D_{\textrm{KL}}(\textrm{Ber}(a) \| \textrm{Ber}(b)) - 2(a - b)^2$. Fixing $b$, we have $\rho(b) = \rho'(b) = 0$ and $\rho''(a) = \frac{1}{a(1-a)} - 4 \ge 0$ (since $a(1-a) \le 1/4$), so $\rho(a) \ge 0$ for all $a \in [0, 1]$.
- **Step 2 (Coarsening to $A^\star = \{p > q\}$ via log-sum inequality):** Merge all outcomes into the two buckets $A^\star$ (probabilities $a = P(A^\star), b = Q(A^\star)$ with $a - b = \textrm{TV}(P, Q)$) and $(A^\star)^c$. By the log-sum inequality ($\sum_i v_i \log\frac{v_i}{w_i} \ge (\sum_i v_i)\log\frac{\sum_i v_i}{\sum_i w_i}$):
  $$D_{\textrm{KL}}(P \| Q) \ge D_{\textrm{KL}}(\textrm{Ber}(a) \| \textrm{Ber}(b)) \ge 2(a - b)^2 = 2\,\textrm{TV}(P, Q)^2 \quad \blacksquare$$
:::

::: {.slide title="Sharpness and Limits of Pinsker's Inequality"}
[Monte Carlo Verification]{.kicker}

Across $10{,}000$ random 5-outcome pairs, $\textrm{TV}/\sqrt{\textrm{KL}/2}$ has median $0.81$ and never exceeds $1$, while nearly fair coins ($\delta \to 0$) prove that the constant $\tfrac12$ cannot be improved:

@!divergences-distances-pinsker
:::

::: {.slide title="The Data-Processing Inequality for f-Divergences"}
[Monotonicity Under Channels]{.kicker}

Can a feature extractor, quantizer, or randomized transformation ever make two distributions $P$ and $Q$ easier to distinguish?

- **Markov channel transformation:** Let $\kappa(y \mid x)$ be any transition kernel mapping $x \mapsto y$, producing output distributions $(P\kappa)(y) = \sum_x \kappa(y \mid x)p(x)$ and $(Q\kappa)(y) = \sum_x \kappa(y \mid x)q(x)$.
- **Universal contraction theorem:** Applying Jensen's inequality under the posterior weights $\frac{\kappa(y \mid x)q(x)}{(Q\kappa)(y)}$ for each output $y$ proves that for *every* $f$-divergence:
  $$D_f(P\kappa \,\|\, Q\kappa) \le D_f(P \| Q)$$
- **Two key consequences for deep learning:**
  1. Any divergence measured on extracted features $\phi(x)$ is a *lower bound* on the raw-space divergence $D_f(P \| Q)$.
  2. Any invertible transformation $\phi$ acts as a channel in both directions, forcing $D_f(\phi_\# P \| \phi_\# Q) = D_f(P \| Q)$ (**coordinate scale invariance**).
:::

::: {.slide title="Legendre–Fenchel Conjugate and Tangent Bounds"}
[Convex Duality Foundations]{.kicker}

How can we estimate $D_f(P \| Q)$ when we have only samples from $P$ and $Q$ and cannot evaluate $p(x)/q(x)$?

::: {.cols .vc}
::: {.col}
- **Convex conjugate:** Every convex function $f(u)$ is the upper envelope of its tangent lines of slope $t$ and intercept $-f^*(t)$:
  $$f^*(t) = \sup_{u}\big(ut - f(u)\big)$$
- **Fenchel–Young inequality:** For every $u$ and slope $t$:
  $$f(u) \ge ut - f^*(t)$$
  with equality at $t = f'(u)$.
- **Examples:** $f(u) = u\log u$ $\implies$ $f^*(t) = e^{t-1}$; $f(u) = (u-1)^2$ $\implies$ $f^*(t) = t + t^2/4$.
:::

::: {.col .fig}
@fig:mdl-it-f-gan-tangent-bound
:::
:::
:::

::: {.slide title="The Variational Critic Bound and f-GAN Training"}
[Estimating Divergences from Samples]{.kicker}

Applying the Fenchel–Young inequality $f(u) \ge ut - f^*(t)$ pointwise at $u = p(x)/q(x)$ with a learned **critic** $t = T(x)$ and multiplying by $q(x)$ eliminates the unknown density ratio:

$$q(x)\,f\!\left(\frac{p(x)}{q(x)}\right) \ge p(x)T(x) - q(x)f^*(T(x)) \implies D_f(P \| Q) \ge \mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}\big[f^*(T(x))\big]$$

- **Sample-only expectations:** Both terms on the right are simple averages over real data $x \sim P$ and generated samples $x \sim Q$.
- **Optimal critic $T^\star(x)$:** Equality holds pointwise when the critic matches the slope of $f$ at the true density ratio:
  $$T^\star(x) = f'\!\left(\frac{p(x)}{q(x)}\right)$$
  (e.g., $T^\star = 1 + \log(p/q)$ for forward KL; $T^\star = 2(p/q - 1)$ for Pearson $\chi^2$).
- **The $f$-GAN minimax game (Nowozin et al., 2016):** The critic $T$ maximizes the lower bound to track $D_f(P \| Q)$, while the generator producing $Q$ minimizes it.
:::

::: {.slide title="Recovering the Original GAN as a JS Dual"}
[Binary Cross-Entropy Discriminator]{.kicker}

The original GAN objective of Goodfellow et al. (2014) is a special case of the $f$-GAN critic bound:

- **1. Choose the generator and conjugate:** Take $f(u) = u\log u - (u+1)\log(u+1)$, whose conjugate over $t < 0$ is $f^*(t) = -\log(1 - e^t)$.
- **2. Parameterize the critic via a sigmoid discriminator $D(x) \in (0, 1)$:** Setting $T(x) = \log D(x)$ turns $-f^*(T(x))$ into $\log(1 - D(x))$, yielding the binary cross-entropy game:
  $$\mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[f^*(T(x))] = \mathbb{E}_{x \sim P}[\log D(x)] + \mathbb{E}_{x \sim Q}[\log(1 - D(x))]$$
- **3. Optimal discriminator and JS value:** Setting $T^\star(x) = f'(p/q) = \log\frac{p(x)}{p(x)+q(x)}$ gives the Bayes-optimal discriminator $D^\star(x) = \frac{p(x)}{p(x)+q(x)}$. Because $f(u) = 2f_{\textrm{JS}}(u) - (u-1)\log 2 - \log 4$, the value at $D^\star$ is exactly $2\,\textrm{JS}(P, Q) - \log 4$.
:::

::: {.slide title="Practical Pitfalls of Adversarial Critic Bounds"}
[Underestimation & Vanishing Gradients]{.kicker}

Training a $\chi^2$ critic by gradient ascent on $P = (0.6, 0.3, 0.1)$ vs. $Q = (0.2, 0.5, 0.3)$ ($\chi^2 = 1.0133$) exposes two structural limits of the variational bound:

@!divergences-distances-f-gan-bound

- **Downward bias & disjoint-support saturation:** Any suboptimal critic ($T \neq T^\star$, including $2T^\star$) underestimates $D_f$ by $\tfrac14\mathbb{E}_Q[(T - T^\star)^2]$; on disjoint supports, $T^\star$ is constant ($f'(\infty)$ on data, $f'(0)$ on model samples), giving the generator zero gradient!
:::

::: {.slide title="Concept Check: f-Divergences, Pinsker, Critics"}
[Check Your Understanding]{.kicker}

Work through these three calculations:

1. **Hand calculation:** For $P = (0.6, 0.3, 0.1)$ and $Q = (0.2, 0.5, 0.3)$, compute $\textrm{TV}(P, Q)$ and Pearson's $\chi^2(P \| Q)$. Which outcome dominates $\chi^2(P \| Q)$ and why?
2. **AI-text detector ceiling:** Suppose human text $P$ and LLM text $Q$ satisfy $D_{\textrm{KL}}(P \| Q) = 0.02\text{ nats}$. What is the highest accuracy any single-sample detector can achieve on a $50/50$ mix? Does passing the text through a feature extractor help?
3. **Interpreting a critic score:** A critic trained with the $\chi^2$ conjugate $f^*(t) = t + t^2/4$ reports $\mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)] = 0.70$ on a fresh held-out test set. What can you conclude about the true $\chi^2(P \| Q)$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **TV & $\chi^2$:** $\textrm{TV} = \tfrac12(0.4 + 0.2 + 0.2) = 0.40$ (attained on $A^\star = \{1\}$). $\chi^2 = \frac{0.4^2}{0.2} + \frac{0.2^2}{0.5} + \frac{0.2^2}{0.3} = 0.80 + 0.08 + 0.133 = 1.0133$; outcome $1$ contributes $0.80$ because $\chi^2$ divides by $q_1 = 0.2$.
2. **Detector ceiling:** By Pinsker, $\textrm{TV}(P, Q) \le \sqrt{0.02/2} = 0.10$, so max accuracy is $\tfrac12(1 + \textrm{TV}) \le 55\%$. By data processing, no feature extractor can increase $D_{\textrm{KL}}$ or $\textrm{TV}$.
3. **Critic bound:** On fresh samples, $\chi^2(P \| Q) \ge 0.70$ in expectation—it is a certified lower bound, but could be much larger if $T \neq T^\star$ (as at step 5 of our ascent where truth was $1.0133$).
:::
:::

::: {.slide title="Summary: The f-Divergence Family and Duals"}
[Key Takeaways]{.kicker}

| Concept / Tool | Mathematical Form | Key Property / Theorem | Deep Learning Role & Caveat |
|:--|:---|:---|:---|
| **$f$-Divergence** | $D_f(P \| Q) = \mathbb{E}_Q[f(p/q)]$ | $D_f \ge 0$ by Jensen; $D_f \approx \tfrac12 f''(1)\chi^2$ locally | Unifies KL, rev-KL, $\chi^2$, Hellinger, TV, JS; freezes at $f(0)+f'(\infty)$ on disjoint supports |
| **Forward vs. Reverse KL** | $\mathbb{E}_P[\log(p/q)]$ vs. $\mathbb{E}_Q[\log(q/p)]$ | $f'(\infty) = \infty$ (forward) vs. $f(0) = \infty$ (reverse) | Forward KL (MLE) covers all modes; Reverse KL (VI) locks onto one mode and underestimates variance |
| **Total Variation (TV)** | $\tfrac12\|p-q\|_1 = \sup_A |P(A)-Q(A)|$ | Optimal 1-sample test accuracy is $\tfrac12(1 + \textrm{TV})$ | True metric in $[0, 1]$; saturates at $1$ on disjoint supports |
| **Pinsker's Inequality** | $\textrm{TV}(P, Q) \le \sqrt{\tfrac12 D_{\textrm{KL}}(P \| Q)}$ | Sharp constant $1/2$ as $P \to Q$; no converse | Converts KL trust regions (PPO) into worst-case event bounds |
| **Data Processing** | $D_f(P\kappa \| Q\kappa) \le D_f(P \| Q)$ | Contraction under any channel $\kappa(y \mid x)$ | Feature divergences lower-bound input divergences; scale invariance |
| **Fenchel Critic Bound** | $D_f(P \| Q) \ge \mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ | Tight at $T^\star = f'(p/q)$; $2\,\textrm{JS} - \log 4$ for GAN | Enables sample-only adversarial training ($f$-GAN), but biased low for imperfect $T$ |
:::

