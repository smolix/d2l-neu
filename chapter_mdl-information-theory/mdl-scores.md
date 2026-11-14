# Score-Based Comparisons
:label:`sec_mdl-scores`

A softmax turns ten logits into probabilities by exponentiating them and
dividing by their sum, and nobody gives the sum a second thought. Now let a
network assign such a number not to ten classes but to every possible input.
By convention the sign is reversed: the network returns an energy
$E(\mathbf{x})$, low for plausible images where a logit would be high, and the
density is proportional to $\tilde{q}(\mathbf{x}) = e^{-E(\mathbf{x})}$. The
sum becomes an integral over all images,
$Z = \int \tilde{q}(\mathbf{x})\, d\mathbf{x}$, and nobody can compute it. Such
an *energy-based model* can say how plausible an input is, but not how
probable.

None of our comparisons applies to it. The likelihood needs the density
$q = \tilde{q}/Z$. The others, by a critic (:numref:`sec_mdl-f-gan-dual`), a
kernel or a transport plan (:numref:`sec_mdl-test-functions-transport`), need
samples of the model, and an energy does not hand out samples. Each one has to
be produced by a long random walk, many small noisy steps toward lower energy,
and every step needs a gradient of the network
(:numref:`sec_mdl-langevin-sampling`).

One such gradient is cheap, however, and it is all that we shall ask of the
model. The gradient of $\log \tilde{q}$ with respect to the input, the slope of
the log-density, costs one backward pass, and $Z$ has disappeared from it,
because a constant has no slope. This gradient is called the *score*, and this
section compares distributions through scores alone. The mean squared
difference of two scores is the objective of score matching, on which diffusion
models (:numref:`chap_diffusion`) are trained. It has a limit that anyone who
trains such a model should know. A slope is local, and it cannot tell how mass
is divided between modes that lie far apart. We measure how little it sees.

As before, $P$ is the data, which we hold as samples, and $Q$ is the model,
which we know through its score $s_Q$. We ask what a difference of slopes says
about two distributions and what it misses, and then how slopes can be compared
at all when the data, a finite set of points, have none.
:numref:`sec_mdl-score-matching` builds on these two answers. A third question,
how to test a sample against a model that cannot be sampled, can be skipped on
a first reading. The section ends with a table of training objectives and
advice on choosing among them.

```{.python .input #mdl-scores-score-based-comparisons-1}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-scores-score-based-comparisons-1}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-scores-score-based-comparisons-1}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-scores-score-based-comparisons-1}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mdl-scores-score-based-comparisons-2}
from scipy import integrate
```

The examples are Gaussians and mixtures of two Gaussians on the line. The
helper functions evaluate the log-density of a mixture, with the largest term
factored out so that the tails do not underflow
(:numref:`subsec_mdl-stable-softmax`), and the KL divergence between two
densities on a grid, in nats.

```{.python .input #mdl-scores-score-based-comparisons-3}
def log_components(x, w, mus, sigmas):
    """log(w_k N(x; mu_k, sigma_k^2)) for every component k; shape (k, len(x))."""
    return (np.log(w)[:, None] - 0.5 * np.log(2 * np.pi * sigmas[:, None] ** 2)
            - (x - mus[:, None]) ** 2 / (2 * sigmas[:, None] ** 2))

def log_mix(x, w, mus, sigmas):
    """Log-density of the mixture sum_k w_k N(mu_k, sigma_k^2), computed stably."""
    c = log_components(x, w, mus, sigmas)
    c_max = c.max(axis=0)
    return c_max + np.log(np.exp(c - c_max).sum(axis=0))

def kl(log_a, log_b, x):
    """D_KL(A||B) by quadrature on the grid x; in nats."""
    return integrate.trapezoid(np.exp(log_a) * (log_a - log_b), x)
```

## The Score and the Fisher Divergence
:label:`sec_mdl-fisher-divergence`

### The Score

The score answers one question at every point: in which direction does the
density rise, and how steeply? For a distribution $P$ with a differentiable
density $p$, the **score** is the gradient of the log-density *with respect to
the data point*:

$$
s_P(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}).
$$
:eqlabel:`eq_mdl-score-def`

(In classical statistics, "score" means the gradient of the log-likelihood
with respect to the *parameters*, as in :numref:`sec_mdl-maximum_likelihood`.
Here the gradient is in $\mathbf{x}$.) For a Gaussian
$\mathcal{N}(\mu, \sigma^2)$ on the line it is

$$
s(x) = -\frac{x - \mu}{\sigma^2},
$$

a spring pulling toward the mean with stiffness $1/\sigma^2$. For a mixture
$p = \sum_k w_k\, p_k$ the chain rule gives $s(x) = \sum_k r_k(x)\, s_k(x)$
with $r_k(x) = w_k p_k(x) / p(x)$: each component's spring, weighted by the
posterior probability, or *responsibility*, that $x$ came from that component.
Near a mode one responsibility is close to $1$ and the point feels that mode's
spring alone, as :numref:`fig_mdl-score-field` shows. The code is the same
formula, with the responsibilities computed from log-densities.

![The score field of a two-component Gaussian mixture. Arrows show the gradient of the log-density: near each mode the field is a spring pulling toward that mode's mean, and along the valley between the modes the responsibilities hand the point off from one attractor to the other. Density contours are underlaid for reference.](../img/mdl-it-score-field.svg)
:label:`fig_mdl-score-field`

```{.python .input #divergences-distances-score}
def mix_score(x, w, mus, sigmas):
    """Score of a Gaussian mixture: responsibility-weighted component scores."""
    r = np.exp(log_components(x, w, mus, sigmas) - log_mix(x, w, mus, sigmas))
    return np.sum(r * -(x - mus[:, None]) / sigmas[:, None] ** 2, axis=0)
```

The score does not depend on a normalizing constant, and that is why we are
interested in it. If $q(\mathbf{x}) = \tilde{q}(\mathbf{x}) / Z$, then

$$
\nabla_{\mathbf{x}} \log q(\mathbf{x})
= \nabla_{\mathbf{x}} \log \tilde{q}(\mathbf{x})
- \nabla_{\mathbf{x}} \log Z
= \nabla_{\mathbf{x}} \log \tilde{q}(\mathbf{x}),
$$

since $Z$ does not depend on $\mathbf{x}$. The unnormalized density
$\tilde{q}(x) = e^{-x^4/4}$, for instance, has the score $-x^3$, and we never
had to find out that $Z \approx 2.56$.

### The Fisher Divergence

A comparison built from scores can therefore be applied to an unnormalized
model. The natural candidate is the mean squared difference of the two vector
fields under the data, the **Fisher divergence**:

$$
D_{\textrm{F}}(P\|Q) = \frac{1}{2}\, \mathbb{E}_{\mathbf{x} \sim P}\!\left[
\big\| s_P(\mathbf{x}) - s_Q(\mathbf{x}) \big\|^2 \right].
$$
:eqlabel:`eq_mdl-fisher-div-def`

It is non-negative, and it is a divergence, zero only for $P = Q$
(:numref:`sec_mdl-what-is-a-divergence`), when both densities are positive on
the same connected region. There, equal scores force $\log p - \log q$ to be
constant, and normalization makes the constant zero.

Two Gaussians show how it differs from the KL divergence. With equal variance
$\sigma^2$ and means a distance $\Delta$ apart, the scores differ by the
constant $\Delta/\sigma^2$ at every $x$, so

$$
D_{\textrm{F}} = \frac{\Delta^2}{2\sigma^4}
\qquad \textrm{whereas} \qquad
D_{\textrm{KL}} = \frac{\Delta^2}{2\sigma^2}
$$

by the formula for two Gaussians in :numref:`sec_mdl-information_theory`. At
$\sigma = 1$ the two agree. Measure $x$ in units half as large, and $\Delta$
and $\sigma$ both double: the KL divergence stays where it was, and the Fisher
divergence falls to a quarter. It is measured in units of $1/x^2$, as a squared
slope must be, and not in nats. Exercise
:exref:`ex_mdl-divergences-distances-fisher-stein`a treats unequal variances.

### Mixing Weights of Distant Modes

The average in :eqref:`eq_mdl-fisher-div-def` runs over the data, and what it
averages is a difference of slopes. For data with several modes the two facts
combine badly. Take two mixtures with the same components and swapped weights,

$$
P_w = w\,\mathcal{N}(-m, 1) + (1 - w)\,\mathcal{N}(m, 1),
\qquad w = 0.7 \textrm{ against } w = 0.3.
$$

As distributions the two are far apart. Once the modes are well separated,
their KL divergence is that of the weight vectors $(0.7, 0.3)$ and
$(0.3, 0.7)$, which is
$0.7 \log\tfrac{7}{3} + 0.3 \log\tfrac{3}{7} = 0.339$ nats, whatever $m$ is.

The scores hardly differ. Near each mode the responsibility of that mode is
close to $1$ in both mixtures, so both scores are that mode's spring. Written
out, the score of $P_w$ is

$$
s_w(x) = -x + m \tanh(m x - c_w), \qquad c_w = \tfrac{1}{2} \log\frac{w}{1 - w}.
$$

This is the spring $-(x + m)$ far to the left, the spring $-(x - m)$ far to
the right, and a hand-over between them around $x = c_w/m$. The weight enters
through the place of the hand-over and through nothing else. At $m = 5$ the
hand-over sits at $x = 0.085$ for $w = 0.7$ and at $x = -0.085$ for $w = 0.3$,
in the valley between the modes, where the density is
$e^{-m^2/2}/\sqrt{2\pi}$, about $1.5 \times 10^{-6}$. The data almost never
visit the one place where the two scores disagree. We compute both
divergences by quadrature as the modes move apart, and expect the Fisher
divergence to fall roughly like $e^{-m^2/2}$.

```{.python .input #mdl-divergences-distances-the-score-and-the-fisher-divergence-1}
xs = np.linspace(-14.0, 14.0, 14001)
w_a, w_b, unit = np.array([0.7, 0.3]), np.array([0.3, 0.7]), np.ones(2)
for m in (1.0, 3.0, 5.0):
    mus_m = np.array([-m, m])
    log_a, log_b = log_mix(xs, w_a, mus_m, unit), log_mix(xs, w_b, mus_m, unit)
    score_gap = mix_score(xs, w_a, mus_m, unit) - mix_score(xs, w_b, mus_m, unit)
    fisher = 0.5 * integrate.trapezoid(np.exp(log_a) * score_gap ** 2, xs)
    print(f'm = {m:.0f}: KL = {kl(log_a, log_b, xs):.3f} nats, '
          f'Fisher = {fisher:.1e}')
```

The KL divergence settles at $0.339$ nats, and the Fisher divergence falls by
more than four orders of magnitude, from about $0.1$ to about
$4 \times 10^{-6}$. A loss built from scores cannot see how mass is divided
between distant modes. The word "connected" above was a warning: on two
separate regions, equal scores allow $\log p - \log q$ a constant of its own on
each. A model fitted by score matching to such data can give both modes the
right place and the right shape and still hold $30\%$ of its mass where the
data hold $70\%$ :cite:`Wenliang.Kanagawa.2020`. Gaussian noise added to the
data fills the valley (Exercise
:exref:`ex_mdl-divergences-distances-score-blindness`c), which is one reason
why diffusion models perturb the data with noise at many scales before they
match scores; :numref:`sec_diffusion-denoising` measures the same blind spot on
trained networks.

## Stein's Identity
:label:`sec_mdl-stein-discrepancy`

The Fisher divergence has a second defect, which stands in the way of any use
at all: it contains $s_P$, the score of the data, and data come as samples. A
finite set of points has no slope. To see where $s_P$ enters, we expand the
square. On the line,

$$
D_{\textrm{F}}(P\|Q) = \tfrac{1}{2}\, \mathbb{E}_P\big[s_P^2\big]
- \mathbb{E}_P\big[s_P\, s_Q\big] + \tfrac{1}{2}\, \mathbb{E}_P\big[s_Q^2\big].
$$
:eqlabel:`eq_mdl-fisher-expanded`

The last term is an average over the data of a function that the model
supplies, and a sample estimates it. The first does not depend on the model,
so it plays no part when models are fitted or ranked. The obstacle is the
cross term in the middle, where the unknown score of the data multiplies the
score of the model. An identity that every smooth density satisfies with its
own score removes it. We state it for one density on the line and then use it
twice: for the data, to remove the cross term, and for the model, to test a
sample.

**Proposition (Stein's identity).** *Let $p > 0$ be a differentiable density
on $\mathbb{R}$ with score $s = (\log p)'$, and let $g$ be differentiable with
$g(x)\, p(x) \to 0$ as $x \to \pm\infty$, $\mathbb{E}_p|g'| < \infty$ and
$\mathbb{E}_p|g\, s| < \infty$. Then*

$$
\mathbb{E}_{x \sim p}\big[ g'(x) + g(x)\, s(x) \big] = 0.
$$
:eqlabel:`eq_mdl-stein-identity`

**Proof.** The score is $s = p'/p$, so $p' = s\, p$. One integration by
parts (:numref:`subsec_mdl-integration-by-parts`) gives

$$
\int g'\, p\, dx = \big[ g\, p \big]_{-\infty}^{\infty} - \int g\, p'\, dx
= -\int g\, s\, p\, dx,
$$

where the boundary term vanishes and both integrals converge by the conditions
on $g$. $\blacksquare$

For the standard normal density the score is $-x$ and the identity reads
$\mathbb{E}[g'(X)] = \mathbb{E}[X g(X)]$, which is Stein's lemma
:cite:`Stein.1981`. With $g(x) = x$ it says $\mathbb{E}[X^2] = 1$. The
identity turns knowledge of a slope into statements about averages, and
averages are what samples estimate.

### Removing the Data Score

Suppose that the data distribution $P$ has a smooth density, even though we
see only samples of it. Then the identity holds for the data, with $s = s_P$,
and we may take the model's score as the test function, $g = s_Q$. It reads
$\mathbb{E}_P[s_Q' + s_Q\, s_P] = 0$. The cross term of
:eqref:`eq_mdl-fisher-expanded` is therefore $-\mathbb{E}_P[s_Q']$, an average
that no longer contains $s_P$, and

$$
D_{\textrm{F}}(P\|Q) = \mathbb{E}_{x \sim P}\Big[ \tfrac{1}{2}\, s_Q(x)^2 + s_Q'(x) \Big]
+ \tfrac{1}{2}\, \mathbb{E}_P\big[s_P^2\big].
$$
:eqlabel:`eq_mdl-score-matching-line`

The derivative has moved from the data, which have none, to the model. The
last term is the same for every model, and the rest is an average over the
data of quantities that the model supplies: it can be estimated from a sample
and minimized, with no $Z$ and no $s_P$. This is *score matching*
:cite:`Hyvarinen.2005`.

The quartic of :numref:`sec_mdl-fisher-divergence` shows what has been gained.
Give it a parameter, $\tilde{q}_\theta(x) = e^{-\theta x^4/4}$, so that
$s_Q(x) = -\theta x^3$ and $s_Q'(x) = -3\theta x^2$. With a bar for an average
over the data, the part of :eqref:`eq_mdl-score-matching-line` that depends on
the model is estimated by

$$
\overline{\tfrac{1}{2}\, s_Q^2 + s_Q'}
= \tfrac{1}{2}\, \theta^2\, \overline{x^6} - 3\, \theta\, \overline{x^2},
\qquad \textrm{which is smallest at} \qquad
\hat\theta = \frac{3\, \overline{x^2}}{\overline{x^6}}.
$$

Two moments of the sample fit the model, and nobody had to integrate
$e^{-\theta x^4/4}$. Maximum likelihood needs $Z(\theta)$ and arrives at a
different function of the data (Exercise
:exref:`ex_mdl-scores-unnormalized-quartic`). For a Gaussian model the two
fits coincide (Exercise :exref:`ex_mdl-scores-gaussian-score-matching`).

On the line, $s_Q'$ is one more derivative. In $\mathbb{R}^n$ it becomes a sum
of $n$ second derivatives, one for each coordinate, and the single backward
pass that gave the score turns into $n$ of them, far too many for an image.
:numref:`sec_mdl-score-matching` carries the argument to $\mathbb{R}^n$, shows
how noise added to the data avoids these derivatives, and derives the training
loss of diffusion models from the result.

### Testing a Sample

The identity can also *test* a sample against a model. Stated for the model,
it says that for every admissible $g$ the function $g' + g\, s_Q$ averages to
zero under $Q$. If its average over a sample is far from zero for some $g$,
the sample did not come from $Q$. The check needs the model's score at the
sample points and nothing else, neither $Z$ nor samples of the model.

How much can one test function detect? Let the model be
$Q = \mathcal{N}(0, 1)$, so that the function to average is
$g'(x) - x\, g(x)$. For $g = 1$ this is $-x$, and the test checks the mean of
the sample. For $g(x) = x$ it is $1 - x^2$, and the test checks the second
moment against $1$. As a third test function we take $g(x) = \sin x$, which
has no such simple reading. The cell draws $200{,}000$ points from the model
itself, from $\mathcal{N}(0.5, 1)$, which is shifted, and from
$\mathcal{N}(0, 4)$, which is twice as wide, and prints every average with its
standard error. From the two moments we expect $-0.5$ and $1 - 1.25 = -0.25$
on the shifted sample, and $0$ and $1 - 4 = -3$ on the wide one.

```{.python .input #divergences-distances-stein}
rng = np.random.default_rng()
n = 200_000
# Stein operator of the model Q = N(0, 1), whose score is -x:  g'(x) - x g(x)
tests = [('g = 1', np.ones_like, np.zeros_like),
         ('g = x', lambda t: t, np.ones_like),
         ('g = sin x', np.sin, np.cos)]
samples = [('N(0, 1)', rng.normal(0.0, 1.0, n)),
           ('N(0.5, 1)', rng.normal(0.5, 1.0, n)),
           ('N(0, 4)', rng.normal(0.0, 2.0, n))]
print(f'{"sample":10s}' + ''.join(f'{name:>18s}' for name, _, _ in tests))
for s_name, x in samples:
    vals = [g_prime(x) - x * g(x) for _, g, g_prime in tests]
    print(f'{s_name:10s}' + ''.join(f'{v.mean():+10.3f} ± {v.std() / np.sqrt(n):.3f}'
                                    for v in vals))
```

On the model's own sample no average is further from zero than a few of its
standard errors. All three test functions catch the shifted sample, at about
$-0.50$, $-0.25$ and $-0.15$. The wide sample is another matter. Its mean is
zero, so $g = 1$ reports no more than a hundredth or so, nothing like the
$-0.5$ of the shift, although a sample with four times the model's variance is
hardly a subtle violation. Only $g = x$, at about $-3$, and $\sin x$, at about
$-0.4$, notice it. A test function checks one property of the sample, and a
sample can be wrong in a property that the function does not check.

The table has a short explanation. A sample from a distribution $P$ with a
smooth density obeys the identity of $P$,
$\mathbb{E}_P[g'] = -\mathbb{E}_P[g\, s_P]$, and therefore

$$
\mathbb{E}_P\big[ g' + g\, s_Q \big] = \mathbb{E}_P\big[ (s_Q - s_P)\, g \big].
$$
:eqlabel:`eq_mdl-stein-score-diff`

The average is the difference of the two scores, weighted by $g$. For the
shifted sample the difference is the constant $-x + (x - 0.5) = -0.5$, and
$g = 1$ picks it up in full. For the wide sample it is
$-x + x/4 = -\tfrac{3}{4}x$, an odd function, which the even function $g = 1$
averages to zero. No single test function is correlated with every possible
difference of scores.

## The Kernel Stein Discrepancy
:label:`sec_mdl-ksd`

One test function checks one property, so we use many at once: all functions
in the unit ball of a feature space, and among them the one whose average is
largest. :numref:`sec_mdl-ipm-mmd` did the same for the difference of two
means and found a closed form, the maximum mean discrepancy (MMD). There is one
here as well, which is rather remarkable. A supremum over infinitely many test
functions, applied to a model without a normalizing constant, reduces to kernel
evaluations on pairs of data points.

Write the *Stein operator* of the model $Q$ as
$(\mathcal{A}_Q g)(x) = g'(x) + s_Q(x)\, g(x)$, so that Stein's identity says
$\mathbb{E}_Q[\mathcal{A}_Q g] = 0$, and measure the data distribution $P$ by
$\sup_g \mathbb{E}_P[\mathcal{A}_Q g]$, the largest amount by which the data
violate the identity of the model.

For $g(x) = \langle \mathbf{w}, \boldsymbol{\varphi}(x) \rangle$ with a feature
vector $\boldsymbol{\varphi}$ and $\|\mathbf{w}\| \leq 1$, the operator acts on
the features:
$(\mathcal{A}_Q g)(x) = \langle \mathbf{w}, \boldsymbol{\psi}(x) \rangle$ with
$\boldsymbol{\psi} = \boldsymbol{\varphi}' + s_Q\, \boldsymbol{\varphi}$. By
the Cauchy--Schwarz inequality the supremum of
$\langle \mathbf{w}, \mathbb{E}_P[\boldsymbol{\psi}] \rangle$ over the ball is
$\big\| \mathbb{E}_P[\boldsymbol{\psi}] \big\|$, and the square of this norm is
$\mathbb{E}\langle \boldsymbol{\psi}(x), \boldsymbol{\psi}(x') \rangle$ for
independent draws $x, x'$ from $P$. Multiplying out leaves four inner products
of features, and each is the kernel
$k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x') \rangle$
or one of its derivatives:

$$
\langle \boldsymbol{\varphi}'(x), \boldsymbol{\varphi}(x') \rangle = \partial_x k, \qquad
\langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}'(x') \rangle = \partial_{x'} k, \qquad
\langle \boldsymbol{\varphi}'(x), \boldsymbol{\varphi}'(x') \rangle = \partial_x \partial_{x'} k.
$$

Their sum, with the scores as coefficients, is the **Stein kernel** of $Q$,

$$
k_Q(x, x') = s_Q(x)\, s_Q(x')\, k(x, x') + s_Q(x)\, \partial_{x'} k(x, x')
+ \partial_x k(x, x')\, s_Q(x') + \partial_x \partial_{x'} k(x, x'),
$$
:eqlabel:`eq_mdl-stein-kernel`

and the squared **kernel Stein discrepancy** (KSD) is its average over two
independent draws from the data:

$$
\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}\big[ k_Q(x, x') \big].
$$
:eqlabel:`eq_mdl-ksd-def`

The average is taken under the data alone. A sample $x_1, \ldots, x_N$ from
$P$ therefore gives the unbiased estimator

$$
\widehat{\mathrm{KSD}}^2 = \frac{1}{N(N-1)} \sum_{i \neq j} k_Q(x_i, x_j),
$$

the same average over pairs without self-pairs as the MMD estimator.

To see what the KSD measures, we apply :eqref:`eq_mdl-stein-score-diff` to
each feature. It gives
$\mathbb{E}_P[\boldsymbol{\psi}] = \mathbb{E}_P[(s_Q - s_P)\, \boldsymbol{\varphi}]$,
and squaring the norm as before,

$$
\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}\big[ (s_Q - s_P)(x)\, k(x, x')\, (s_Q - s_P)(x') \big].
$$
:eqlabel:`eq_mdl-ksd-score-diff`

The KSD is a kernel-smoothed relative of the Fisher divergence
$D_{\textrm{F}}(P\|Q)$: the same difference of scores under the same data
distribution, multiplied across pairs of points with weight $k$ where the
Fisher divergence squares it at each point. The blind spot comes along, and it
is deeper. On the two mixtures of :numref:`sec_mdl-fisher-divergence` both
points of a pair must now fall into the valley where the scores differ, and
$\mathrm{KSD}^2$ decays roughly like $e^{-m^2}$, the square of the rate of
the Fisher divergence (Exercise
:exref:`ex_mdl-divergences-distances-score-blindness`e).

The formula also gives us a value to predict, provided that we know both
scores. For this reason, and for no other, the models of our experiment are
Gaussians, whose normalizing constants everybody knows. For data
$P = \mathcal{N}(0, 1)$ and the wrong model $\mathcal{N}(1, 1)$ the difference
of scores is $-(x - 1) - (-x) = 1$ at every point, so
$\mathrm{KSD}^2 = \mathbb{E}[k(X, X')]$. For the RBF kernel
$k(x, x') = e^{-(x - x')^2/(2\ell^2)}$ with length scale $\ell$ this is the
Gaussian integral of :numref:`sec_mdl-ipm-mmd`: the difference $X - X'$ has
the distribution $\mathcal{N}(0, 2)$, and the average of the kernel over it is
$(1 + 2/\ell^2)^{-1/2}$. At $\ell = 1$ we obtain
$\mathrm{KSD}^2 = 1/\sqrt{3} \approx 0.577$. Against the correct model the
difference of scores vanishes and $\mathrm{KSD}^2 = 0$.

For the RBF kernel on the line the derivatives in :eqref:`eq_mdl-stein-kernel`
are $\partial_{x'} k = \frac{x-x'}{\ell^2}\, k$,
$\partial_x k = -\frac{x-x'}{\ell^2}\, k$ and
$\partial_x \partial_{x'} k = \big( \frac{1}{\ell^2} - \frac{(x-x')^2}{\ell^4} \big)\, k$
(the code calls the second argument `y`). We test one sample of $2{,}000$
points from $\mathcal{N}(0, 1)$ against the correct model $\mathcal{N}(0, 1)$,
whose score is $-x$, and against the wrong model $\mathcal{N}(1, 1)$, whose
score is $-(x - 1)$. The function receives each model as a score function and
as nothing else. Handed `lambda t: -t**3`, it would test the same sample
against the quartic, and it would not ask for $Z$.

```{.python .input #divergences-distances-ksd}
def ksd2_ustat(x, score, ell=1.0):
    """U-statistic KSD^2 with the RBF kernel exp(-(a-b)^2 / (2 ell^2))."""
    d = x[:, None] - x[None, :]
    k = np.exp(-d ** 2 / (2 * ell ** 2))
    dk_dx = -d / ell ** 2 * k
    dk_dy = d / ell ** 2 * k
    d2k = (1 / ell ** 2 - d ** 2 / ell ** 4) * k
    s = score(x)
    kappa = (s[:, None] * s[None, :] * k + s[:, None] * dk_dy
             + dk_dx * s[None, :] + d2k)
    n = len(x)
    return (kappa.sum() - np.trace(kappa)) / (n * (n - 1))

x_p = rng.standard_normal(2000)          # the data sample, P = N(0, 1)
for name, s, pop in [('true model  N(0,1)', lambda t: -t, 0.0),
                     ('wrong model N(1,1)', lambda t: -(t - 1), 1 / np.sqrt(3))]:
    print(f'KSD^2 vs the {name}: {ksd2_ustat(x_p, s):+.4f}  '
          f'(population {pop:.4f})')
```

Against the correct model the estimate is within a few thousandths of zero and
may be slightly negative, as an unbiased estimate of zero must sometimes be.
Against the wrong model it lands close to the predicted
$1/\sqrt{3} \approx 0.58$, within about $0.1$. Between the two lines of output
only the score function changed. The sample is the same, and neither model was
sampled.

With the RBF kernel a value of zero can be trusted. For data with a smooth
density, and under mild conditions on the two scores,
$\mathrm{KSD}^2(P, Q) = 0$ if and only if $P = Q$
:cite:`Liu.Lee.Jordan.2016,Chwialkowski.Strathmann.Gretton.2016`. The two
mixtures are a reminder that the value can nevertheless be tiny for
distributions that are far apart.

In $\mathbb{R}^n$ only the notation changes: $g$ is a vector field, the
operator is $s_Q^\top g + \nabla \cdot g$, the products in
:eqref:`eq_mdl-stein-kernel` become inner products of vectors, and the last
term becomes $\sum_i \partial_{x_i} \partial_{x'_i} k$. The KSD serves as a
goodness-of-fit test for unnormalized models and as a check on the output of a
sampler. Applied to the kernel itself, the Stein operator also gives a
direction in which to move the points of a sample toward the model, a method
called Stein variational gradient descent :cite:`Liu.Wang.2016`. No training
objective in this book uses the KSD.

## Training Objectives and Their Divergences
:label:`sec_mdl-divergence-objective-map`

:numref:`tab_mdl-divergence-objectives` collects this section and the two
before it. Each row names an objective, what it needs from the model and the
data, and the divergence that it minimizes when its averages are taken over
the whole distributions and, for the adversarial objectives, the critic is the
best possible. A critic, as in :numref:`sec_mdl-f-gan-dual`, is a function
trained on the two samples, in practice a second network. The fourth column
returns to the test case with which
:numref:`sec_mdl-divergences-distances` began, the data at a single point,
$P = \delta_0$, and the model's samples at a point a distance $d$ away,
$Q = \delta_d$. A comparison that does not change with $d$ gives such a model
no gradient toward the data.

:Training objectives, ordered by what they need, and the divergences they minimize when averages are exact and the critic is optimal. The column on point masses is the value for $\delta_0$ against $\delta_d$ with $d \neq 0$; $f(0)$ and $f'(\infty)$ are the limits of $f(u)$ at $0$ and of $f(u)/u$ at infinity, and the MMD entry uses the RBF kernel with length scale $\ell$.
:label:`tab_mdl-divergence-objectives`

| What the objective needs | Objective | Divergence | Two point masses $d$ apart | Treated in |
|:----|:----|:----|:----|:----|
| model density at data points | maximum likelihood (autoregressive models, flows) | forward KL | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |
| samples of both, through a critic | original GAN | Jensen--Shannon | $\log 2$, constant | :numref:`sec_mdl-f-gan-dual` |
| samples of both, through a critic | f-GAN | the chosen $D_f$ | $f(0) + f'(\infty)$, constant | :numref:`sec_mdl-f-gan-dual` |
| samples of both, no critic for a fixed kernel | moment matching, two-sample tests | MMD | $\textrm{MMD}^2 = 2 - 2e^{-d^2/(2\ell^2)}$, flat beyond a few $\ell$ | :numref:`sec_mdl-ipm-mmd` |
| samples of both, a 1-Lipschitz critic | WGAN | $W_1$ | $d$ | :numref:`sec_mdl-optimal-transport` |
| model samples, the model's density at them, the unnormalized target | variational inference, VAE posterior | reverse KL | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |
| model score, data samples | score matching, diffusion models | Fisher | undefined (a point mass has no score) | :numref:`sec_mdl-fisher-divergence` |
| model score, data samples | goodness-of-fit tests, sampler quality | KSD | undefined (a point mass has no score) | :numref:`sec_mdl-ksd` |

Read from the top, the table is advice, because its rows are ordered by what a
model can supply. A model whose density can be evaluated, and is positive
wherever data may fall, should be trained by maximum likelihood: forward KL
needs nothing but the data, its sample average is unbiased, and it leaves no
mode of the data uncovered. A model that only produces samples leaves three
choices. A critic gives an estimate that is biased low and goes flat when the
samples are far from the data (:numref:`sec_mdl-f-gan-dual`). A fixed kernel
needs no training and loses its gradient beyond a few length scales. A
Lipschitz critic is the choice when distance in the space of the data must
count.

The last three rows are for the cases in which a normalizing constant is
missing. When it is the target that is known only up to its constant, and the
model can be sampled and has a density, as in variational inference, reverse
KL applies, at the risk of settling on one mode. And a model known only
through its score calls for the Fisher divergence, which leaves the weights of
distant modes undetermined unless noise is added, as diffusion models do.

:numref:`fig_mdl-divergence-taxonomy` places the ratio, test-function and
transport families and the members they share. It does not draw the Fisher
divergence, which belongs to none of the three.

![The ratio, test-function and transport families and their common members. f-divergences average a convex function of the density ratio, integral probability metrics take a supremum of mean differences over a class of test functions, and optimal-transport distances measure the cost of moving mass. Total variation belongs to the first two families and Wasserstein-1 to the last two. (Total variation is also a transport cost, the one that charges $1$ for moving mass by any distance at all, which the figure does not draw.) Boxed names are metrics; JS and the squared Hellinger divergence $d_{\mathrm{H}}^2$ are squares of metrics.](../img/mdl-it-divergence-taxonomy.svg)
:label:`fig_mdl-divergence-taxonomy`

The table says which comparison an objective makes. It does not say how well
that comparison can be made from data. Every sample-based number of these
three sections, the critic bound, $\textrm{MMD}^2$ and $\mathrm{KSD}^2$, is an
estimate with an error, and the critic bound is biased low besides. The next
two sections turn from two distributions to two variables:
:numref:`sec_mdl-mutual-information` uses the KL divergence to measure how
much one variable tells about another, and :numref:`sec_mdl-mi-estimation`
asks how much of it a sample of $n$ pairs can certify. Without assumptions
on the distributions, the answer turns out to be a small multiple of
$\log n$ nats and no more.

## Summary

A model that is known only up to its normalizing constant can still be compared
with data through its score, the gradient of the log-density with respect to
the input, from which the constant drops out. The Fisher divergence is the mean
squared difference of two scores under the data, and Stein's identity, one
integration by parts, removes the unknown score of the data from it. Scores are
local: between two mixtures that differ only in the weights of two modes ten
standard deviations apart, the Fisher divergence is $4 \times 10^{-6}$ while
the KL divergence is $0.339$ nats. The kernel Stein discrepancy uses the same
identity to test a sample against a model through the model's score alone.
Which comparison to train on is decided by what the model can supply: a
density, samples, or a score.

## Exercises

1. **Fisher divergence and Stein moments.** The score of
   $\mathcal{N}(\mu, \sigma^2)$ is $-(x - \mu)/\sigma^2$.
    1. Compute the Fisher divergence :eqref:`eq_mdl-fisher-div-def` between
       $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$
       in closed form, and check that it reduces to
       $(\mu_1 - \mu_2)^2 / (2\sigma^4)$ for equal variances. Is the
       Fisher divergence symmetric?
    1. Apply Stein's identity :eqref:`eq_mdl-stein-identity` for
       $\mathcal{N}(0, 1)$ to $g(x) = x^k$, derive the recursion
       $\mathbb{E}[X^{k+1}] = k\, \mathbb{E}[X^{k-1}]$, and compute
       $\mathbb{E}[X^4]$ and $\mathbb{E}[X^6]$. What does $g(x) = 1$ give for
       a general density that satisfies the conditions of the identity?
    1. Let the data distribution $P$ on $\mathbb{R}$ have finite moments of
       all orders, and let $\mathbb{E}_P[g'(X) - X g(X)] = 0$ for every
       polynomial $g$. Explain why $P = \mathcal{N}(0, 1)$, using that the
       normal distribution is determined by its moments. Then let
       $P(X = 1) = P(X = -1) = \tfrac{1}{2}$, and show that it satisfies the
       identity for every polynomial $g$ of degree at most $2$ but not for
       $g(x) = x^3$. What does this imply for a goodness-of-fit test that
       checks the identity only on polynomials up to a fixed degree?
   :exlabel:`ex_mdl-divergences-distances-fisher-stein`

1. [code] **Score blindness to mixing weights.** ● Let
   $P_w = w\, \mathcal{N}(-m, 1) + (1 - w)\, \mathcal{N}(m, 1)$.
    1. The text computes $D_{\textrm{F}}(P_{0.7}\|P_{0.3})$ and
       $D_{\textrm{KL}}(P_{0.7}\|P_{0.3})$ for $m \in \{1, 3, 5\}$. Extend
       the computation to $m \in \{1, \ldots, 6\}$. Show that the KL
       divergence tends to $0.4 \log\tfrac{7}{3}$, the value $0.339$ of the
       text, and find the rate at which
       the Fisher divergence decays by fitting $\log D_{\textrm{F}}$ against
       $m^2$.
    1. Derive the score $s_w(x) = -x + m \tanh(m x - c_w)$ of the text from
       the responsibilities. Show that $s_{0.7}$ and $s_{0.3}$ differ most
       at $x = 0$, by $0.8\,m$, and substitute $u = m x$ in the Fisher
       divergence to show that it behaves like $C\, m\, e^{-m^2/2}$ for
       large $m$, with a constant $C$ that depends only on the weights.
       Compare with the rate you fitted in (a). What does the result imply
       for the mixing weights of a model fitted by score matching
       :cite:`Wenliang.Kanagawa.2020`?
    1. Add independent $\mathcal{N}(0, \sigma^2)$ noise to samples of both
       mixtures, so that each component becomes
       $\mathcal{N}(\pm m, 1 + \sigma^2)$. Repeat (a) at $m = 5$ for
       $\sigma \in \{0, 0.5, 1, 2\}$. How does the noise change the two
       divergences? Relate the result to the noise-perturbed score models
       of :citet:`song2019generative`.
    1. Write $P_t$ and $Q_t$ for two distributions after independent
       $\mathcal{N}(0, t)$ noise has been added to both. For two Gaussians
       of equal variance $\sigma^2$, which the noise raises to
       $\sigma^2 + t$, check from the two closed forms of the text that
       $\frac{d}{dt} D_{\textrm{KL}}(P_t\|Q_t) = -D_{\textrm{F}}(P_t\|Q_t)$.
       The identity holds for every pair of smooth densities. Confirm it by
       a finite difference for the two mixtures at $m = 5$ and
       $t \in \{1, 4, 9\}$, and conclude that
       $D_{\textrm{KL}}(P\|Q) = \int_0^\infty D_{\textrm{F}}(P_t\|Q_t)\, dt$.
       At which noise level is the Fisher divergence largest? What does the
       integral say about matching scores at one noise level against
       matching them at all levels :cite:`Song.Durkan.Murray.ea.2021`?
    1. The kernel Stein discrepancy has the same blind spot. Evaluate
       :eqref:`eq_mdl-ksd-score-diff` for the two mixtures by quadrature on a
       grid, with the RBF kernel at $\ell = 1$ and $m \in \{1, 3, 5\}$, and
       compare with the Fisher divergences of the text (at $m = 5$ you
       should find about $10^{-11}$). Where must both points of a pair fall
       to contribute, and how wide is that region compared with $\ell$?
       Conclude that $\mathrm{KSD}^2$ decays like $e^{-m^2}$. Would a
       smaller length scale help?
   :exlabel:`ex_mdl-divergences-distances-score-blindness`

1. **Score matching for a Gaussian model.** Let the data distribution $P$
   have mean $a$ and variance $v$, take the model
   $Q = \mathcal{N}(\mu, \sigma^2)$, and write
   $J = \mathbb{E}_P\big[\tfrac{1}{2} s_Q^2 + s_Q'\big]$ for the part of
   :eqref:`eq_mdl-score-matching-line` that depends on the model.
    1. Show that $J = \big(v + (a - \mu)^2\big)/(2\sigma^4) - 1/\sigma^2$
       and that it is smallest at $\mu = a$ and $\sigma^2 = v$, whatever
       else is true of $P$. Compare with the Gaussian that minimizes the
       forward KL divergence (:numref:`sec_mdl-fwd-vs-rev-kl`).
    1. The minimum is $J = -1/(2v)$, so the Fisher divergence from $P$ to
       its best Gaussian fit is
       $\tfrac{1}{2}\big(\mathbb{E}_P[s_P^2] - 1/v\big)$. Apply Stein's
       identity for $P$ to $g(x) = x - a$, and then the Cauchy--Schwarz
       inequality, to show that $\mathbb{E}_P[s_P^2] \geq 1/v$, with
       equality only if $P$ is Gaussian. Which density of a given variance
       has the smallest mean squared score?
    1. Write the model as
       $\tilde{q}_{\boldsymbol{\theta}}(x) = \exp(\theta_1 x + \theta_2 x^2)$.
       Show that $J$ is a quadratic function of $\boldsymbol{\theta}$, so
       that score matching solves two linear equations in the moments of
       the data. Show the same for every model of the form
       $\tilde{q}_{\boldsymbol{\theta}}(x) = \exp\big(\boldsymbol{\theta}^\top \boldsymbol{\phi}(x)\big)$.
       What would maximum likelihood have to compute instead?
   :exlabel:`ex_mdl-scores-gaussian-score-matching`

1. [code] **An unnormalized quartic.** Let
   $\tilde{q}_\theta(x) = e^{-\theta x^4/4}$ with $\theta > 0$.
    1. Compute the score. Use Stein's identity with $g(x) = x$ and with
       $g(x) = x^3$ to show that under the model
       $\mathbb{E}[X^4] = 1/\theta$ and
       $\mathbb{E}[X^6] = (3/\theta)\, \mathbb{E}[X^2]$, without computing
       the normalizing constant. Can the identity with polynomial $g$ also
       give $\mathbb{E}[X^2]$?
    1. The text finds the score-matching estimate
       $\hat\theta = 3\, \overline{x^2} / \overline{x^6}$, where a bar
       denotes an average over the data. Show that the normalizing constant
       is $Z(\theta) = \theta^{-1/4} Z(1)$ and that maximum likelihood gives
       $\hat\theta = 1 / \overline{x^4}$. Use (a) to argue that both
       estimates approach the true $\theta$ on a large sample from the
       model, although they are different functions of the data.
    1. If $U$ has the Gamma distribution with shape $\tfrac{1}{4}$ and
       scale $1$, then $(4U/\theta)^{1/4}$ with a random sign has the
       density $\tilde{q}_\theta / Z(\theta)$. Draw $1{,}000$ points at
       $\theta = 1$, compute both estimates, and repeat $2{,}000$ times.
       Which estimate has the smaller standard deviation, and by what
       factor? What does score matching give up for not needing $Z$?
   :exlabel:`ex_mdl-scores-unnormalized-quartic`

1. [code] **A polynomial Stein kernel.** Let the model be
   $Q = \mathcal{N}(0, 1)$ and take the kernel
   $k(x, x') = \sum_{j=0}^{J} x^j x'^j$, whose features are the monomials
   $1, x, \ldots, x^J$.
    1. Show that
       $\mathrm{KSD}^2(P, Q) = \sum_{j=0}^{J} \big(\mathbb{E}_P[j X^{j-1} - X^{j+1}]\big)^2$,
       a sum of squared averages of the kind that the experiment on Stein's
       identity prints, one for each monomial. Evaluate it for $J = 1$ on
       $P = \mathcal{N}(0.5, 1)$ and on $P = \mathcal{N}(0, 4)$, and compare
       with the first two columns of that experiment's output.
    1. Let $P(X = 1) = P(X = -1) = \tfrac{1}{2}$, as in Exercise
       :exref:`ex_mdl-divergences-distances-fisher-stein`c. Show that
       $\mathrm{KSD}^2(P, Q) = 0$ for $J \leq 2$ and that it is $4$ for
       $J = 3$. Why can a kernel with finitely many features not have the
       property that $\mathrm{KSD}^2 = 0$ only for $P = Q$?
    1. The Stein kernel :eqref:`eq_mdl-stein-kernel` needs no density of
       $P$. For the RBF kernel with $\ell = 1$ and the two-point
       distribution of (b), show that $k_Q(x, x) = 2$ and
       $k_Q(1, -1) = -8e^{-2}$, so that
       $\mathrm{KSD}^2(P, Q) = 1 - 4e^{-2} \approx 0.46$. Check the value
       with `ksd2_ustat` on $2{,}000$ draws. Which features of the RBF
       kernel (:numref:`sec_mdl-ipm-mmd`) detect what the monomials up to
       degree $2$ miss?
   :exlabel:`ex_mdl-scores-polynomial-stein-kernel`

<!-- Discussions: forum topic to be created at deploy -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.5]{.kicker}

Score-Based Comparisons<br>**comparisons that never see the normalizer**
:::
:::

::: {.slide title="A Model Known up to a Constant Has No Computable Likelihood"}
[The problem]{.kicker}

A network gives every input an energy $E(\mathbf x)$, low for plausible inputs. The model is

$$q(\mathbf x)=\tilde q(\mathbf x)/Z,\qquad\tilde q=e^{-E},\qquad Z=\int\tilde q(\mathbf x)\,d\mathbf x.$$

- For a $28\times28$ image, $Z$ is an integral over $784$ dimensions.
- Likelihood: needs $q$, hence $Z$.
- Critics, MMD, $W_1$: need samples of the model, each the end of a long random walk toward low energy.

::: {.d2l-note .warn}
What is left to compare when $Z$ is unknown?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the score of a model without its normalizing constant;
- compare two distributions by the Fisher divergence, and say what it misses;
- remove the unknown score of the data by one integration by parts;
- test a sample against a model that is known only through its score;
- choose a training objective from what the model can supply.

**Running examples:** $P_w=w\,\mathcal N(-m,1)+(1-w)\,\mathcal N(m,1)$ at $w=0.7$ against $w=0.3$; the model $\mathcal N(0,1)$ against shifted and widened samples.

**The idea:** compare slopes of log-densities. A constant has no slope.
:::

::: {.slide title="Slopes Are the Third Way to Compare, After Ratios and Averages"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Entropy, Cross-Entropy, and KL | surprise, KL, the loss floor |
| Codes, Compression, and Perplexity | the code behind the log-loss |
| f-Divergences | one convex function of the ratio |
| Test Functions and Transport | averages and moving mass |
| **Score-Based Comparisons** | **slopes, no normalizer** |
| Mutual Information | dependence and what it limits |
| Estimating Mutual Information | what samples can show |
:::

::: {.col .narrow}
- **Uses:** KL, kernels and MMD, integration by parts.
- **Skips:** score matching in many dimensions and the diffusion loss (Dynamics).
- $P$: data, as samples. $Q$: model, through its score $s_Q$.
:::
:::
:::

::: {.slide title="The Score Is a Field of Springs, and Z Drops Out of It"}
[The score]{.kicker}

::: {.cols .vc}
::: {.col}
- Score: $s_P(\mathbf x)=\nabla_{\mathbf x}\log p(\mathbf x)$, the gradient in the data point.
- $q=\tilde q/Z$: $\ \nabla\log q=\nabla\log\tilde q$, since $Z$ is constant in $\mathbf x$.
- Gaussian: $s(x)=-(x-\mu)/\sigma^2$, a spring of stiffness $1/\sigma^2$.
- Mixture: $s=\sum_kr_k\,s_k$ with responsibilities $r_k=w_kp_k/p$.
- $\tilde q=e^{-x^4/4}$: $s(x)=-x^3$; $Z\approx2.56$ never needed.
:::

::: {.col .fig}
@fig:mdl-it-score-field
:::
:::
:::

::: {.slide title="The Fisher Divergence Is the Mean Squared Gap Between Two Score Fields"}
[Fisher divergence]{.kicker}

$$D_{\mathrm F}(P\Vert Q)=\tfrac12\,\mathbb{E}_{\mathbf x\sim P}\big\Vert s_P(\mathbf x)-s_Q(\mathbf x)\big\Vert^2$$

- Averaged where the data are, and nowhere else.
- Two Gaussians, variance $\sigma^2$, means $\Delta$ apart: the scores differ by $\Delta/\sigma^2$.
- $D_{\mathrm F}=\dfrac{\Delta^2}{2\sigma^4}$, where $D_{\textrm{KL}}=\dfrac{\Delta^2}{2\sigma^2}$.

::: {.d2l-note}
Not in nats: the unit is $1/x^2$. Zero only for $P=Q$ when both densities are positive on one connected region.
:::
:::

::: {.slide title="Check: Fisher Against KL for Two Pairs of Gaussians"}
[Check]{.kicker}

Compute $D_{\mathrm F}=\Delta^2/(2\sigma^4)$ and $D_{\textrm{KL}}=\Delta^2/(2\sigma^2)$ for

- $\mathcal N(0,1)$ against $\mathcal N(1,1)$;
- $\mathcal N(0,4)$ against $\mathcal N(2,4)$, the same pair with $x$ doubled.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** First pair: both $\tfrac12$. Second pair: KL $=\tfrac4{2\cdot4}=\tfrac12$ again; Fisher $=\tfrac4{2\cdot16}=\tfrac18$.
KL ignores the scale of the axis. The Fisher divergence does not.
:::
:::
:::

::: {.slide title="The Fisher Divergence Cannot See a 70/30 Split of Distant Modes"}
[A blind spot]{.kicker}

$P_w=w\,\mathcal N(-m,1)+(1-w)\,\mathcal N(m,1)$, with $w=0.7$ against $w=0.3$. The score of $P_w$ is

$$s_w(x)=-x+m\tanh(mx-c_w),\qquad c_w=\tfrac12\log\tfrac w{1-w}.$$

- The weight only moves the hand-over between the two springs: $x=\pm0.085$ at $m=5$, where the density is $e^{-m^2/2}/\sqrt{2\pi}\approx1.5\times10^{-6}$.

@!mdl-divergences-distances-the-score-and-the-fisher-divergence-1

::: {.d2l-note .warn}
KL settles at $0.7\log\tfrac73+0.3\log\tfrac37=0.339$ nats. Noise added to the data fills the valley.
:::
:::

::: {.slide title="One Integration by Parts Removes the Score of the Data"}
[Stein's identity]{.kicker}

**I.** Any density $p$ with score $s=p'/p$, test function $g$: $\int g'p\,dx=[g\,p]_{-\infty}^{\infty}-\int g\,s\,p\,dx$, so $\;\boxed{\mathbb{E}_p[g'+g\,s]=0}$

. . .

**II.** On the line, $D_{\mathrm F}=\tfrac12\mathbb{E}_P[s_P^2]-\mathbb{E}_P[s_Q\,s_P]+\tfrac12\mathbb{E}_P[s_Q^2]$, and $s_P$ is unknown.

. . .

**III.** The identity for the data, with $s=s_P$ and $g=s_Q$, gives $\mathbb{E}_P[s_Q\,s_P]=-\mathbb{E}_P[s_Q']$:

$$D_{\mathrm F}(P\Vert Q)=\mathbb{E}_P\big[\tfrac12s_Q^2+s_Q'\big]+\textrm{const}$$

::: {.d2l-note}
Score matching: an average over data of what the model supplies. No $Z$, no $s_P$.
:::

[Assumes smooth positive densities and $g\,p\to0$ at $\pm\infty$.]{.small}
:::

::: {.slide title="Score Matching Fits a Quartic from Two Moments of the Data"}
[Worked example]{.kicker}

Model $\tilde q_\theta(x)=e^{-\theta x^4/4}$ with unknown $Z(\theta)$. A bar is an average over the data.

**I.** $s_Q=-\theta x^3$ and $s_Q'=-3\theta x^2$.

. . .

**II.** $\overline{\tfrac12s_Q^2+s_Q'}=\tfrac12\theta^2\,\overline{x^6}-3\theta\,\overline{x^2}$, a parabola in $\theta$.

. . .

**III.** Its minimum: $\;\boxed{\hat\theta=3\,\overline{x^2}\,/\,\overline{x^6}}$

::: {.d2l-note}
Two moments of the sample fit the model. Maximum likelihood needs $Z(\theta)$.
:::

[In $n$ dimensions $s_Q'$ becomes a sum of $n$ second derivatives: $n$ backward passes.]{.small}
:::

::: {.slide title="One Test Function Checks One Property of a Sample"}
[Testing a sample]{.kicker}

Model $Q=\mathcal N(0,1)$, $s_Q=-x$: under $Q$, $g'(x)-x\,g(x)$ averages to zero. Its average over $200{,}000$ sample points:

@!divergences-distances-stein

- $g=1$ checks the mean; $g=x$ checks the second moment: $1-1.25$ and $1-4$.
- $\mathbb{E}_P[g'+g\,s_Q]=\mathbb{E}_P[(s_Q-s_P)\,g]$: the score gap is $-0.5$ for the shift, $-\tfrac34x$ for the wide sample.

::: {.d2l-note .warn}
$g=1$ misses a sample with four times the model's variance.
:::
:::

::: {.slide title="The Largest Violation over a Kernel Ball Has a Closed Form"}
[Kernel Stein discrepancy]{.kicker}

- Test functions $g=\langle\mathbf w,\boldsymbol\varphi\rangle$, $\Vert\mathbf w\Vert\le1$; kernel $k(x,x')=\langle\boldsymbol\varphi(x),\boldsymbol\varphi(x')\rangle$.
- $\sup_g\mathbb{E}_P[g'+s_Q\,g]=\big\Vert\mathbb{E}_P[\boldsymbol\varphi'+s_Q\,\boldsymbol\varphi]\big\Vert$, and its square is

$$\mathrm{KSD}^2(P,Q)=\mathbb{E}_{x,x'\sim P}\big[k_Q(x,x')\big]$$

$$k_Q(x,x')=s_Q(x)\,s_Q(x')\,k+s_Q(x)\,\partial_{x'}k+\partial_xk\,s_Q(x')+\partial_x\partial_{x'}k$$

- Also $\mathrm{KSD}^2=\mathbb{E}\big[(s_Q-s_P)(x)\,k(x,x')\,(s_Q-s_P)(x')\big]$: a kernel-smoothed Fisher divergence.

::: {.d2l-note}
Needs the model's score and data samples. No $Z$, no samples of the model.
:::
:::

::: {.slide title="KSD Tells the Wrong Model from the Right One by Its Score Alone"}
[Kernel Stein discrepancy]{.kicker}

Data: $2{,}000$ points from $\mathcal N(0,1)$. RBF kernel, length scale $1$.

- Wrong model $\mathcal N(1,1)$: the score gap is $1$ everywhere, so $\mathrm{KSD}^2=\mathbb{E}[k(X,X')]=1/\sqrt3\approx0.577$.
- Right model $\mathcal N(0,1)$: $\mathrm{KSD}^2=0$.
- Gaussians only so that the value can be predicted: the score $-x^3$ would test the quartic.

@!divergences-distances-ksd

::: {.d2l-note}
Between the two lines only the score function changed. Neither model was sampled.
:::
:::

::: {.slide title="Check: Which Comparisons Survive an Unknown Normalizer?"}
[Check]{.kicker}

An energy-based model $\tilde q(\mathbf x)$ with unknown $Z$, and a sample of data.
Which of KL, JS, MMD, $W_1$, Fisher, KSD can you compute?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** KSD directly: model score, data samples. Fisher up to a
constant that does not depend on the model, after the integration by parts.
JS (through a critic), MMD and $W_1$ need samples of the model. KL from the
data to the model, the likelihood, needs $Z$.
:::
:::
:::

::: {.slide title="What the Model Can Supply Decides the Objective"}
[Choosing a comparison]{.kicker}

| the model supplies | train on | it compares | caveat |
|:--|:--|:--|:--|
| a density, positive on the data | maximum likelihood | forward KL | $\infty$ if data fall where $q=0$ |
| samples only | a critic (GAN, f-GAN) | JS, $D_f$ | biased low; flat when far |
| samples only | a fixed kernel | MMD | flat beyond a few length scales |
| samples only | a Lipschitz critic (WGAN) | $W_1$ | constraint is approximate |
| samples and density; target up to $Z$ | variational inference | reverse KL | may settle on one mode |
| a score | score matching, diffusion | Fisher | blind to weights of far modes |

::: {.d2l-note}
On two point masses a distance $d$ apart, only $W_1=d$ and, up to a few length scales, MMD still change with $d$.
:::

[A critic is a second network trained to tell the two samples apart.]{.small}
:::

::: {.slide title="Recap: Slopes Compare What Densities Cannot"}
[Wrap-up]{.kicker}

- **Score:** $\nabla_{\mathbf x}\log\tilde q$; the constant $Z$ drops out.
- **Fisher:** $\tfrac12\mathbb{E}_P\Vert s_P-s_Q\Vert^2$; Gaussians $\Delta^2/(2\sigma^4)$.
- **Blind spot:** at $m=5$, KL $0.339$ nats against Fisher $4\times10^{-6}$.
- **Stein:** $\mathbb{E}_Q[g'+g\,s_Q]=0$ removes the data score and tests a sample: $-0.5$ on a shift, $-3$ on a wide sample.
- **KSD:** $1/\sqrt3$ for the wrong model, $0$ for the right one, no sampling.

::: {.d2l-note}
Next: how much does one variable tell about another, and how much of it can $n$ samples certify?
:::
:::
