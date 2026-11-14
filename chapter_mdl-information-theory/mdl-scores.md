# Score-Based Comparisons
:label:`sec_mdl-scores`

<!-- WRITER: this file was cut from mdl-divergences-distances.md by the code step (text moved unchanged, headings promoted one level). Opening, summary, exercises to the band, cross-page :eqref:s and slides are the worklist items of 37.5. -->

An energy-based model specifies a density only up to its normalizing constant:
it gives $\tilde{q}(\mathbf{x})$, but not
$Z = \int \tilde{q}(\mathbf{x})\, d\mathbf{x}$. Without $Z$, the likelihood,
the KL divergence and every other f-divergence
(:numref:`sec_mdl-f-divergences`) are out of reach, and drawing samples needs a
Markov chain, so the sample-based comparisons of :numref:`sec_mdl-ipm-mmd` are
costly too. Here we compare distributions through the slopes of their
log-densities, which do not depend on $Z$. As before, $P$ is the data and $Q$
the model. The Fisher divergence and its blind spot are what the chapters on
diffusion models use; Stein's identity and the kernel discrepancy built on it
can be skipped on a first reading.

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

<!-- WRITER: new cell (moved helpers): defines the log-density of a Gaussian mixture and a quadrature KL, used by the score and blind-spot cells below; it needs a one-line lead. -->

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

The **score** of a distribution $P$ with differentiable density $p$ is the
gradient of its log-density *with respect to the data point*:

$$
s_P(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}).
$$
:eqlabel:`eq_mdl-score-def`

(In classical statistics, "score" means the gradient with respect to the
*parameters*, as in :numref:`sec_mdl-maximum_likelihood`. Here the gradient is
in $\mathbf{x}$.) The score is a vector field on the sample space that points
uphill toward the mass, as :numref:`fig_mdl-score-field` shows. For a Gaussian
$\mathcal{N}(\mu, \sigma^2)$ it is

$$
s(x) = -\frac{x - \mu}{\sigma^2},
$$

a spring pulling toward the mean with stiffness $1/\sigma^2$. For a mixture
$p = \sum_k w_k\, p_k$ the chain rule gives $s(x) = \sum_k r_k(x)\, s_k(x)$
with $r_k(x) = w_k p_k(x) / p(x)$: each component's spring, weighted by the
posterior probability, or *responsibility*, that $x$ came from that component.

![The score field of a two-component Gaussian mixture. Arrows show the gradient of the log-density: near each mode the field is a spring pulling toward that mode's mean, and along the valley between modes the responsibilities hand the point off from one attractor to the other. Density contours are underlaid for reference.](../img/mdl-it-score-field.svg)
:label:`fig_mdl-score-field`

The score does not depend on a normalizing constant. If
$q(\mathbf{x}) = \tilde{q}(\mathbf{x}) / Z$, then

$$
\nabla_{\mathbf{x}} \log q(\mathbf{x})
= \nabla_{\mathbf{x}} \log \tilde{q}(\mathbf{x})
- \nabla_{\mathbf{x}} \log Z
= \nabla_{\mathbf{x}} \log \tilde{q}(\mathbf{x}),
$$

since $Z$ does not depend on $\mathbf{x}$. A divergence built from scores can
therefore compare an unnormalized model with data. The natural choice is the
mean squared difference of the two vector fields under the data distribution,
the **Fisher divergence**:

$$
D_{\textrm{F}}(P\|Q) = \frac{1}{2}\, \mathbb{E}_{\mathbf{x} \sim P}\!\left[
\big\| s_P(\mathbf{x}) - s_Q(\mathbf{x}) \big\|^2 \right].
$$
:eqlabel:`eq_mdl-fisher-div-def`

It is non-negative, and it is a divergence on distributions whose densities are
positive on the same connected support. There, equal scores force
$\log p - \log q$ to be constant, and normalization makes the constant zero.
For two Gaussians with equal variance $\sigma^2$ the scores differ by the
constant $(\mu_1 - \mu_2)/\sigma^2$, so
$D_{\textrm{F}} = (\mu_1 - \mu_2)^2 / (2\sigma^4)$, against the KL value
$(\mu_1 - \mu_2)^2/(2\sigma^2)$ from :eqref:`eq_mdl-dd-gaussian-kl`. The
general case is problem :exref:`ex_mdl-divergences-distances-fisher-stein`a.

<!-- WRITER: see the note after the cell: the next two sentences no longer hold. -->
The next cell defines the mixture score by the responsibility formula. The
number it prints is the largest difference from a finite-difference derivative
of $\log p$ for the mixture of :numref:`sec_mdl-fwd-vs-rev-kl`.

```{.python .input #divergences-distances-score}
def mix_score(x, w, mus, sigmas):
    """Score of a Gaussian mixture: responsibility-weighted component scores."""
    r = np.exp(log_components(x, w, mus, sigmas) - log_mix(x, w, mus, sigmas))
    return np.sum(r * -(x - mus[:, None]) / sigmas[:, None] ** 2, axis=0)
```

<!-- WRITER: the finite-difference check and its print are dropped (review DD-29; the mixture of the forward/reverse fit is not defined on this page); the cell only defines mix_score. The sentence 'The two agree to the accuracy of finite differences on this grid.' and 'The number it prints ... sec_mdl-fwd-vs-rev-kl' above are now false and go. -->
The two agree to the accuracy of finite differences on this grid.

The Fisher divergence has a blind spot, and the mixture score explains it. Take
two mixtures with the same components and swapped weights,
$P_w = w\,\mathcal{N}(-m, 1) + (1 - w)\,\mathcal{N}(m, 1)$ with $w = 0.7$
against $w = 0.3$. Near each mode the responsibility of that mode is close to
one for both mixtures, so the two scores agree. They differ only in the valley
between the modes, where the responsibilities hand over, and there $P_{0.7}$
has little mass. How much can the Fisher divergence miss? We compute it, and KL
for comparison, as the modes move apart.

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

As the modes separate, KL settles at
$0.7 \log\tfrac{7}{3} + 0.3 \log\tfrac{3}{7} = 0.339$ nats, the divergence
between the two weight vectors, while the Fisher divergence falls by more than
four orders of magnitude, from about $0.1$ to about $4 \times 10^{-6}$. A loss
built from scores cannot see how mass is divided between distant modes. Adding
noise to the data fills the valley (problem
:exref:`ex_mdl-divergences-distances-score-blindness`c), which is one reason
why diffusion models perturb the data with noise before they match scores;
:numref:`sec_diffusion-denoising` measures the blind spot on trained networks.

One obstacle remains. :eqref:`eq_mdl-fisher-div-def` contains $s_P$, the score
of the data distribution, which we do not know. Expanding the square gives
$\tfrac12\mathbb{E}_P\|s_Q\|^2 - \mathbb{E}_P[s_Q^\top s_P]$ plus a term that
does not depend on the model $Q$, and an integration by parts removes $s_P$
from the cross term :cite:`Hyvarinen.2005`. :numref:`sec_mdl-score-matching`
carries it out in $\mathbb{R}^n$ (:eqref:`eq_mdl-hyvarinen`) and derives the
training loss of diffusion models from it. On the line, the integration by
parts is the identity of the next subsection.

## Stein's Identity
:label:`sec_mdl-stein-discrepancy`

The score can also *test* a sample against a model. The starting point is an
identity that every distribution satisfies with its own score; we state it for
the model $Q$. Let $Q$ have a differentiable density $q > 0$ on $\mathbb{R}$
with score $s_Q = (\log q)'$, and let $g$ be differentiable with
$g(x)\, q(x) \to 0$ as $x \to \pm\infty$, $\mathbb{E}_Q|g'| < \infty$ and
$\mathbb{E}_Q|g\, s_Q| < \infty$. Then *Stein's identity* :cite:`Stein.1981`
holds:

$$
\mathbb{E}_{x \sim Q}\big[ g'(x) + g(x)\, s_Q(x) \big] = 0.
$$
:eqlabel:`eq_mdl-stein-identity`

It is one integration by parts (:numref:`subsec_mdl-integration-by-parts`).
Since $q' = s_Q\, q$, we have
$\int g'\, q\, dx = \big[ g\, q \big]_{-\infty}^{\infty} - \int g\, q'\, dx = -\int g\, s_Q\, q\, dx$,
where the boundary term vanishes and both integrals converge by the conditions
on $g$.

For $Q = \mathcal{N}(0, 1)$ the score is $s_Q(x) = -x$ and the identity reads
$\mathbb{E}[g'(X)] = \mathbb{E}[X g(X)]$; with $g(x) = x$ it gives
$\mathbb{E}[X^2] = 1$. Applied to the data distribution $P$, with $g$ equal to
the model's score $s_Q$, it reads
$\mathbb{E}_P[s_Q\, s_P] = -\mathbb{E}_P[s_Q']$, which is the integration by
parts that removes the data score from the Fisher divergence.

The identity holds for a large class of $g$ at once and, for a rich enough
class, only for $Q$ itself. If the average of $g' + g\, s_Q$ over a sample is
far from zero for some $g$, the sample does not come from $Q$ (problem
:exref:`ex_mdl-divergences-distances-fisher-stein`c shows that a finite class
of polynomials is not rich enough). The cell evaluates this average for
$g(x) = x^3$ and $g(x) = \sin x$ on a million draws from $\mathcal{N}(0, 1)$
itself, where the identity holds.

<!-- WRITER: the Stein cell is redesigned (review DD-31): the model is Q = N(0, 1) with operator g' - x g; the samples are N(0, 1), N(0.5, 1) and N(0, 4) (variance 4), 200,000 draws each; the test functions are g = 1, x, sin x; it prints the mean of g' + g s_Q with its standard error in a 3x3 table. The two sentences above ("The cell evaluates this average for g(x) = x^3 ... where the identity holds") are false now and need the new description. -->

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

<!-- WRITER: measured (24 unseeded development runs, six per framework tab; the cell is NumPy only): on the model's own sample N(0, 1) every mean is within 0.005 of 0, at most about 2.5 SE (SE 0.002-0.003). On N(0.5, 1): g = 1 gives -0.49 to -0.50 (SE 0.002), g = x gives -0.24 to -0.26 (SE 0.004), sin x -0.14 to -0.15 (SE 0.002). On N(0, 4): g = 1 gives 0.00 within 3 SE (-0.011 to +0.007, SE 0.004), g = x gives -2.97 to -3.02 (SE 0.013), sin x -0.40 to -0.41 (SE 0.003). Population values: -1/2 and 1 - 1.25 = -1/4 on the shift; 0 and 1 - 4 = -3 on the wider sample. Reading (worklist section 0 item 3): each test function checks one moment, g = 1 the mean and g = x the second moment; g = 1 is blind to the change of width; g = x sees both, the shift only weakly; sin x sees both. Not "each g catches one kind". "Both means are within a few standard errors of zero, as they must be" goes. -->
Both means are within a few standard errors of zero, as they must be. A test
says something only on a sample that does not come from the model, and for that
one test function is not enough.

## The Kernel Stein Discrepancy

One test function catches one kind of violation. As with MMD, we take the
largest violation over the unit ball of a feature space, and the supremum again
has a closed form. Write the *Stein operator* of the model $Q$ as
$(\mathcal{A}_Q g)(x) = g'(x) + s_Q(x)\, g(x)$, so that Stein's identity says
$\mathbb{E}_Q[\mathcal{A}_Q g] = 0$, and measure the data distribution $P$ by
$\sup_g \mathbb{E}_P[\mathcal{A}_Q g]$.

The derivation follows that of MMD in :numref:`sec_mdl-ipm-mmd` step by step.
For $g(x) = \langle \mathbf{w}, \boldsymbol{\varphi}(x) \rangle$ with
$\|\mathbf{w}\| \leq 1$ the operator acts on the features,
$(\mathcal{A}_Q g)(x) = \langle \mathbf{w}, \boldsymbol{\psi}(x) \rangle$ with
$\boldsymbol{\psi} = \boldsymbol{\varphi}' + s_Q\, \boldsymbol{\varphi}$, and
the supremum over the ball is $\big\| \mathbb{E}_P[\boldsymbol{\psi}] \big\|$
by Cauchy--Schwarz. Its square is
$\mathbb{E}\langle \boldsymbol{\psi}(x), \boldsymbol{\psi}(x') \rangle$ for
independent draws $x, x'$ from $P$. Multiplying out leaves four inner products
of features, and each is the kernel or one of its derivatives:
$\langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x') \rangle = k$,
$\langle \boldsymbol{\varphi}'(x), \boldsymbol{\varphi}(x') \rangle = \partial_x k$,
$\langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}'(x') \rangle = \partial_{x'} k$
and
$\langle \boldsymbol{\varphi}'(x), \boldsymbol{\varphi}'(x') \rangle = \partial_x \partial_{x'} k$.
Their sum $\langle \boldsymbol{\psi}(x), \boldsymbol{\psi}(x') \rangle$ is the
**Stein kernel** $k_Q$ of $Q$. In $\mathbb{R}^n$, $g$ is a vector field, the
operator is $s_Q^\top g + \nabla \cdot g$, the kernel $k$ must be continuously
differentiable in each argument, and the Stein kernel is

$$
\begin{aligned}
k_Q(\mathbf{x}, \mathbf{x}') = {}& s_Q(\mathbf{x})^\top s_Q(\mathbf{x}')\, k(\mathbf{x}, \mathbf{x}')
+ s_Q(\mathbf{x})^\top \nabla_{\mathbf{x}'} k(\mathbf{x}, \mathbf{x}') \\
&+ \nabla_{\mathbf{x}} k(\mathbf{x}, \mathbf{x}')^\top s_Q(\mathbf{x}')
+ \nabla_{\mathbf{x}} \cdot \nabla_{\mathbf{x}'} k(\mathbf{x}, \mathbf{x}'),
\end{aligned}
$$
:eqlabel:`eq_mdl-stein-kernel`

where the last term is $\sum_i \partial_{x_i} \partial_{x'_i} k$. The squared
**kernel Stein discrepancy** (KSD) is the average Stein kernel over two
independent draws from the data:

$$
\mathrm{KSD}^2(P, Q) = \mathbb{E}_{\mathbf{x}, \mathbf{x}' \sim P}\big[ k_Q(\mathbf{x}, \mathbf{x}') \big].
$$
:eqlabel:`eq_mdl-ksd-def`

A second reading gives its population value. If $P$ also has a score and
satisfies its own Stein identity, then
$\mathbb{E}_P[g' + s_Q\, g] = \mathbb{E}_P[g' + s_P\, g] + \mathbb{E}_P[(s_Q - s_P)\, g] = \mathbb{E}_P[(s_Q - s_P)\, g]$,
and the same feature argument gives

$$
\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}\big[ (s_Q - s_P)(x)\, k(x, x')\, (s_Q - s_P)(x') \big].
$$
:eqlabel:`eq_mdl-ksd-score-diff`

So the KSD is a kernel-smoothed relative of the Fisher divergence
$D_{\textrm{F}}(P\|Q)$: the same difference of scores under the same data
distribution, multiplied across pairs of points with weight $k$ where the
Fisher divergence squares it at each point. For data $P = \mathcal{N}(0, 1)$
and the wrong model $Q' = \mathcal{N}(1, 1)$, the score difference is
$-(x - 1) - (-x) = 1$, so $\mathrm{KSD}^2 = \mathbb{E}[k(X, X')]$ with
$X - X' \sim \mathcal{N}(0, 2)$. By :eqref:`eq_mdl-rbf-gaussian-integral` this
is $(1 + 2/\ell^2)^{-1/2} = 1/\sqrt{3} \approx 0.577$ at $\ell = 1$.

What recommends the KSD is what it does not need. :eqref:`eq_mdl-stein-kernel`
involves the model only through its score, so an unnormalized model serves as
well as a normalized one. And :eqref:`eq_mdl-ksd-def` is a double expectation
under $P$ alone, so a sample $x_1, \ldots, x_N$ from the data gives the
unbiased estimator

$$
\widehat{\mathrm{KSD}}^2 = \frac{1}{N(N-1)} \sum_{i \neq j} k_Q(x_i, x_j),
$$

the same average without self-pairs as the MMD estimator, and the model is
never sampled. For suitable kernels (the RBF kernel among them, under mild
conditions on the score) $\mathrm{KSD}^2(P, Q) = 0$ if and only if $P = Q$
:cite:`Liu.Lee.Jordan.2016,Chwialkowski.Strathmann.Gretton.2016`. No training
objective in this book uses the KSD. The literature uses it as a
goodness-of-fit test for unnormalized models, as a measure of how well the
output of a sampler matches its target, and, turned from a statistic into a
velocity field, in Stein variational gradient descent :cite:`Liu.Wang.2016`.

For the one-dimensional RBF kernel $k(x, y) = e^{-(x-y)^2/(2\ell^2)}$ the
derivatives in :eqref:`eq_mdl-stein-kernel` are
$\partial_y k = \frac{x-y}{\ell^2}\, k$,
$\partial_x k = -\frac{x-y}{\ell^2}\, k$ and
$\partial_x \partial_y k = \big( \frac{1}{\ell^2} - \frac{(x-y)^2}{\ell^4} \big)\, k$.
We test one sample of $1{,}000$ points from $\mathcal{N}(0, 1)$ against the
correct model $\mathcal{N}(0, 1)$, whose score is $-x$, and against the wrong
model $\mathcal{N}(1, 1)$, whose score is $-(x - 1)$. Nothing but a score
function tells the statistic which model is wrong.

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

<!-- WRITER: the KSD sample is now 2,000 points, not 1,000 (the sentence "We test one sample of $1{,}000$ points" above needs the new size). At 1,000 points the wrong-model estimate has a run-to-run SD of 0.043 and fell to 0.457 in one of twelve development runs, outside the quoted ±0.1; at 2,000 the SD is 0.031 (200 replicates: range 0.50-0.65, mean 0.575) and the true-model estimate has SD 0.0007 (|value| < 0.004 in 200 replicates); the twelve notebook runs at 2,000 (three per tab) gave 0.544-0.630 against the wrong model and -0.0008 to +0.0013 against the true one. Both sentences below hold at 2,000. -->
Against the correct model the estimate is within about $\pm 0.005$ of zero and
may dip slightly negative; against the wrong model it is about $0.58$, within
about $\pm 0.1$ of the population value $1/\sqrt{3}$. The sample never changed
and the wrong model was never sampled: the discrepancy reads the mismatch off
the score.

## Training Objectives and Their Divergences
:label:`sec_mdl-divergence-objective-map`

:numref:`tab_mdl-divergence-objectives` pairs common training objectives with
the divergence that each one's population optimum minimizes, what the objective
needs from the model and the data, and what it reports for two point masses a
distance $d$ apart, the test case of :numref:`tab_mdl-point-mass-table`.

:Training objectives and the divergences they minimize at their population optimum. The column on point masses is the value for $\delta_0$ against $\delta_d$ with $d \neq 0$; the MMD entry uses the RBF kernel with length scale $\ell$.
:label:`tab_mdl-divergence-objectives`

| Objective | Divergence | What it needs | Two point masses $d$ apart | Treated in |
|:----|:----|:----|:----|:----|
| maximum likelihood (autoregressive models, flows) | forward KL | model density at data points | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |
| variational inference, VAE posterior | reverse KL | model samples and the unnormalized target | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |
| original GAN at the optimal discriminator | Jensen--Shannon | samples of both, through a critic | $\log 2$, constant | :numref:`sec_mdl-f-gan-dual` |
| f-GAN | the chosen $D_f$ | samples of both, through a critic | $f(0) + f'(\infty)$, constant | :numref:`sec_mdl-f-gan-dual` |
| WGAN | $W_1$ | samples of both, a 1-Lipschitz critic | $d$ | :numref:`sec_mdl-optimal-transport` |
| moment matching, two-sample tests | MMD | samples of both, no critic for a fixed kernel | $2 - 2e^{-d^2/(2\ell^2)}$, moves for $d$ below a few $\ell$ | :numref:`sec_mdl-ipm-mmd` |
| score matching, diffusion models | Fisher (blind to the weights of distant modes) | model score, data samples | undefined (a point mass has no score) | :numref:`sec_mdl-fisher-divergence` |
| goodness-of-fit tests, sampler quality | KSD | model score, data samples | undefined (a point mass has no score) | :numref:`sec_mdl-stein-discrepancy` |

:numref:`fig_mdl-divergence-taxonomy` places the ratio, test-function and
transport families and their overlaps. The score-based comparisons, which it
does not draw, lie outside all three.

![Three of the four families and where their common members live. f-divergences average a convex function of the density ratio; integral probability metrics take a supremum of mean differences over a class of test functions; optimal-transport distances measure the cost of moving mass. Total variation sits in the first two families and Wasserstein-1 in the last two. Boxed names are metrics; JS and $d_{\mathrm{H}}^2$ are squares of metrics.](../img/mdl-it-divergence-taxonomy.svg)
:label:`fig_mdl-divergence-taxonomy`

The table says which comparison an objective makes. It does not say how well
the comparison can be made from data. Every sample-based number above, the
f-GAN bound, $\textrm{MMD}^2$ and $\mathrm{KSD}^2$, is an estimate with an
error, and the f-GAN bound is biased low. :numref:`sec_mdl-mutual-information`
uses KL to measure the dependence between two variables, and
:numref:`sec_mdl-mi-estimation` asks how well a divergence can be estimated
from $N$ samples at all: no distribution-free estimate certifies more than
about $\log N$ nats.

## Summary

Score-based comparisons never see the normalizing constant. The Fisher
divergence is the objective of score matching, but it cannot see how mass is
split between distant modes, and the kernel Stein discrepancy tests a sample
against a model that is known only through its score.

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
       a general $P$ that satisfies the conditions of the identity?
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
       divergence tends to $0.4 \log\tfrac{7}{3}$, and find the rate at which
       the Fisher divergence decays by fitting $\log D_{\textrm{F}}$ against
       $m^2$.
    1. Use the responsibility form of the mixture score to explain where
       the scores of $P_{0.7}$ and $P_{0.3}$ differ, and why the weights
       barely affect them. What does this imply for the mixing weights of
       a model fitted by score matching :cite:`Wenliang.Kanagawa.2020`?
    1. Add independent $\mathcal{N}(0, \sigma^2)$ noise to samples of both
       mixtures, so that each component becomes
       $\mathcal{N}(\pm m, 1 + \sigma^2)$. Repeat (a) at $m = 5$ for
       $\sigma \in \{0, 0.5, 1, 2\}$. How does the noise change the two
       divergences? Relate the result to the noise-perturbed score models
       of :citet:`song2019generative`.
   :exlabel:`ex_mdl-divergences-distances-score-blindness`

<!-- Discussions: forum topic to be created at deploy -->

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §37.5]{.kicker}

Score-Based Comparisons<br>**comparisons that never see the normalizer**
:::
:::

::: {.slide title="The Score Ignores Z; the Fisher Divergence Compares Score Fields"}
[Fisher divergence]{.kicker}

::: {.cols .vc}
::: {.col}
- $s_P(\mathbf x)=\nabla_{\mathbf x}\log p=\nabla_{\mathbf x}\log\tilde p$, since $Z$ is constant in $\mathbf x$.
- $D_{\mathrm F}(P\|Q)=\tfrac12\mathbb{E}_P\|s_P-s_Q\|^2$.
- Equal-variance Gaussians: $\frac{(\mu_1-\mu_2)^2}{2\sigma^4}$; KL gives $\frac{(\mu_1-\mu_2)^2}{2\sigma^2}$.
:::

::: {.col .fig}
@fig:mdl-it-score-field
:::
:::
:::

::: {.slide title="The Fisher Divergence Cannot See a 70/30 Split of Distant Modes"}
[A blind spot]{.kicker}

$0.7\,\mathcal N(-m,1)+0.3\,\mathcal N(m,1)$ against the 30/70 mixture:

@!mdl-divergences-distances-the-score-and-the-fisher-divergence-1

- KL settles at $0.4\log\tfrac73=0.339$; Fisher falls to $4\times10^{-6}$.
- The scores differ only in the valley, where there is almost no mass.

[Adding noise to the data fills the valley.]{.small}
:::

::: {.slide title="Stein's Identity Needs Only the Score; KSD Measures Its Violation"}
[Kernel Stein discrepancy]{.kicker}

- $\mathbb{E}_P[g'+g\,s_P]=0$ for every smooth $g$ that decays suitably.
- $\mathrm{KSD}^2(Q,P)=\mathbb{E}_{x,x'\sim Q}[k_P(x,x')]=\mathbb{E}[(s_P-s_Q)\,k\,(s_P-s_Q)]$.
- Sample from $\mathcal N(0,1)$; wrong model $\mathcal N(1,1)$: $\mathrm{KSD}^2=1/\sqrt3\approx0.577$.

@!divergences-distances-ksd

The wrong model is never sampled.
:::

::: {.slide title="Check: Which Comparisons Survive an Unknown Normalizer?"}
[Check]{.kicker}

An energy-based model $\tilde p(x)$ with unknown $Z$, and a data sample.
Which of KL, JS, MMD, $W_1$, Fisher, KSD can you compute?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** KSD directly (model score, data samples). Fisher after an
integration by parts. MMD and $W_1$ need samples from the model; KL and JS
need $Z$.
:::
:::
:::

::: {.slide title="What You Compare Decides What You Can Detect"}
[Wrap-up]{.kicker}

- **Ratio:** KL $0.396$ / $0.365$; $\infty$ on point masses; $D_f\ge0$ by one Jensen step.
- **Test functions:** TV $0.4\le\sqrt{\textrm{KL}/2}$; MMD $0$ / $9$ / $0.094$ by feature.
- **Transport:** $W_1=1.7$ on six points; $W_1(\delta_0,\delta_d)=d$.
- **Scores:** KL $0.339$ against Fisher $4\times10^{-6}$; KSD $1/\sqrt3$ without sampling.
- ML → forward KL · VI → reverse KL · GAN → JS · WGAN → $W_1$ · moment matching → MMD · diffusion → Fisher.

::: {.d2l-note}
Next: every number here is an estimate. How well can a divergence be estimated from $N$ samples?
:::
:::
