# Stochastic Gradient Methods
:label:`sec_mdl-stochastic-gradient-methods`

Training a neural network almost never uses the exact gradient of its loss.
Instead of sweeping through the entire dataset for every update, each step
computes the gradient on a random minibatch of a few hundred examples. Every
section so far assumed exact gradients, where curvature determines the number
of steps and convexity turns any stationary point into a global minimizer. A
minibatch gradient points in the right direction on average, but it is noisy,
and that noise changes the central question of optimization from how many steps
a run needs to how close to the minimizer it can get. This section asks what
gradient noise costs and how we can overcome it.

Even the simplest possible loss reveals the cost. Consider
$f(x) = \tfrac12 x^2$, whose minimizer is $0$, and suppose we add independent
noise of unit variance to every gradient while taking constant steps of size
$\eta = 0.1$. With exact gradients, each step would shrink $x$ by a tenth,
driving the iterate as close to $0$ as we like. With noisy gradients, the
iterate never settles: its mean square converges to $0.1/1.9 \approx 0.053$, so
it wanders roughly $0.23$ away from the minimizer no matter how many steps we
take. Three quantities determine how low the loss can get: the size of the
noise, the step size, and the number of steps we can afford.

We begin by quantifying the noise of a minibatch gradient. On a quadratic
objective, we then compute where a fixed step size stalls and how many steps it
takes to get there. Next, we show how decaying step sizes eliminate that stall
and what constant a $1/k$ schedule requires, before bounding the error that a
constant step tuned to a fixed budget achieves on both convex and nonconvex
losses. Finally, we show how to cancel the noise on a finite training set using
a stored snapshot gradient. Three takeaways are worth keeping in mind
throughout: the noise floor is proportional to the step size, a $1/k$ schedule
needs a large enough constant to achieve its rate, and a fixed budget of $K$
steps calls for a step size proportional to $1/\sqrt{K}$. The final subsection
on variance reduction can be skipped on a first reading.

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Minibatch Gradients
:label:`subsec_mdl-stochastic-gradients`

### The Cost of a Full Gradient

A typical training objective averages the losses of $n$ examples,

$$
f(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x}),
$$

where $f_i$ is the loss on example $i$ and $\mathbf{x} \in \mathbb{R}^d$ holds
the model parameters. Because its gradient
$\nabla f = \tfrac1n \sum_i \nabla f_i$ is also an average, evaluating it
touches all $n$ examples. When $n = 10^9$, a single step of gradient descent
requires a full pass over the dataset, and gradient descent typically needs
thousands of steps. Even on this chapter's running example,
$\tfrac12(x_1^2 + 1000\,x_2^2)$, whose curvatures differ by a factor of
$1000$, the optimal fixed step requires about $7{,}000$ iterations to shrink
the error a million-fold (:numref:`sec_mdl-gradient-based-optimization`).
**Stochastic gradient descent** (SGD) avoids this cost by replacing the full
average with the average over a random **minibatch** $\mathcal{B}$ of $b$
indices,

$$
\hat{\mathbf{g}}_b(\mathbf{x}) = \frac{1}{b} \sum_{i \in \mathcal{B}} \nabla f_i(\mathbf{x}),
\qquad
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k\, \hat{\mathbf{g}}_b(\mathbf{x}_k).
$$
:eqlabel:`eq_mdl-opt-minibatch`

Each step now inspects only $b$ examples instead of $n$: a minibatch of
$b = 256$ out of $10^9$ examples is roughly four million times cheaper than a
full pass. Throughout this section, $k$ indexes optimizer steps, and the step
size $\eta_k$ may vary with $k$. For our analysis, we draw a fresh minibatch at
every step whose $b$ indices are independent and uniform on $\{1, \ldots, n\}$
(sampling with replacement). Real training loops usually shuffle the dataset
and sample without replacement, which multiplies the gradient variance derived
below by $(n-b)/(n-1)$—a factor close to one whenever $b \ll n$
(:numref:`sec_sgd`).

### Unbiasedness and the $1/b$ Law

How accurate is a minibatch gradient? Two quantities capture its quality: its
mean, which tells us whether it points in the right direction on average, and
its mean squared distance from the true gradient, which we call its *noise
power*. That noise power depends on how widely individual example gradients
scatter around their mean, as measured by their covariance matrix.

**Proposition (minibatch gradients are unbiased with noise power
$\operatorname{tr}\boldsymbol{\Sigma}/b$).** *Fix $\mathbf{x}$ and let*

$$
\boldsymbol{\Sigma} = \frac{1}{n} \sum_{i=1}^n \big(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\big)\big(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\big)^\top
$$

*be the covariance of a single uniformly drawn example gradient. If the $b$
indices of $\mathcal{B}$ are independent and uniform on $\{1, \ldots, n\}$, then*

$$
\mathbb{E}\big[\hat{\mathbf{g}}_b(\mathbf{x})\big] = \nabla f(\mathbf{x}),
\qquad
\mathbb{E}\big[\|\hat{\mathbf{g}}_b(\mathbf{x}) - \nabla f(\mathbf{x})\|^2\big] = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.
$$
:eqlabel:`eq_mdl-opt-variance`

**Proof.** For a single uniform index $i$,
$\mathbb{E}[\nabla f_i(\mathbf{x})] = \tfrac1n \sum_i \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x})$.
Every term in the minibatch therefore has mean $\nabla f(\mathbf{x})$, and so
does their average. Because the $b$ indices are independent and each term has
covariance $\boldsymbol{\Sigma}$, the covariance of their sum is
$b\boldsymbol{\Sigma}$, and dividing by $b$ scales the covariance by $1/b^2$ to
give $\boldsymbol{\Sigma}/b$. Finally, the expected squared norm of any
mean-zero random vector equals the trace of its covariance matrix.
$\blacksquare$

During optimization, $\mathbf{x}$ is the current iterate $\mathbf{x}_k$ and the
minibatch is drawn independently of past steps, so both identities hold
conditionally on $\mathbf{x}_k$. Note that unbiasedness depends on uniform
sampling: if a sampler draws example $i$ with probability $p_i$, each gradient
must be reweighted by $1/(np_i)$ to keep the estimator unbiased.

Because the noise power decays like $1/b$, its root-mean-square amplitude
decays only like $1/\sqrt{b}$: cutting the gradient noise in ten requires a
hundred times as many examples per step. Meanwhile, the true gradient signal
has squared norm $\|\nabla f\|^2$ regardless of batch size. Signal and noise
have equal power at the batch size
$b_{\textrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}/\|\nabla f\|^2$,
which the main text calls the *gradient-noise scale* (:numref:`sec_batch_size`);
exercise :exref:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl` derives
its curvature-weighted counterpart.

To see how large this noise is on concrete data, we set up a small
logistic-regression problem that we will reuse in the step-size experiment
below. It has $n = 200$ examples $\mathbf{a}_i \in \mathbb{R}^5$ with standard
normal features, labels $y_i \in \{-1, +1\}$ generated from a logistic model
with true weights $(1, -2, 1.5, 0, 2)$, and the regularized loss

$$
f(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n \log\big(1 + e^{-y_i \mathbf{a}_i^\top \mathbf{x}}\big) + \frac{0.1}{2} \|\mathbf{x}\|^2.
$$

The quadratic penalty makes $f$ strongly convex with $\mu \ge 0.1$. The gradient
of example $i$ is
$\nabla f_i(\mathbf{x}) = -y_i s_i \mathbf{a}_i + 0.1\,\mathbf{x}$, where
$s_i = 1/(1 + e^{y_i \mathbf{a}_i^\top \mathbf{x}})$ is the probability that
the model assigns to the wrong label. The cell below generates the dataset and
defines the loss together with a vectorized helper returning all $n$
per-example gradients as the rows of an $n \times d$ matrix.

```{.python .input #mdl-stochastic-gradient-methods-unbiasedness-and-the-1-b-law-1}
n, d, reg = 200, 5, 0.1                          # examples, parameters, l2 weight
x_true = np.array([1.0, -2.0, 1.5, 0.0, 2.0])
rng = np.random.default_rng()
A = rng.normal(size=(n, d))                       # row i: the features a_i
y = np.where(rng.random(n) < 1 / (1 + np.exp(-A @ x_true)), 1.0, -1.0)

def loss(x, A, y, reg):
    """The regularized logistic loss f(x)."""
    return np.mean(np.log1p(np.exp(-y * (A @ x)))) + 0.5 * reg * x @ x

def example_grads(x, A, y, reg):
    """The n x d matrix whose row i is the gradient of f_i at x."""
    s = 1 / (1 + np.exp(y * (A @ x)))             # probability of the wrong label
    return -(y * s)[:, None] * A + reg * x
```

Since the proposition is exact, simulation serves mainly as a sanity check and
to show the actual scale of $\operatorname{tr}\boldsymbol{\Sigma}$ on this
problem. Below, we evaluate $10{,}000$ random minibatches of each size at
$\mathbf{x} = \mathbf{0}$ and compare the empirical error against
$\operatorname{tr}\boldsymbol{\Sigma}/b$.

```{.python .input #gradient-based-optimization-sgd-variance}
G = example_grads(np.zeros(d), A, y, reg)        # per-example gradients at x = 0
g = G.mean(axis=0)                                # the full gradient
tr_sigma = ((G - g)**2).sum(axis=1).mean()
print(f'tr Sigma = {tr_sigma:.3f}')
for b in [1, 4, 16, 64]:
    idx = rng.integers(0, n, size=(10000, b))    # minibatches, with replacement
    err = ((G[idx].mean(axis=1) - g)**2).sum(axis=1).mean()
    print(f'b = {b:2d}: measured {err:.4f}, tr Sigma / b = {tr_sigma / b:.4f}, '
          f'ratio {err * b / tr_sigma:.3f}')
```

The single-example noise power $\operatorname{tr}\boldsymbol{\Sigma}$ is about
$1.1$, and across all four batch sizes the measured error matches
$\operatorname{tr}\boldsymbol{\Sigma}/b$ to within a few percent of sampling
fluctuation. Knowing the noise of a single step, we can now ask what that noise
does over an entire optimization run.

### The Noise Floor

Does gradient noise average out over many steps, or does it accumulate?
:numref:`sec_sgd` measured the cloud in which SGD settles inside a valley and
derived its radius. Here we solve the entire trajectory in closed form, which
also tells us how many steps it takes to reach that cloud. The simplest case
already contains the full mechanism: a single coordinate of a quadratic,
$f(x) = \tfrac12 \lambda x^2$ with curvature $\lambda > 0$ and minimizer $0$.
Model the stochastic gradient as the true gradient $\lambda x_k$ plus random
noise $\xi_k$ with mean zero given $x_k$ and variance $\sigma^2$. Here
$\sigma^2$ plays the role of a diagonal entry of $\boldsymbol{\Sigma}/b$, so
every quantity proportional to $\sigma^2$ scales as $1/b$. One SGD step reads

$$
x_{k+1} = x_k - \eta\,(\lambda x_k + \xi_k) = \rho\, x_k - \eta\, \xi_k,
\qquad \rho = 1 - \eta\lambda.
$$

At each step, the deterministic part pulls the iterate toward the minimizer by
the factor $\rho$, just as exact gradient descent does, while the stochastic
part delivers a random kick $-\eta\xi_k$. To track the iterate's size, we
follow its second moment $e_k = \mathbb{E}[x_k^2]$. Squaring the update and
taking expectations, the cross term $-2\rho\eta\,\mathbb{E}[x_k \xi_k]$
vanishes because $\xi_k$ has mean zero conditioned on $x_k$, leaving

$$
e_{k+1} = \rho^2 e_k + \eta^2 \sigma^2.
$$

Every step contracts the second moment by $\rho^2$ and injects fresh noise
power $\eta^2\sigma^2$. Nearly every result in this section can be read off
this one recursion. A linear recurrence with a constant driving term is solved
by finding its fixed point $V$, where $V = \rho^2 V + \eta^2\sigma^2$.
Subtracting that fixed-point equation from the recursion shows that the
remainder $e_k - V$ shrinks by $\rho^2$ at every step, giving

$$
e_k = V + \rho^{2k}\,(e_0 - V)
$$
:eqlabel:`eq_mdl-opt-noise-floor-trajectory`

with the **noise floor**

$$
V = \frac{\eta^2 \sigma^2}{1 - \rho^2}
= \frac{\eta\, \sigma^2}{\lambda\, (2 - \eta\lambda)}
\;\approx\; \frac{\eta\, \sigma^2}{2\lambda}
\qquad (\eta\lambda \ll 1).
$$
:eqlabel:`eq_mdl-opt-noise-ball`

The floor grows linearly with the step size and the noise variance, and shrinks
inversely with the curvature. Both formulas require $0 < \eta < 2/\lambda$, the
same stability ceiling as exact gradient descent on a quadratic; above it, the
second moment diverges even without noise.

The trajectory splits cleanly into two phases. The transient
$\rho^{2k}(e_0 - V)$ decays geometrically, just as in deterministic gradient
descent, whereas the floor $V$ is purely an effect of noise. Far from the
minimizer, the transient dominates and SGD behaves much like gradient descent.
Close to the minimizer, the inward pull $\eta\lambda x$ weakens while the
random kicks keep the same variance $\eta^2\sigma^2$, until the two balance at
$e = V$. From then on the iterate never converges; instead, its distribution
settles into a stationary **noise ball** around the minimizer whose
root-mean-square radius, roughly $\sqrt{\eta\sigma^2/(2\lambda)}$, scales as
$\sqrt{\eta}$. :numref:`fig_mdl-opt-sgd-noise-ball` illustrates both phases on
a two-dimensional quadratic bowl.

![Gradient descent and fixed-step SGD on $\tfrac12\|\mathbf{x}\|^2$ with $\eta=0.15$ and gradient noise of standard deviation $0.6$. The full-batch path converges to the minimizer, whereas SGD descends like gradient descent and then fluctuates inside a noise ball. Dashed circles mark the noise ball at this step size and at half the step size; their radii, drawn at three times the per-coordinate root-mean-square distance $\sqrt{\eta\sigma^2/(2\lambda)}$ to enclose the scatter, differ by a factor of $\sqrt{2}$.](../img/mdl-opt-sgd-noise-ball.svg)
:label:`fig_mdl-opt-sgd-noise-ball`

In practice we care about the expected loss rather than the squared distance.
Because $\mathbb{E} f(x_k) = \tfrac12 \lambda e_k$, the loss floor is

$$
\tfrac12 \lambda V = \frac{\eta\, \sigma^2}{2\,(2 - \eta\lambda)} \;\approx\; \frac{\eta\, \sigma^2}{4}.
$$

Surprisingly, for small steps the curvature $\lambda$ cancels out completely: a
flatter direction lets the iterate wander farther, by precisely the factor that
its flatness discounts from the loss.

On a $d$-dimensional quadratic, the coordinates along the eigenvectors of the
Hessian decouple into $d$ independent copies of this scalar recursion, each
with its own eigenvalue $\lambda_i$ and noise variance $\sigma_i^2$. The total
expected loss floor is therefore the sum
$\sum_i \eta\sigma_i^2/(2(2 - \eta\lambda_i))$. Because the $\sigma_i^2$ are
the diagonal entries of $\boldsymbol{\Sigma}/b$ in the eigenbasis, they sum to
$\operatorname{tr}\boldsymbol{\Sigma}/b$. Whenever every $\eta\lambda_i \ll 1$,
the total loss floor simplifies to approximately
$\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$, which depends on the step size
and batch size only through their ratio $\eta/b$. Halving the step size and
doubling the batch size therefore lower the floor by the same factor—the
coupling behind the large-batch learning-rate rules in
:numref:`sec_batch_size`.

How quickly does a run reach its floor? Take $\lambda = \sigma = 1$, as in the
opening example, where $\eta = 0.1$ produced $V = 0.053$. Raising the step to
$\eta = 0.3$ raises the floor to $V = 0.176$ (a little more than three times as
high, with a loss floor of $0.088$). Starting from $x_0 = 3$ ($e_0 = 9$), the
transient $e_k - V$ drops below a tenth of $V$ after $36$ steps at $\eta = 0.1$
($\rho^2 = 0.81$), compared with just $9$ steps at $\eta = 0.3$
($\rho^2 = 0.49$). Cutting the step size by a third cuts the floor by roughly a
third, but demands four times as many steps to reach it. Every method in the
rest of this section is a different way of navigating this trade-off between
speed and accuracy.

Now return to our running example $\tfrac12(x_1^2 + 1000\,x_2^2)$, modeling the
noise directly by adding independent mean-zero noise of variance $\sigma^2 = 1$
to each coordinate of the gradient $(x_1, 1000\,x_2)$ at every step. The two
coordinates obey the scalar recursion with $\lambda_1 = 1$ and
$\lambda_2 = 1000$. Set $\eta = 1/L = 10^{-3}$, half the stability ceiling
$2/1000$. The stiff coordinate has $\rho_2 = 0$: a single step erases its
transient completely, leaving it on its floor $V_2 = 10^{-3}/1000 = 10^{-6}$.
The slow coordinate has $\rho_1 = 0.999$ and floor
$V_1 = 10^{-3}/1.999 = 5.0 \times 10^{-4}$, and its transient takes
$\log 1000/(-2\log 0.999) \approx 3{,}450$ steps to shrink a thousandfold. The
total loss floor is $7.5 \times 10^{-4}$, split between $2.5 \times 10^{-4}$
from the slow coordinate and $5 \times 10^{-4}$ from the stiff one. Notice the
irony: the stiff coordinate that gradient descent eliminates in one step
accounts for two-thirds of the loss floor, because at $\eta\lambda_2 = 1$ its
loss floor is $\eta\sigma^2/2$, twice the small-step value $\eta\sigma^2/4$.
Ill-conditioning slows down the flat direction, while noise raises the loss
along both.

Our closed-form floor is exact only for a quadratic with additive noise. Next,
we test whether the scaling "floor proportional to $\eta$" still holds on the
non-quadratic logistic loss, and see how decaying step sizes break through the
floor.

## Step-Size Decay
:label:`subsec_mdl-step-size-decay`

### The Robbins--Monro Conditions

Because the noise floor scales linearly with $\eta$, actually reaching the
minimizer requires shrinking the step size toward zero. Shrink it too fast,
however, and the iterate stalls short of the goal: with unit-sized gradients,
the total distance traveled cannot exceed $\sum_k \eta_k$. The classical
conditions of :citet:`Robbins.Monro.1951` capture both requirements at once,

$$
\sum_{k} \eta_k = \infty,
\qquad
\sum_{k} \eta_k^2 < \infty.
$$
:eqlabel:`eq_mdl-opt-robbins-monro`

The first condition guarantees that the iterate can travel arbitrarily far from
its starting point, while the second keeps the cumulative variance injected by
the noise, $\sigma^2 \sum_k \eta_k^2$, finite. The schedule $\eta_k = c/k$
satisfies both conditions, whereas $\eta_k = c/\sqrt{k}$ and a constant step
satisfy only the first. Together with unbiased gradients whose variance grows
at most quadratically with the distance to the minimizer, these two conditions
guarantee that SGD converges to the minimizer of any strongly convex loss
:cite:`Robbins.Monro.1951,Bottou.Curtis.Nocedal.2018`.

On our scalar quadratic, convergence in mean square actually needs less.
Unrolling the recursion shows that $e_k \to 0$ whenever $\eta_k \to 0$ and
$\sum_k \eta_k = \infty$ (with $\eta_k < 2/\lambda$); in particular,
$\eta_k = c/\sqrt{k}$ yields $e_k \approx c\sigma^2/(2\lambda\sqrt{k})$, which
simply tracks the instantaneous floor of the current step.
The stronger square-summability condition is what the general theorem requires
to guarantee convergence along almost every individual sample path.

Two questions remain for a non-quadratic objective: does the noise floor still
scale linearly with $\eta$, and does a decaying schedule push below it? To
check on our logistic-regression problem, we first compute the reference
optimum $f^\star$ with $4{,}000$ full-gradient steps, then define a minibatch
SGD loop that records the suboptimality gap $f(\mathbf{x}_k) - f^\star$ at
every step.

```{.python .input #mdl-stochastic-gradient-methods-the-robbins-monro-conditions-1}
x_star = np.zeros(d)
for _ in range(4000):                             # full-gradient reference optimum
    x_star -= 0.5 * example_grads(x_star, A, y, reg).mean(axis=0)
f_star = loss(x_star, A, y, reg)

def sgd(schedule, A, y, reg, f_star, rng, steps=4000, b=8):
    """Minibatch SGD from x = 0; returns the gap f(x_k) - f* after every step."""
    x, gaps = np.zeros(A.shape[1]), np.empty(steps)
    for k in range(steps):
        idx = rng.integers(0, len(y), size=b)
        x -= schedule(k) * example_grads(x, A[idx], y[idx], reg).mean(axis=0)
        gaps[k] = loss(x, A, y, reg) - f_star
    return gaps
```

If the quadratic model is a reliable guide, halving a fixed step size should
halve the gap at which SGD stalls. We run $4{,}000$ steps with $b = 8$ using
constant steps $\eta = 0.8$ and $\eta = 0.4$, alongside the decaying schedule
$\eta_k = 80/(k + 100) = 0.8/(1 + k/100)$, which begins at $0.8$. The cell
prints the mean gap over the final $1{,}000$ steps of each run, reports the
ratios between them, and plots $100$-step moving averages on log--log axes.

```{.python .input #gradient-based-optimization-sgd-schedule}
rules = {'fixed 0.8': lambda k: 0.8, 'fixed 0.4': lambda k: 0.4,
         'decay 0.8/(1+k/100)': lambda k: 0.8 / (1 + k / 100)}
gaps = {name: sgd(s, A, y, reg, f_star, rng) for name, s in rules.items()}
tail = [gap[-1000:].mean() for gap in gaps.values()]
print('mean gap, last 1000 steps: ' + ', '.join(f'{t:.1e}' for t in tail))
print(f'floor(0.8) / floor(0.4) = {tail[0] / tail[1]:.2f}, '
      f'floor(0.8) / decay = {tail[0] / tail[2]:.0f}')
window = np.ones(100) / 100                       # plot 100-step moving averages
d2l.plot(np.arange(100, 4001), [np.convolve(gap, window, 'valid') for gap in gaps.values()],
         'step k', 'f(x_k) - f*', xscale='log', yscale='log', legend=list(gaps))
```

Both fixed-step runs stall. At $\eta = 0.8$, the mean gap over the last
$1{,}000$ steps settles around $1$ to $2 \times 10^{-2}$, and halving the step
size cuts it roughly in half: across repeated runs, the ratio fluctuates
between about $1.7$ and $2.8$, centered on the factor of $2$ predicted by the
quadratic model even though the logistic loss is not quadratic. Meanwhile, the
decaying schedule finishes roughly $20$ to $90$ times below the $\eta = 0.8$
floor, and its curve on the log--log plot is still trending downward despite
minibatch jitter. Its large early steps cover the distance to the minimizer
quickly, while its small late steps shrink the noise ball.

### The Constant in a $1/k$ Schedule

The schedule $\eta_k = c/(k + k_0)$ satisfies both Robbins--Monro conditions
for any $c > 0$, with the shift $k_0$ keeping the initial steps safely below
the stability ceiling. Does every choice of $c$ achieve an $O(1/k)$ rate? The
scalar quadratic with curvature $\lambda = \mu$ gives an exact answer. Writing
$t = k + k_0$, the second-moment recursion becomes

$$
e_{k+1} = \Big(1 - \frac{c\mu}{t}\Big)^2 e_k + \frac{c^2 \sigma^2}{t^2}.
$$

Here the contraction weakens like $1/t$, while the injected noise power falls
like $1/t^2$. Which effect dominates depends critically on $c$: if $c$ is too
small, the step size shrinks faster than the iterate can contract toward the
minimizer, so the schedule itself becomes the bottleneck rather than the noise.

**Proposition (the constant of a $1/k$ schedule).** *On the scalar model with
curvature $\mu$ and step sizes $\eta_k = c/(k + k_0)$, if $2c\mu > 1$ then
$k\, e_k \to A$, where*

$$
A = \frac{c^2 \sigma^2}{2c\mu - 1}.
$$
:eqlabel:`eq_mdl-opt-one-over-k-constant`

*If $2c\mu < 1$, then $e_k$ decays like $k^{-2c\mu}$, strictly slower than
$1/k$, and at the boundary $2c\mu = 1$ it decays like $\log k / k$. For every
$c > 0$, $e_k \to 0$.*

We omit the full proof here, deriving the constant $A$ by matching leading
terms and sketching the other two regimes informally; exercise
:exref:`ex_mdl-gradient-based-optimization-one-over-k-constant` provides a
complete proof by induction above the threshold and by unrolling the recursion
below it.

**Derivation of the constant.** Try the asymptotic form $e_k \approx A/t$. The
left-hand side of the recursion is $A/(t+1) \approx A/t - A/t^2$. On the
right-hand side, expanding $(1 - c\mu/t)^2 \approx 1 - 2c\mu/t$ gives
$A/t - 2c\mu A/t^2 + c^2\sigma^2/t^2$ up to terms of order $t^{-3}$. The $1/t$
terms match automatically, and equating the $1/t^2$ coefficients yields
$-A = -2c\mu A + c^2\sigma^2$, or $A(2c\mu - 1) = c^2\sigma^2$. Notice the
subtle term on the left: advancing from $t$ to $t+1$ contributes $-A/t^2$,
which is of the same order as the contraction term and creates the threshold
$2c\mu > 1$. Ignoring that shift would falsely predict $A = c\sigma^2/(2\mu)$
for every $c > 0$.

**Below the threshold (informally).** When $2c\mu < 1$, the cumulative
contraction sets the decay rate for both the initial error and the early noise.
Because $\log(1 - u) \approx -u$ and $\sum_{j \le t} 1/j \approx \log t$, the
product of contractions behaves as
$\prod_{j}(1 - c\mu/(j + k_0))^2 \approx (k_0/t)^{2c\mu}$, which decays more
slowly than $1/t$. At the boundary $2c\mu = 1$, noise injected at step $j$
survives at step $t$ with weight roughly $c^2\sigma^2/(jt)$, and summing over
$j$ produces the extra $\log k$ factor.

To see the sensitivity in numbers, take $\mu = \sigma = 1$. Setting $c = 1$
gives $A = 1$, and iterating the recursion from $e_0 = 1$ with $k_0 = 10$
yields $k\, e_k = 1.0001$ at $k = 10^6$. Doubling to $c = 2$ raises the
constant only modestly, to $A = 4/3$. Dropping below the threshold is far more
damaging: $c = 0.25$ yields a log--log slope of $-0.50$ between $k = 10^5$ and
$10^6$, and $c = 0.1$ yields a slope of just $-0.20$. Minimizing $A$ over $c$
gives the optimal constant $c = 1/\mu$, where $A = \sigma^2/\mu^2$ and
$e_k \approx \sigma^2/(\mu^2 k)$. Remarkably, averaging the iterates of a
constant-step run achieves the exact same optimal constant without decaying the
step size at all (exercise
:exref:`ex_mdl-stochastic-gradient-methods-iterate-averaging`). The sharp
threshold at $c = 1/(2\mu)$ makes $1/k$ schedules fragile in practice: the
smallest curvature $\mu$ is rarely known, and underestimating $c$ sacrifices
the convergence rate itself. This fragility is one major reason practical
learning-rate schedules decay more slowly than $1/k$.

On the running example $\tfrac12(x_1^2 + 1000\,x_2^2)$, the shift $k_0$
creates a second difficulty. Stability at the very first step requires
$c/k_0 < 2/L$, or $k_0 > cL/2$. Choosing the optimal constant $c = 1/\mu = 1$
for the slow coordinate therefore forces $k_0 > 500$. With $k_0 = 1000$ and
$\mathbf{x}_0 = (1, 1)$, the slow coordinate's transient still retains a
fraction $(k_0/(k + k_0))^2 \approx 0.008$ of its initial error after
$k = 10^4$ steps, leaving the expected loss at $4.2 \times 10^{-3}$—more than
five times worse than the constant-step floor of $7.5 \times 10^{-4}$. The
small initial steps demanded by the stiff coordinate starve the slow
coordinate, and the condition number reappears inside $k_0$.

### Schedule Shapes under the Exact Recursion

Modern training rarely uses $1/k$ decay; instead, practitioners favor cosine
decay and warmup--stable--decay, which :numref:`sec_scheduler` compares on
neural networks. Although no general theorem ranks those schedules on deep
networks, our quadratic model lets us compute the expected loss of *any*
schedule exactly, with no sampling and no random noise. Because our earlier
derivation of the second-moment recursion holds step by step even when $\eta_k$
varies, we simply have

$$
e_{k+1} = (1 - \eta_k \lambda)^2 e_k + \eta_k^2 \sigma^2,
\qquad
\mathbb{E} f(x_k) = \tfrac12 \lambda e_k.
$$

Iterating this update computes the exact expected loss curve across a budget of
$K$ steps in $K$ multiplications. We compare four shapes over the same budget
$K$, illustrated in :numref:`fig_mdl-opt-schedule-zoo`: a constant step
$\eta_0$; the $1/k$ schedule $c/(k + k_0)$ with $c = \eta_0 k_0$ so that it
also starts at $\eta_0$; **cosine decay** :cite:`Loshchilov.Hutter.2016`; and
**warmup--stable--decay** (WSD) :cite:`Hu.Tu.Han.ea.2024`, which holds $\eta_0$
constant for the first $80\%$ of the budget and then decays linearly to zero:

$$
\begin{aligned}
\eta_k^{\textrm{cos}} &= \tfrac12 \eta_0 \Big(1 + \cos\frac{\pi k}{K}\Big), \\
\eta_k^{\textrm{WSD}} &=
\begin{cases}
\eta_0, & k < 0.8K, \\
\eta_0\, \dfrac{K - k}{0.2K}, & k \ge 0.8K.
\end{cases}
\end{aligned}
$$

Both cosine and WSD start at $\eta_0$ and end at zero, but cosine decays
continuously throughout the run while WSD delays all decay until the final
fifth. We omit the brief initial warmup ramp (discussed in
:numref:`subsec_mdl-schedules-warmup`) because the quadratic model is already
stable at $\eta_0$ from step zero. WSD is popular in large-scale pretraining
for an operational reason: any checkpoint from its constant plateau can be
decayed to any chosen horizon, so the total budget need not be committed in
advance. Our quadratic model evaluates final expected loss, not that
flexibility.

![Schedule shapes over one budget $K$: constant, the $1/k$ decay (labelled $c/k$), cosine, and warmup--stable--decay (WSD), each preceded by a short linear warmup omitted in the computation below. The figure plots the step-size sequences; the expected loss each produces on the quadratic model is computed below.](../img/mdl-opt-schedule-zoo.svg)
:label:`fig_mdl-opt-schedule-zoo`

Which of these four schedules reaches the lowest final loss? It is worth
guessing before running the numbers. The cell below iterates the exact
recursion with $\lambda = \sigma = 1$, $K = 2000$, $\eta_0 = 0.3$, $x_0 = 3$,
and $k_0 = 50$ for the $1/k$ schedule, printing the expected loss at $80\%$ of
the budget and at the end before plotting $\mathbb{E} f(x_k)$.

```{.python .input #adaptive-stochastic-methods-schedules}
lam, sigma, K, eta0, x0 = 1.0, 1.0, 2000, 0.3, 3.0

def expected_loss(schedule, lam, sigma, K, x0):
    """E f(x_k) = lam e_k / 2 for k = 1..K from the exact second-moment recursion."""
    e, Ef = x0**2, np.empty(K)
    for k in range(K):
        eta = schedule(k)
        e = (1 - eta * lam)**2 * e + eta**2 * sigma**2
        Ef[k] = 0.5 * lam * e
    return Ef

schedules = {
    'constant': lambda k: eta0,
    'c/k': lambda k: eta0 / (1 + k / 50),
    'cosine': lambda k: 0.5 * eta0 * (1 + np.cos(np.pi * k / K)),
    'WSD': lambda k: eta0 if k < 0.8 * K else eta0 * (K - k) / (0.2 * K)}
traces = {name: expected_loss(s, lam, sigma, K, x0) for name, s in schedules.items()}
print(f'floor {eta0 * sigma**2 / (2 * (2 - eta0 * lam)):.4f}   '
      + ''.join(f'{name:>10s}' for name in traces))
for label, k in [('E f at 80%', int(0.8 * K) - 1), ('E f at end', K - 1)]:
    print(f'{label:15s}' + ''.join(f'{Ef[k]:10.2e}' for Ef in traces.values()))
d2l.plot(np.arange(1, K + 1), list(traces.values()), 'step k', 'E f(x_k)',
         yscale='log', legend=list(traces))
```

Because the recursion is deterministic, the numbers are identical on every run.
The constant step settles within a tenth of its loss floor $0.0882$ by step $9$
and stays there. WSD *is* the constant run until its decay phase begins, so at
$80\%$ of the budget it sits on the exact same floor; its final $400$ steps
then drop the loss by a factor of $14.4$, down to $6.1 \times 10^{-3}$,
accounting for all of its gain. Cosine decay and $1/k$ decay finish close
together at $1.7 \times 10^{-3}$ and $1.9 \times 10^{-3}$, roughly three to
four times lower than WSD.

The $1/k$ endpoint matches our proposition closely: here
$c = 0.3 \cdot 50 = 15$, giving $A = 225/29 \approx 7.8$ and
$\mathbb{E} f(x_K) \approx \tfrac12 A/(K + k_0) = 1.9 \times 10^{-3}$. Keep in
mind that this ranking is exact for these quadratic parameters and need not
transfer directly to neural networks. On the quadratic itself, holding a
constant plateau does not improve the final loss: lengthening WSD's decay phase
reduces the final loss monotonically, reaching $2.7 \times 10^{-3}$ for a
linear decay across the entire run.

## Rates for a Fixed Budget
:label:`subsec_mdl-fixed-budget-rates`

A decaying schedule is built for an indefinite run. In practice, however, we
usually know our compute budget $K$ ahead of time, which reframes the question:
what constant step size should we choose for $K$ steps, and what accuracy does
it guarantee? Without strong convexity, there is no geometric contraction
factor and no exact noise floor, so we prove upper bounds rather than exact
trajectories. Both bounds below share the same two-term structure: a transient
term that decreases as $1/(\eta K)$ plus a noise term that grows as $\eta$.
Balancing the two selects $\eta \propto 1/\sqrt{K}$, as shown in
:numref:`fig_mdl-opt-budget-tradeoff`.

![The two-term bound $D^2/(2\eta K)+\eta G^2/2$ of :eqref:`eq_mdl-opt-convex-averaged` plotted against the step size $\eta$ for $D=G=1$ and two budgets. The transient falls with $\eta$ while the floor rises with it; their sum reaches its minimum $DG/\sqrt{K}$ at $\eta^\star=D/(G\sqrt{K})$, so a $100\times$ larger budget lowers both the optimal step and the bound by $10\times$.](../img/mdl-opt-budget-tradeoff.svg)
:label:`fig_mdl-opt-budget-tradeoff`

### Averaged Iterates on Convex Losses

Our first bound requires nothing beyond convexity, applied through the
subgradient inequality of :numref:`sec_mdl-convexity`:
$f(\mathbf{y}) \ge f(\mathbf{x}) + \mathbf{s}^\top(\mathbf{y} - \mathbf{x})$
for all $\mathbf{y}$ and every subgradient
$\mathbf{s} \in \partial f(\mathbf{x})$ (which reduces to $\nabla f(\mathbf{x})$
wherever $f$ is differentiable).

**Proposition (SGD on a convex loss, averaged iterate).** *Let $f$ be convex
with a minimizer $\mathbf{x}^\star$ and optimal value
$f^\star = f(\mathbf{x}^\star)$. Run
$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{g}_k$ from an initial point
satisfying $\|\mathbf{x}_0 - \mathbf{x}^\star\| \le D$, where each stochastic
gradient satisfies
$\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] \in \partial f(\mathbf{x}_k)$ and
$\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le G^2$. Then the averaged
iterate $\bar{\mathbf{x}}_K = \frac1K \sum_{k=0}^{K-1} \mathbf{x}_k$ satisfies*

$$
\mathbb{E}\big[f(\bar{\mathbf{x}}_K)\big] - f^\star \;\le\; \frac{D^2}{2\eta K} + \frac{\eta\, G^2}{2},
$$
:eqlabel:`eq_mdl-opt-convex-averaged`

*and choosing $\eta = D/(G\sqrt{K})$ gives
$\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le DG/\sqrt{K}$.*

**Proof.** We track the squared distance to the minimizer. Expanding one update
gives

$$
\|\mathbf{x}_{k+1} - \mathbf{x}^\star\|^2
= \|\mathbf{x}_k - \mathbf{x}^\star\|^2 - 2\eta\, \mathbf{g}_k^\top(\mathbf{x}_k - \mathbf{x}^\star) + \eta^2 \|\mathbf{g}_k\|^2.
$$

Now take the conditional expectation given $\mathbf{x}_k$. Writing
$\mathbf{s}_k = \mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k]$, the cross term
becomes $-2\eta\, \mathbf{s}_k^\top(\mathbf{x}_k - \mathbf{x}^\star)$. By the
subgradient inequality at $\mathbf{y} = \mathbf{x}^\star$,
$\mathbf{s}_k^\top(\mathbf{x}_k - \mathbf{x}^\star) \ge f(\mathbf{x}_k) - f^\star$.
The entire argument hinges on this step: whenever the loss gap is large, the
gradient aligns with the displacement from $\mathbf{x}^\star$ and pulls the
iterate closer. Bounding the final term by $\eta^2 G^2$, taking total
expectations, and setting
$r_k = \mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2$, we obtain

$$
2\eta\, \mathbb{E}\big[f(\mathbf{x}_k) - f^\star\big] \;\le\; r_k - r_{k+1} + \eta^2 G^2 .
$$

Summing over $k = 0, \ldots, K-1$ telescopes the distances, and since
$r_0 - r_K \le r_0 \le D^2$, we have
$2\eta \sum_{k<K} \mathbb{E}[f(\mathbf{x}_k) - f^\star] \le D^2 + K\eta^2 G^2$.
Dividing by $2\eta K$ and applying Jensen's inequality to the convex function
$f$, $f(\bar{\mathbf{x}}_K) \le \frac1K \sum_{k<K} f(\mathbf{x}_k)$, yields
:eqref:`eq_mdl-opt-convex-averaged`. The two terms match at
$\eta = D/(G\sqrt{K})$, where each equals $DG/(2\sqrt{K})$. $\blacksquare$

The meaning of the two terms is transparent: $D^2/(2\eta K)$ is the transient
cost of traversing distance $D$ in $K$ steps of size $\eta$, while
$\eta G^2/2$ is the noise floor injected per step. Notably, the proof requires
no smoothness at all, so the bound holds even for nonsmooth losses with
corners such as the hinge loss. Multiplying the budget by $100$ reduces both
the optimal step size and the guaranteed error by a factor of $10$. This
$O(1/\sqrt{K})$ rate is unimprovable in the worst case: over the class of
problems in the proposition, no algorithm that sees only unbiased stochastic
gradients with $\mathbb{E}\|\mathbf{g}\|^2 \le G^2$ can guarantee better than
a constant multiple of $DG/\sqrt{K}$ :cite:`Nemirovski.Yudin.1983`.

Adding strong convexity changes the scaling dramatically. On our scalar
quadratic, the optimal constant step for a budget $K$ achieves
$e_K = \Theta(\log K / K)$, because strong
convexity shrinks the transient geometrically like $\rho^{2K}$ rather than
algebraically like $1/(\eta K)$.

Two caveats are worth noting. First, the proposition bounds the *averaged*
iterate $\bar{\mathbf{x}}_K$; the final iterate obeys a weaker bound, and for
the final iterate over a known budget the worst-case optimal schedule decays
linearly to zero :cite:`Defazio.Cutkosky.Mehta.ea.2023`. Second, the assumption
$\mathbb{E}\|\mathbf{g}_k\|^2 \le G^2$ excludes unbounded-gradient losses such
as a quadratic on all of $\mathbb{R}^d$, though it holds for Lipschitz losses
with bounded features (such as unregularized logistic regression) or whenever
the iterates remain inside a bounded set. Because it is a worst-case bound, it
can also be conservative on benign problems: on our logistic-regression task
restricted to a ball, it overestimates the true gap by roughly an order of
magnitude.

### Random Iterates on Nonconvex Losses
:label:`subsec_mdl-nonconvex-sgd`

Neural networks confront us with nonconvex losses *and* noisy gradients at the
same time, whereas our results so far have tackled one difficulty at a time.
For a function $f$ whose gradient is $L$-Lipschitz (an $L$-smooth function),
the loss surface lies everywhere below a tangent parabola of curvature $L$:

$$
f(\mathbf{y}) \;\le\; f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2
\quad \textrm{for all } \mathbf{x}, \mathbf{y}.
$$

With exact gradients and $\eta = 1/L$, this quadratic upper bound yields
$\min_{k<K}\|\nabla f(\mathbf{x}_k)\|^2 \le 2L\Delta/K$ without any convexity
assumption, where $f^\star$ is a lower bound on $f$ and
$\Delta = f(\mathbf{x}_0) - f^\star$
(:numref:`sec_mdl-gradient-based-optimization`). With noisy gradients on a
quadratic, a constant step stalls on a noise floor. The theorem of
:citet:`Ghadimi.Lan.2013` combines both ingredients. Model each SGD step as
$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \mathbf{g}_k$, where the stochastic
gradient satisfies

$$
\mathbb{E}\left[\mathbf{g}_k \mid \mathbf{x}_k\right] = \nabla f(\mathbf{x}_k),
\qquad
\mathbb{E}\left[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k\right] \;\le\; \sigma^2.
$$
:eqlabel:`eq_mdl-opt-sgd-oracle`

In words, the stochastic gradient is unbiased and its noise power is uniformly
bounded by $\sigma^2$. By :eqref:`eq_mdl-opt-variance`, a minibatch gradient
satisfies both assumptions with
$\sigma^2 = \sup_{\mathbf{x}} \operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x})/b$,
provided that supremum is finite. It is finite for the bounded $\tanh$ model
below, though it is infinite for least squares on all of $\mathbb{R}^d$, where
the gradient noise grows with the distance from the minimizer.

**Proposition (SGD finds approximate stationary points).** *Let $f$ be
$L$-smooth and bounded below by $f^\star$, set
$\Delta = f(\mathbf{x}_0) - f^\star$, and run SGD with stochastic gradients
satisfying :eqref:`eq_mdl-opt-sgd-oracle` and a constant step size
$\eta \le 1/L$. Then*

$$
\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right]
\;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2,
$$
:eqlabel:`eq_mdl-opt-ghadimi-lan`

*and choosing
$\eta = \min\bigl(1/L,\ \sqrt{2\Delta/(L\sigma^2 K)}\bigr)$ ensures that for
an index $R$ drawn uniformly from $\{0, \ldots, K-1\}$ independently of the
run,*

$$
\mathbb{E}\left[\|\nabla f(\mathbf{x}_R)\|^2\right]
\;\le\; \frac{2L\Delta}{K} + 2\sigma\,\sqrt{\frac{2L\Delta}{K}}.
$$
:eqlabel:`eq_mdl-opt-ghadimi-lan-balanced`

**Proof.** Apply the $L$-smoothness upper bound to the actual step
$\mathbf{y} - \mathbf{x} = -\eta\, \mathbf{g}_k$:

$$
f(\mathbf{x}_{k+1}) \;\le\; f(\mathbf{x}_k) - \eta\, \nabla f(\mathbf{x}_k)^\top \mathbf{g}_k + \tfrac{L\eta^2}{2}\, \|\mathbf{g}_k\|^2.
$$

Taking the conditional expectation given $\mathbf{x}_k$ accomplishes two things
at once: unbiasedness turns the middle term into
$-\eta\,\|\nabla f(\mathbf{x}_k)\|^2$, and the variance decomposition splits
the quadratic term into signal plus noise,
$\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] = \|\nabla f(\mathbf{x}_k)\|^2 + \mathbb{E}[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k] \le \|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2$.
Collecting terms gives

$$
\mathbb{E}\left[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\right]
\;\le\; f(\mathbf{x}_k) - \eta\left(1 - \tfrac{L\eta}{2}\right) \|\nabla f(\mathbf{x}_k)\|^2 + \tfrac{L\eta^2 \sigma^2}{2}.
$$

This is the standard descent lemma with one extra variance penalty
$\tfrac12 L\eta^2\sigma^2$ per step. Whenever $\eta \le 1/L$, we have
$1 - \tfrac{L\eta}{2} \ge \tfrac12$. Taking total expectations and summing over
$k = 0, \ldots, K-1$, the losses telescope to
$\mathbb{E}[f(\mathbf{x}_K)] - f(\mathbf{x}_0) \ge -\Delta$; dividing both
sides by $\eta K/2$ yields :eqref:`eq_mdl-opt-ghadimi-lan`. The left-hand side
is precisely $\mathbb{E}[\|\nabla f(\mathbf{x}_R)\|^2]$ for a uniformly random
index $R$. Finally, for the balanced step size there are two cases. If
$\sqrt{2\Delta/(L\sigma^2 K)} \le 1/L$, the two terms in
:eqref:`eq_mdl-opt-ghadimi-lan` are equal, each giving
$\sigma\sqrt{2L\Delta/K}$. Otherwise $\eta = 1/L$ and
$\sigma^2 < 2L\Delta/K$, so the bound is
$2L\Delta/K + \sigma^2 \le 2L\Delta/K + \sigma\sqrt{2L\Delta/K}$. Both cases
are bounded by :eqref:`eq_mdl-opt-ghadimi-lan-balanced`. $\blacksquare$

When $\sigma = 0$, the second term disappears and we recover the deterministic
rate $2L\Delta/K$. When noise is present and the budget is large enough that
$K > 2L\Delta/\sigma^2$ (so $\eta < 1/L$), the noise term dominates and decays
only like $K^{-1/2}$. With $L = \Delta = \sigma = 1$ and $K = 10^4$, for
instance, the balanced step is $\eta \approx 0.014$ and the bound reads
$2 \times 10^{-4} + 0.028$—the noise term is $140$ times larger than the
deterministic term. Notice also that this bound controls the *squared* gradient
norm: multiplying the budget by $100$ reduces $\mathbb{E}\|\nabla f\|^2$ by a
factor of $10$, meaning that shrinking the gradient norm itself by a factor of
$10$ requires $10^4$ times as many steps. If the gradient estimator is biased,
an extra inner product survives the proof and adds a constant floor that never
vanishes with $K$ (exercise
:exref:`ex_mdl-stochastic-gradient-methods-biased-estimators`).

Why does the theorem return a *random* iterate $\mathbf{x}_R$ rather than the
best one? With exact gradients, we observe $\|\nabla f(\mathbf{x}_k)\|$ at
every step and can simply pick the iterate with the smallest gradient. With
stochastic gradients, the true gradient norm is hidden from the algorithm;
picking an index $R$ uniformly at random is an implementable rule, and the
time-average in :eqref:`eq_mdl-opt-ghadimi-lan` is exactly its expected squared
gradient norm.

Compared with the convex case, two things changed: we bound the squared
gradient norm instead of the loss gap, and we evaluate at a random iterate
instead of the average. Without convexity, a small gradient is all a first-order
method can certify, since a stationary point need not be a global minimizer.
Like the convex bound, this $K^{-1/2}$ rate is worst-case optimal for any
method given access only to stochastic gradients satisfying
:eqref:`eq_mdl-opt-sgd-oracle` :cite:`Arjevani.Carmon.Duchi.ea.2023`.

### The Rate on a Single $\tanh$ Neuron

Because the proposition is an upper bound, an experiment cannot prove that the
rate is tight; it can only check how real runs behave relative to the $K^{-1/2}$
ceiling. Below, we measure both the random-iterate quantity and the best
iterate on a small nonconvex problem—fitting a single $\tanh$ neuron by least
squares to $n = 64$ noisy examples in five dimensions:

$$
f(\mathbf{x}) = \frac1n \sum_{i=1}^n \big(\tanh(\mathbf{a}_i^\top \mathbf{x}) - y_i\big)^2 .
$$

Each component is smooth and bounded, but none is convex because $\tanh$
saturates into flat plateaus for large arguments. The cell below creates the
data and defines a vectorized helper computing the exact full gradient for a
batch of parameter vectors, one per row of `X`.

```{.python .input #mdl-stochastic-gradient-methods-the-measured-slope-1}
n_nc, d_nc = 64, 5                                # a small nonconvex problem
A_nc = rng.normal(size=(n_nc, d_nc))
y_nc = np.tanh(A_nc @ rng.normal(size=d_nc)) + 0.5 * rng.normal(size=n_nc)

def full_grad(X, A, y):
    """Exact gradient of mean_i (tanh(a_i^T x) - y_i)^2 at each row x of X."""
    t = np.tanh(X @ A.T)                          # (runs, n)
    return 2 * ((t - y) * (1 - t**2)) @ A / len(y)
```

We run single-example SGD for budgets $K \in \{125, 500, 2000, 8000\}$ using
the theorem's scaling $\eta = 0.5/\sqrt{K}$, executing $20$ independent runs in
parallel as the rows of `X`. For each budget, the cell records both the average
of $\|\nabla f(\mathbf{x}_k)\|^2$ across the run (the random-iterate metric)
and its minimum (the best iterate), prints their log--log slopes against $K$,
and plots the random-iterate curve beside a reference slope of $-1/2$.

```{.python .input #adaptive-stochastic-methods-ghadimi-lan}
runs, Ks, avg, best = 20, np.array([125, 500, 2000, 8000]), [], []
for K in Ks:
    eta = 0.5 / np.sqrt(K)                        # the theorem's eta ~ 1/sqrt(K)
    X = np.zeros((runs, d_nc))                    # one row per independent run
    total, low = np.zeros(runs), np.full(runs, np.inf)
    for k in range(K):
        g2 = (full_grad(X, A_nc, y_nc)**2).sum(axis=1)   # |grad f(x_k)|^2
        total, low = total + g2, np.minimum(low, g2)
        i = rng.integers(0, n_nc, size=runs)             # one example per run
        t = np.tanh((X * A_nc[i]).sum(axis=1))
        X -= eta * (2 * (t - y_nc[i]) * (1 - t**2))[:, None] * A_nc[i]
    avg.append(total.mean() / K), best.append(low.mean())

def slope(v):
    return np.polyfit(np.log(Ks), np.log(v), 1)[0]

for name, v in [('random iterate', avg), ('best iterate', best)]:
    print(f'{name:15s}' + ' '.join(f'{u:.1e}' for u in v) + f'   slope {slope(v):.2f}')
d2l.plot(Ks, [avg, avg[0] * (Ks / Ks[0])**-0.5], 'budget K', 'E |grad f(x_R)|^2',
         xscale='log', yscale='log', legend=['random iterate', 'slope -1/2'])
```

Over a $64$-fold increase in budget, the random-iterate metric drops by a
factor of about $12$ (typically between $9$ and $16$ across runs), corresponding
to a log--log slope near $-0.6$. That is slightly steeper than the theorem's
$-1/2$ upper bound, as expected. The best iterate falls much faster—with a
slope around $-1$ or steeper—but as noted above, an algorithm that sees only
noisy gradients has no way to identify it.

## Variance Reduction
:label:`sec_mdl-variance-reduction`

### Control Variates

Decaying the step size eliminates the noise floor by shrinking $\eta$, but only
by slowing progress to an algebraic rate. When the objective is a *finite* sum
over $n$ examples, there is a second option: eliminate the gradient noise
itself. At the minimizer $\mathbf{x}^\star$, the full gradient vanishes, yet
the individual example gradients $\nabla f_i(\mathbf{x}^\star)$ generally do
not—and those nonzero component gradients are the noise that a constant step
size keeps injecting.

One important regime needs no remedy at all. If a model interpolates the
training set—fitting every single example with zero loss—then every
$\nabla f_i(\mathbf{x}^\star)$ vanishes at the minimizer, and constant-step
SGD on a strongly convex objective converges linearly with no noise floor
(exercise :exref:`ex_mdl-stochastic-gradient-methods-interpolation`). Large
neural networks often interpolate their training sets, which helps explain why
constant-step SGD performs much better on them than the additive-noise floor
predicts.

When the component gradients do not vanish at $\mathbf{x}^\star$, we must
cancel their variance directly. A standard Monte Carlo trick does this: from
the noisy estimator $\nabla f_i(\mathbf{x})$, subtract a correlated random
variable whose exact mean we know, and add that mean back. Such a correction
is called a **control variate**; we meet it again in the baseline of policy
gradients (:numref:`sec_baselines`). On a finite sum, a natural control
variate is the gradient $\nabla f_i(\tilde{\mathbf{x}})$ of the same example
evaluated at a saved snapshot $\tilde{\mathbf{x}}$, whose average over all $i$
is the full gradient $\nabla f(\tilde{\mathbf{x}})$, computed once per pass
over the dataset. This yields the **SVRG** (stochastic variance-reduced
gradient) estimator :cite:`Johnson.Zhang.2013`,

$$
\tilde{\mathbf{g}}_i(\mathbf{x})
= \nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}}) + \nabla f(\tilde{\mathbf{x}}).
$$
:eqlabel:`eq_mdl-opt-svrg`

In words, SVRG starts from the exact full gradient at the snapshot and adjusts
it by how much a single example's gradient has changed between
$\tilde{\mathbf{x}}$ and $\mathbf{x}$. For a uniformly drawn index $i$, the
estimator remains unbiased,

$$
\mathbb{E}_i\big[\tilde{\mathbf{g}}_i(\mathbf{x})\big]
= \nabla f(\mathbf{x}) - \nabla f(\tilde{\mathbf{x}}) + \nabla f(\tilde{\mathbf{x}})
= \nabla f(\mathbf{x}),
$$

while its variance shrinks to zero as the iterate and snapshot converge.

**Proposition (SVRG variance is bounded by the distance to the snapshot).**
*If each component $f_i$ has an $L_i$-Lipschitz gradient and
$L_{\max} = \max_i L_i$, then for a uniformly drawn index $i$,*

$$
\mathbb{E}_i\big\|\tilde{\mathbf{g}}_i(\mathbf{x}) - \nabla f(\mathbf{x})\big\|^2
\;\le\; \mathbb{E}_i\big\|\nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}})\big\|^2
\;\le\; L_{\max}^2\, \|\mathbf{x} - \tilde{\mathbf{x}}\|^2 .
$$
:eqlabel:`eq_mdl-opt-svrg-variance`

**Proof.** The left-hand side is the variance of the random vector
$\nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}})$, whose mean is
$\nabla f(\mathbf{x}) - \nabla f(\tilde{\mathbf{x}})$. Because the variance of
any random vector is at most its second moment, the first inequality follows
immediately. Applying the $L_i$-Lipschitz bound to each term of the second
moment gives the second inequality. $\blacksquare$

Because the noise power is bounded by a multiple of
$\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$, it vanishes automatically as both
$\mathbf{x}$ and $\tilde{\mathbf{x}}$ approach $\mathbf{x}^\star$, removing
the noise floor even for a constant step size. Intuitively, evaluating the same
example at two nearby points produces gradients that share nearly identical
noise; taking their difference cancels that shared noise while retaining the
small genuine change in the gradient.

### SVRG and SAGA

To keep the bound :eqref:`eq_mdl-opt-svrg-variance` shrinking, the snapshot
must periodically catch up to the iterate; yet recomputing a full gradient at
every step would defeat the purpose of stochastic optimization. SVRG resolves
this by running in epochs. At the beginning of each epoch, it saves the current
iterate as the snapshot $\tilde{\mathbf{x}}$ and computes its full gradient
$\nabla f(\tilde{\mathbf{x}})$. It then performs $m$ inner updates
$\mathbf{x} \leftarrow \mathbf{x} - \eta\, \tilde{\mathbf{g}}_i(\mathbf{x})$
with fresh random indices, and sets the final inner iterate as the next
snapshot. One epoch therefore costs $n + 2m$ example-gradient evaluations.
:citet:`Johnson.Zhang.2013` (Theorem 1) prove that if every $f_i$ is convex
with an $L_{\max}$-Lipschitz gradient, $f$ is $\mu$-strongly convex,
$\eta < 1/(4L_{\max})$, and $m$ is large enough that the per-epoch contraction
factor falls below one, then the expected suboptimality
$\mathbb{E}[f(\tilde{\mathbf{x}})] - f^\star$ decays geometrically with the
number of epochs. (The theorem selects a random inner iterate as the next
snapshot; our implementation uses the last iterate, which behaves similarly in
practice.)

We test SVRG on ridge regression, a strongly convex finite sum with
$f_i(\mathbf{x}) = \tfrac12(\mathbf{a}_i^\top\mathbf{x} - y_i)^2 + \tfrac{0.1}{2}\|\mathbf{x}\|^2$,
$n = 300$ examples, and $d = 12$ parameters. Its exact minimizer solves the
normal equations, and each component gradient is $L_i$-Lipschitz with
$L_i = \|\mathbf{a}_i\|^2 + 0.1$. The cell below generates the data, defines
the component and full gradients, and prints $L_{\max}$ along with the
theoretical step-size ceiling $1/(4L_{\max})$.

```{.python .input #mdl-stochastic-gradient-methods-svrg-and-saga-1}
n_r, d_r, reg_r = 300, 12, 0.1                    # ridge regression, a finite sum
A_r = rng.standard_normal((n_r, d_r))
y_r = A_r @ rng.standard_normal(d_r) + 0.5 * rng.standard_normal(n_r)
H_r = A_r.T @ A_r / n_r + reg_r * np.eye(d_r)     # Hessian of f
x_star_r = np.linalg.solve(H_r, A_r.T @ y_r / n_r)

def component_grad(x, i, A, y, reg):
    """Gradient of f_i(x) = (a_i^T x - y_i)^2 / 2 + reg |x|^2 / 2 (i: index or array)."""
    return A[i] * (A[i] @ x - y[i])[..., None] + reg * x

def full_grad_sum(x, A, y, reg):
    """Gradient of the finite sum f = mean_i f_i."""
    return A.T @ (A @ x - y) / len(y) + reg * x

def objective(x, A, y, reg):
    """The ridge objective f(x)."""
    return 0.5 * np.mean((A @ x - y)**2) + 0.5 * reg * x @ x

L_max = (A_r**2).sum(axis=1).max() + reg_r       # largest per-example smoothness
print(f'L_max = {L_max:.0f}, 1 / (4 L_max) = {1 / (4 * L_max):.4f}')
```

Across random draws of the data, $L_{\max}$ is typically between $25$ and $40$,
putting the theorem's step-size ceiling $1/(4L_{\max})$ between roughly $0.006$
and $0.01$.

Next, we compare SVRG ($m = n$) against single-example SGD at the same constant
step size $\eta = 0.02$—more than twice the theorem's bound—for $12$ epochs
costing $3n$ example-gradient evaluations each. We also test our earlier
quadratic-floor formula on SGD: its steady-state loss gap should equal the sum
of $\eta\sigma_i^2/(2(2 - \eta\lambda_i))$ over the eigenvectors of the
Hessian, where $\sigma_i^2$ is the variance of the example gradients at
$\mathbf{x}^\star$ along eigenvector $i$.

```{.python .input #mdl-adaptive-svrg}
eta, epochs = 0.02, 12
f_star_r = objective(x_star_r, A_r, y_r, reg_r)
snap, gaps_svrg = np.zeros(d_r), []
for _ in range(epochs):                           # n + 2n example gradients per epoch
    g_snap = full_grad_sum(snap, A_r, y_r, reg_r)
    x = snap.copy()
    for i in rng.integers(0, n_r, size=n_r):
        x -= eta * (component_grad(x, i, A_r, y_r, reg_r)
                    - component_grad(snap, i, A_r, y_r, reg_r) + g_snap)
    snap = x
    gaps_svrg.append(objective(snap, A_r, y_r, reg_r) - f_star_r)
x, gaps_sgd = np.zeros(d_r), []
for _ in range(epochs):
    for i in rng.integers(0, n_r, size=3 * n_r):  # the same 3n budget per epoch
        x -= eta * component_grad(x, i, A_r, y_r, reg_r)
    gaps_sgd.append(objective(x, A_r, y_r, reg_r) - f_star_r)
G_r = component_grad(x_star_r, np.arange(n_r), A_r, y_r, reg_r)  # noise at x*
ev, Q = np.linalg.eigh(H_r)                       # Hessian eigenvalues, eigenvectors
s2 = ((G_r @ Q)**2).mean(axis=0)                  # noise variance per eigenvector
floor = (eta * s2 / (2 * (2 - eta * ev))).sum()   # the loss floors add
print(f'SGD: predicted floor {floor:.1e}, mean gap in epochs 4-12 '
      f'{np.mean(gaps_sgd[3:]):.1e}')
print('SVRG: gap after epochs 1, 4, 12: '
      + ', '.join(f'{gaps_svrg[e - 1]:.0e}' for e in (1, 4, 12)))
d2l.plot(np.arange(1, epochs + 1), [gaps_sgd, gaps_svrg], 'epoch (3n gradients)',
         'f(x) - f*', yscale='log', legend=['SGD', 'SVRG'])
```

SGD drops rapidly at first and then plateaus at a few times $10^{-2}$, right
where our floor formula predicts. Across runs, the measured plateau falls
between roughly $10\%$ below and $40\%$ above the prediction—slightly above
more often than below, because the formula evaluates gradient variance only at
$\mathbf{x}^\star$, whereas the example gradients spread slightly more away
from the minimizer. SVRG starts behind and finishes far ahead. After its first
epoch—which takes only $n$ inner steps anchored at the starting point—its gap
still exceeds SGD's. From epoch $2$ onward, however, it shrinks by roughly a
factor of eight per epoch (between five and thirteen across runs), reaching
$10^{-13}$ to $10^{-8}$ by epoch $12$. Notably, it achieves this linear
convergence at a step size more than twice the theorem's limit, showing that
the theoretical step condition is sufficient rather than necessary. Keep in
mind the three assumptions behind this win: a finite sum, multiple passes over
it, and strong convexity for the linear rate. On streaming data, where each
example is seen only once, there is no finite sum to snapshot.

**SAGA** :cite:`Defazio.Bach.Lacoste-Julien.2014` avoids periodic full passes
altogether by replacing the snapshot with a table
$\mathbf{t}_1, \ldots, \mathbf{t}_n$ storing the most recent gradient evaluated
for each example. At each step it forms the estimator
$\nabla f_i(\mathbf{x}) - \mathbf{t}_i + \tfrac1n\sum_j \mathbf{t}_j$ and then
overwrites $\mathbf{t}_i$ with $\nabla f_i(\mathbf{x})$. SAGA computes only one
new example gradient per step, at the price of storing $n \times d$ numbers (or
just one scalar per example for generalized linear models; see exercise
:exref:`ex_mdl-adaptive-stochastic-methods-svrg-inner-loop`).

On deep neural networks, unfortunately, neither SVRG nor SAGA has delivered
consistent gains: the control variate quickly goes stale when the loss
landscape shifts rapidly or when data augmentation perturbs each example on
every pass. Meanwhile, a second limitation has been with us since the
ill-conditioned valley: every method in this section applies a single global
step size to all coordinates, forcing the stiffest direction of our running
example to cap $\eta$ at $2/1000$. In the next section, we remove that
bottleneck by letting each coordinate adapt its own step size from the
gradients it has observed.

## Summary

A minibatch gradient of size $b$ is an unbiased estimator of $\nabla f$ whose
noise power is $\operatorname{tr}\boldsymbol{\Sigma}/b$. On a quadratic, a
fixed step size $\eta$ drives the second moment of each coordinate
geometrically at rate $(1 - \eta\lambda)^2$ onto the noise floor
$\eta\sigma^2/(\lambda(2 - \eta\lambda))$, and the per-coordinate loss floors
$\eta\sigma^2/(2(2 - \eta\lambda))$ add up across dimensions. Because the floor
grows linearly with $\eta$ while the transient shrinks faster at larger $\eta$,
no constant step reaches the minimizer, and every technique in this section
manages that trade-off. A Robbins--Monro schedule removes the floor, with
$\eta_k = c/k$ achieving $O(1/k)$ only when $c > 1/(2\mu)$. Over a fixed budget
of $K$ steps, a constant step $\eta \propto 1/\sqrt{K}$ guarantees
$DG/\sqrt{K}$ for the averaged iterate on convex losses and
$O(\sigma\sqrt{L\Delta/K})$ for the expected squared gradient at a random
iterate on smooth nonconvex losses. Finally, on a finite sum, the SVRG control
variate stays unbiased with variance at most
$L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$, allowing a fixed step size
to converge linearly on strongly convex problems.

## Exercises

1. [code] **Noise scale and critical batch size.** ● How much does one
   minibatch step lower the loss, and how does the gain depend on $b$? Write
   $\mathbf{g} = \nabla f(\mathbf{x})$ and $\mathbf{H} = \nabla^2 f(\mathbf{x})$,
   and approximate the loss after the step
   $\mathbf{x} - \eta\,\hat{\mathbf{g}}_b$ of :eqref:`eq_mdl-opt-minibatch`
   by its second-order Taylor expansion at $\mathbf{x}$. The analysis follows
   :citet:`McCandlish.Kaplan.Amodei.ea.2018`.
    1. Use the covariance $\boldsymbol{\Sigma}/b$ of $\hat{\mathbf{g}}_b$,
       established in the proof of :eqref:`eq_mdl-opt-variance`, to show that
       the expected loss after the step is minimized by
       $\eta_{\textrm{opt}}(b) = \eta_{\max} / (1 + B_{\textrm{noise}}/b)$,
       with expected improvement $\Delta_{\max} / (1 + B_{\textrm{noise}}/b)$,
       where

       $$
       \eta_{\max} = \frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top \mathbf{H} \mathbf{g}},
       \qquad
       \Delta_{\max} = \frac{\|\mathbf{g}\|^4}{2\,\mathbf{g}^\top \mathbf{H} \mathbf{g}},
       \qquad
       B_{\textrm{noise}} = \frac{\operatorname{tr}(\mathbf{H} \boldsymbol{\Sigma})}{\mathbf{g}^\top \mathbf{H} \mathbf{g}}.
       $$

    1. The linear scaling rule of :citet:`Goyal.Dollar.Girshick.ea.2017`
       multiplies $\eta$ by the same factor as $b$. For which
       batch sizes does this rule follow from $\eta_{\textrm{opt}}(b)$? What
       does a larger batch gain per step once $b \gg B_{\textrm{noise}}$?
    1. For $\mathbf{H}$ proportional to the identity, $B_{\textrm{noise}}$
       reduces to $B_{\textrm{simple}} = \operatorname{tr}\boldsymbol{\Sigma} / \|\mathbf{g}\|^2$,
       the gradient-noise scale $b_{\textrm{noise}}$ of the text. Compute
       $B_{\textrm{simple}}$ and $B_{\textrm{noise}}$ for the logistic-regression
       problem of this section at $\mathbf{x} = \mathbf{0}$ and after $5$, $10$,
       and $20$ full-gradient steps with $\eta = 0.5$, using `example_grads`
       and the Hessian
       $\mathbf{H}(\mathbf{x}) = \frac1n \sum_i s_i(1 - s_i)\,\mathbf{a}_i \mathbf{a}_i^\top + 0.1\,\mathbf{I}$
       with $s_i = 1/(1 + e^{y_i \mathbf{a}_i^\top \mathbf{x}})$. How does the
       noise scale change as the iterates approach the optimum? Treat
       $B_{\textrm{noise}}$ as the critical batch size that separates the two
       regimes of (b). How should the batch size change over a training run?
   :exlabel:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl`

1. [code] **The constant in a $1/k$ schedule.** ●● The text found the
   constant $A$ of :eqref:`eq_mdl-opt-one-over-k-constant` to leading order.
   This exercise makes it rigorous and carries it to the running example.
    1. Let $t = k + k_0$ and suppose
       $u_{k+1} \le (1 - a/t)\, u_k + q/t^2$ for all $k \ge 0$, with $a > 1$,
       $q > 0$, $u_k \ge 0$ and $k_0 \ge a$. Prove by induction that
       $u_k \le \nu/(k + k_0)$ with $\nu = \max\{q/(a - 1),\ k_0 u_0\}$.
       (Hint: $1 - a/t \ge 0$, and $1/t - 1/t^2 \le 1/(t + 1)$.)
    1. Apply the lemma to the recursion
       $e_{k+1} = (1 - c\mu/t)^2 e_k + c^2\sigma^2/t^2$. Show that
       $(1 - c\mu/t)^2 \le 1 - (2c\mu - (c\mu)^2/k_0)/t$ for $t \ge k_0$, so
       that any $2c\mu > 1$ gives $e_k = O(1/k)$ once $k_0$ is large enough.
       Identify $a$ and $q$, and show that $q/(a - 1) \to A$ as $k_0 \to \infty$.
    1. For $2c\mu < 1$, show $e_k = \Theta(k^{-2c\mu})$ from the product
       $\prod_{j<k}(1 - c\mu/(j + k_0))^2$ and the unrolled noise sum.
    1. Iterate the recursion with $\mu = \sigma = 1$, $k_0 = 10$ and $e_0 = 1$
       up to $k = 10^6$ for $c \in \{0.1, 0.25, 0.5, 1, 2\}$. Report the
       log–log slope between $k = 10^5$ and $10^6$ and the value of $k\, e_k$
       at $10^6$. Which costs more, a constant twice the minimizer of $A$ or
       half of it?
    1. On the running example $\tfrac12(x_1^2 + 1000\,x_2^2)$ with unit
       noise per coordinate, stability at the first step forces
       $c/k_0 < 2/1000$. Take $c = 1/\mu = 1$, $k_0 = 1000$,
       $\mathbf{x}_0 = (1, 1)$ and $K = 10^4$. Compute the expected loss by the
       two-coordinate recursion (it is about $4.2 \times 10^{-3}$) and compare
       it with the constant step $\eta = 10^{-3}$ ($7.5 \times 10^{-4}$) and with
       cosine decay from the same $\eta_0$ ($1.0 \times 10^{-4}$). Which
       coordinate does the $1/k$ decay starve, and why does the condition
       number reappear in $k_0$?
   :exlabel:`ex_mdl-gradient-based-optimization-one-over-k-constant`

1. [code] **SVRG inner loop and SAGA.** Use the ridge data and the functions
   `component_grad`, `full_grad_sum` and `objective` of this section with a
   budget of $36n = 10{,}800$ example-gradient evaluations, the budget behind
   the $12$ reported epochs of the SVRG cell. Measure each run by the number
   of evaluations it needs to bring the gap below $10^{-10}$, checking once per
   epoch.
    1. Let the inner loop run $m$ steps, so that one epoch costs $n + 2m$
       evaluations, and vary $m \in \{10, 30, 75, 300, 1200\}$ at
       $\eta \in \{0.01, 0.02\}$. Which $m$ is best at each step size?
       Explain the trade-off between amortizing the snapshot gradient and
       letting the control variate go stale, and why the best $m$ moves
       with $\eta$.
    1. Implement SAGA: keep a table with the most recent gradient of every
       example, filled at $\mathbf{x}_0$ for $n$ evaluations, and use
       $\nabla f_i(\mathbf{x}) - \mathbf{t}_i + \tfrac1n \sum_j \mathbf{t}_j$
       as the estimator, where $\mathbf{t}_i$ is the table entry of example
       $i$. Compare SAGA with the best SVRG run at both step sizes. How many
       numbers does the table hold for this problem, and how many would
       suffice if you stored only the scalar $\mathbf{a}_i^\top\mathbf{x} - y_i$
       per example?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-svrg-inner-loop`

1. **Biased gradient estimators.** ● Keep the variance bound of
   :eqref:`eq_mdl-opt-sgd-oracle`, now measured around the conditional mean,
   but replace unbiasedness by
   $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] = \nabla f(\mathbf{x}_k) + \boldsymbol{\delta}_k$
   with $\|\boldsymbol{\delta}_k\| \le \delta$.
    1. Redo the proof of :eqref:`eq_mdl-opt-ghadimi-lan` for
       $\eta \le 1/L$ and show

       $$\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right] \;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2 + \delta^2.$$

       Use $L\eta \le 1$ on the term
       $\tfrac{L\eta^2}{2}\|\nabla f(\mathbf{x}_k) + \boldsymbol{\delta}_k\|^2$
       before expanding it.
    1. Show that the term $\delta^2$ cannot be removed: give a
       one-dimensional $f$ and a constant bias for which SGD with $\sigma = 0$
       converges to a point where $|f'|^2 = \delta^2$.
    1. A single minibatch drawn uniformly without replacement is still an
       unbiased estimator of the full gradient. Which of the following
       estimators are biased in this sense: the example gradient
       at a step of random reshuffling (:numref:`sec_sgd`), which visits
       every example once per epoch in a fresh random order; a stochastic
       gradient clipped to a fixed maximum norm; the gradient evaluated at the
       iterate of $\tau$ steps earlier, as in asynchronous training?
   :exlabel:`ex_mdl-stochastic-gradient-methods-biased-estimators`

1. [code] **Iterate averaging.** ● :citet:`Polyak.Juditsky.1992` showed that
   averaging the iterates of a slowly decaying or constant-step run reaches
   the best possible constant. Consider the scalar model with a constant step
   $\eta \in (0, 2/\lambda)$, started in its stationary distribution.
    1. Show that $\textrm{Cov}(x_j, x_k) = V\rho^{|j - k|}$.
    1. Show that the average $\bar{x}_K = \frac1K \sum_{k<K} x_k$ has
       $\mathbb{E}[\bar{x}_K^2] = \frac{V}{K}\cdot\frac{1 + \rho}{1 - \rho} - O(K^{-2})$,
       and that $V(1 + \rho)/(1 - \rho) = \sigma^2/\lambda^2$ for every such
       $\eta$: to leading order the averaged iterate does not depend on the
       step.
    1. Check by simulation with $K = 2000$ and $4{,}000$ runs at $\eta = 0.3$ and
       $\eta = 0.1$, with $\lambda = \sigma = 1$. You should find
       $\mathbb{E}[\bar{x}_K^2] \approx 5 \times 10^{-4}$ in both cases, against
       last-iterate floors of $0.18$ and $0.053$.
    1. Started at $x_0 = 3$ instead, the average remembers the transient. Show
       that its contribution to $\mathbb{E}[\bar{x}_K^2]$ is about
       $(x_0/(\eta\lambda K))^2$, and decide when averaging only the second half
       of the run is better.
    1. Compare with the best $1/k$ schedule, whose constant is
       $A = \sigma^2/\mu^2$. What does each of the two routes to the rate
       $\sigma^2/(\mu^2 K)$ require?
   :exlabel:`ex_mdl-stochastic-gradient-methods-iterate-averaging`

1. [code] **Interpolation and linear convergence.** ●● Suppose each $f_i$ is
   convex with an $L_i$-Lipschitz gradient, $f$ is $\mu$-strongly convex, and
   every $f_i$ is minimized at the same point $\mathbf{x}^\star$, as when a
   model fits every training example exactly. The analysis follows
   :citet:`Vaswani.Bach.Schmidt.2019`.
    1. Show that $\mathbb{E}_i\|\nabla f_i(\mathbf{x})\|^2 \le 2L_{\max}(f(\mathbf{x}) - f^\star)$.
       (Hint: a convex $L$-smooth $h$ minimized at $\mathbf{x}^\star$ satisfies
       $\|\nabla h(\mathbf{x})\|^2 \le 2L(h(\mathbf{x}) - h(\mathbf{x}^\star))$.)
    1. Deduce that single-example SGD with a constant $\eta \le 1/L_{\max}$
       satisfies
       $\mathbb{E}\|\mathbf{x}_{k+1} - \mathbf{x}^\star\|^2 \le (1 - \eta\mu)\,\mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2$:
       linear convergence with no floor. What is
       $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ in this case?
    1. Check it on least squares with $n = 50$ examples and $d = 100$
       parameters, $\eta = 1/L_{\max}$ and $\mathbf{x}_0 = \mathbf{0}$. The
       iterates stay in the row space of $\mathbf{A}$, where $f$ is strongly
       convex, so (b) applies there. Show that the gap falls linearly with
       consistent labels and also with labels corrupted by noise of standard
       deviation $0.5$, and explain why the noise does not matter here.
    1. Repeat with $n = 200$ and $d = 100$. Now the noisy labels bring the
       floor back. Why?
   :exlabel:`ex_mdl-stochastic-gradient-methods-interpolation`

[Discussions](https://d2l.discourse.group/t/36859)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.4]{.kicker}

Stochastic Gradient Methods<br>**minibatch variance · stationary noise floor · Robbins–Monro decay · fixed-budget rates · variance reduction**
:::
:::

::: {.slide title="Why Minibatch Noise Changes the Optimization Question"}
[Motivation]{.kicker}

Training a foundation model on $n = 10^9$ examples never evaluates the exact gradient $\nabla f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})$:

::: {.cols}
::: {.col}
- **The cost of exact gradients:** Reaching six digits of accuracy on $\tfrac12(x_1^2 + 1000\,x_2^2)$ takes $\approx 7{,}000$ full-batch steps—requiring $7 \times 10^{12}$ example evaluations.
- **The minibatch shortcut:** Averaging over a random minibatch $\mathcal{B}$ of size $b = 256$ is roughly $4 \times 10^6$ times cheaper per step and points in the right direction on average.
:::

::: {.col}
- **The price of gradient noise:** On $f(x) = \tfrac12 x^2$ with unit-variance gradient noise and constant step $\eta = 0.1$, the iterate never converges to $x^\star = 0$.
- Instead, its mean square settles at $\mathbb{E}[x_k^2] \to \frac{0.1}{1.9} \approx 0.053$, wandering at distance $\approx 0.23$ from the minimum forever.
:::
:::

::: {.d2l-note}
Gradient noise shifts the central question from *how many steps* an optimizer needs to *how close to the minimum* a given step size and budget can get.
:::
:::

::: {.slide title="Minibatch Gradients and the 1/b Variance Law"}
[Minibatch Statistics]{.kicker}

::: {.cols .vc}
::: {.col}
For $b$ independent example gradients with per-example covariance $\boldsymbol{\Sigma}$, the minibatch average $\hat{\mathbf{g}}_b(\mathbf{x}) = \frac{1}{b}\sum_{i \in \mathcal{B}}\nabla f_i(\mathbf{x})$ is unbiased with variance:

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b(\mathbf{x}) - \nabla f(\mathbf{x})\|^2\big] = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$$

- **Square-root amplitude law:** RMS noise amplitude falls as $1/\sqrt{b}$, so cutting noise $10\times$ requires $100\times$ larger batches.
- **Gradient-noise scale:** Signal and noise have equal power at $b_{\mathrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}/\|\nabla f(\mathbf{x})\|^2$.
:::

::: {.col .fig}
@fig:ch35-opt-minibatch-variance-law
:::
:::
:::

::: {.slide title="Empirical Verification of the 1/b Noise Law"}
[Minibatch Statistics]{.kicker}

On $\ell_2$-regularized logistic regression ($n = 200$ examples, $d = 5$ features) evaluated at $\mathbf{x} = \mathbf{0}$, single-example noise power is $\operatorname{tr}\boldsymbol{\Sigma} \approx 1.1$, and $10{,}000$ random minibatches per size match $\operatorname{tr}\boldsymbol{\Sigma}/b$ across all scales:

@!gradient-based-optimization-sgd-variance

::: {.d2l-note}
Sampling without replacement (standard epoch shuffling) multiplies this variance by the finite-population factor $(n - b)/(n - 1)$, which is nearly $1$ whenever $b \ll n$.
:::
:::

::: {.slide title="Exact Second-Moment Recursion Under Gradient Noise"}
[The Noise Floor]{.kicker}

::: {.cols .vc}
::: {.col}
On a scalar mode $f(x) = \tfrac12\lambda x^2$ with noisy gradient $g_k = \lambda x_k + \xi_k$ ($\mathbb{E}[\xi_k \mid x_k] = 0, \mathbb{E}[\xi_k^2 \mid x_k] = \sigma^2$), the second moment $e_k = \mathbb{E}[x_k^2]$ obeys ($\rho = 1 - \eta\lambda$):

$$e_{k+1} = \rho^2\,e_k + \eta^2\sigma^2$$

- **Orthogonal noise injection:** The cross term $\mathbb{E}[x_k\xi_k] = 0$ vanishes by iterated expectations.
- **Physical balance:** Every step contracts existing variance by $\rho^2$ while injecting fresh noise power $\eta^2\sigma^2$.
:::

::: {.col .fig}
@fig:ch35-opt-second-moment-balance
:::
:::
:::

::: {.slide title="Closed-Form Trajectory and the Stationary Noise Ball"}
[The Noise Floor]{.kicker}

::: {.cols .vc}
::: {.col}
When $0 < \eta < 2/\lambda$ ($|\rho| < 1$), the linear recurrence $e_{k+1} = \rho^2 e_k + \eta^2\sigma^2$ has a unique stationary fixed point $V = \rho^2 V + \eta^2\sigma^2$:

$$e_k = V + \rho^{2k}(e_0 - V)$$

- **The stationary noise floor:**
  $$V = \frac{\eta^2\sigma^2}{1 - \rho^2} = \frac{\eta\,\sigma^2}{\lambda(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{2\lambda}$$
- **Two-phase dynamics:** A deterministic transient $\rho^{2k}(e_0 - V)$ decays geometrically into a stationary **noise ball** of RMS radius $\sqrt{V} \propto \sqrt{\eta}$.
:::

::: {.col .fig}
@fig:mdl-opt-sgd-noise-ball
:::
:::
:::

::: {.slide title="Additive Loss Floors and Linear Batch-Size Scaling"}
[The Noise Floor]{.kicker}

In deep learning we care about the expected suboptimality gap $\mathbb{E}[f(\mathbf{x}_k)] - f^\star$ rather than parameter distance:

- **Curvature cancellation in the loss floor:** For a scalar mode $f(x) = \tfrac12\lambda x^2$, multiplying $V$ by $\tfrac12\lambda$ cancels $\lambda$ in the small-step regime ($\eta\lambda \ll 1$):
  $$\mathbb{E}[f(x_\infty)] = \tfrac12\lambda V = \frac{\eta\,\sigma^2}{2(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{4}$$
  Flatter directions let the iterate wander farther ($\propto 1/\lambda$), by the exact factor that flatness discounts from the loss ($\propto \lambda$).
- **Summing across $d$ eigenmodes:** On $\tfrac12\mathbf{x}^\top\mathbf{A}\mathbf{x}$ with minibatch covariance $\boldsymbol{\Sigma}/b$, the mode variances sum to $\sum_{i=1}^d \sigma_i^2 = \operatorname{tr}\boldsymbol{\Sigma}/b$, giving total expected loss floor:
  $$\mathbb{E}[f(\mathbf{x}_\infty)] = \sum_{i=1}^d \frac{\eta\,\sigma_i^2}{2(2 - \eta\lambda_i)} \approx \frac{\eta\,\operatorname{tr}\boldsymbol{\Sigma}}{4b}$$
- **Linear scaling rule:** Because the floor depends on $(\eta, b)$ only through $\eta/b$, doubling batch size $b$ and halving learning rate $\eta$ lower the noise floor by the exact same factor.
:::

::: {.slide title="Speed vs. Accuracy on the κ = 1000 Valley"}
[Running Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
On $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ with $\eta = 1/L = 10^{-3}$ and unit noise ($\sigma_1^2 = \sigma_2^2 = 1$):

- **Flat mode ($\lambda_1 = 1$):** Contracts at $\rho_1 = 0.999$, taking $\approx 3{,}450$ steps to decay its transient to the floor $\tfrac12\lambda_1 V_1 = 2.5 \times 10^{-4}$ ($33\%$).
- **Steep mode ($\lambda_2 = 1000$):** Contracts in $1$ step ($\rho_2 = 0$), but because $2 - \eta\lambda_2 = 1$, it contributes $\tfrac12\lambda_2 V_2 = 5.0 \times 10^{-4}$ ($67\%$ of the total $7.5 \times 10^{-4}$ loss floor!).
:::

::: {.col .fig .big}
@fig:ch35-opt-valley-noise-penalty
:::
:::
:::

::: {.slide title="Breaking the Floor: The Robbins–Monro Conditions"}
[Step-Size Decay]{.kicker}

::: {.cols .vc}
::: {.col}
To break the constant-step noise floor, a decaying schedule $\eta_k \to 0$ must balance two series:

$$\sum_{k=1}^\infty \eta_k = \infty, \qquad \sum_{k=1}^\infty \eta_k^2 < \infty$$

- **Infinite reach ($\sum \eta_k = \infty$):** Ensures the iterate can travel arbitrarily far from $\mathbf{x}_0$ to reach $\mathbf{x}^\star$.
- **Finite cumulative noise ($\sum \eta_k^2 < \infty$):** Ensures total injected variance $\sigma^2\sum \eta_k^2$ remains finite (satisfied by $\eta_k = c/(k + k_0)$).
:::

::: {.col .fig}
@fig:ch35-opt-robbins-monro-balance
:::
:::
:::

::: {.slide title="Fixed Steps vs. 1/k Decay on Logistic Regression"}
[Step-Size Decay]{.kicker}

On our $\ell_2$-regularized logistic regression benchmark ($n = 200, d = 5, b = 8$), compare constant steps $\eta = 0.8$ and $\eta = 0.4$ against the decaying schedule $\eta_k = 0.8 / (1 + k/100)$:

@!gradient-based-optimization-sgd-schedule

- Halving the constant step from $0.8$ to $0.4$ cuts the asymptotic loss plateau in half (confirming $V \propto \eta$ on a non-quadratic objective), while the $1/k$ schedule drops $20\times$–$90\times$ lower and keeps descending.
:::

::: {.slide title="Why the Constant c in a c/k Schedule Is Critical"}
[Step-Size Decay]{.kicker}

::: {.cols .vc}
::: {.col}
Substituting $\eta_k = c/t$ ($t = k + k_0$) into the second-moment recursion gives:

$$e_{k+1} = \left(1 - \frac{c\mu}{t}\right)^{\!2} e_k + \frac{c^2\sigma^2}{t^2}$$

- **Critical threshold ($2c\mu > 1$):** Matching $e_k \approx A/t$ yields $A = \frac{c^2\sigma^2}{2c\mu - 1}$, minimized at $c^\star = 1/\mu$ ($e_k \approx \frac{\sigma^2}{\mu^2 k}$).
- **Catastrophic slowdown ($2c\mu < 1$):** Steps shrink faster than the iterate contracts, degrading the rate to $\Theta(k^{-2c\mu})$.
:::

::: {.col .fig}
@fig:ch35-opt-one-over-k-threshold
:::
:::
:::

::: {.slide title="Sensitivity of 1/k Schedules and Condition Number"}
[Step-Size Decay]{.kicker}

The threshold $2c\mu > 1$ explains why raw $1/k$ decay is fragile in deep learning and why modern pipelines prefer slower decay schedules or iterate averaging:

- **Optimal constant $c^\star = 1/\mu$:** Minimizing $A(c) = \frac{c^2\sigma^2}{2c\mu - 1}$ over $c > \frac{1}{2\mu}$ selects $c^\star = \frac{1}{\mu}$, achieving the minimax rate $e_k \approx \frac{\sigma^2}{\mu^2 k}$.
- **Asymmetric sensitivity:** Choosing $c$ twice too large ($c = 2/\mu$) raises $A$ by only $33\%$ ($A = \frac{4}{3}\frac{\sigma^2}{\mu^2}$), whereas choosing $c$ four times too small ($c = 1/(4\mu)$) ruins the exponent from $O(k^{-1})$ to $O(k^{-1/2})$.
- **Ill-conditioning re-enters via $k_0$:** On $\tfrac12(x_1^2 + 1000\,x_2^2)$, setting $c = 1/\mu = 1$ for the flat mode while keeping $\eta_0 = c/k_0 < 2/L = 2/1000$ for the stiff mode forces $k_0 > 500$, starving the flat mode of early progress.
:::

::: {.slide title="Modern Schedule Shapes: Cosine and WSD"}
[Practical Schedules]{.kicker}

::: {.cols .vc}
::: {.col}
Over a fixed training horizon of $K$ steps starting from base learning rate $\eta_0$, modern pretraining favors schedules that stay large early and decay smoothly to zero at $k = K$:

- **Cosine decay:**
  $$\eta_k^{\mathrm{cos}} = \tfrac12\eta_0\left(1 + \cos\frac{\pi k}{K}\right)$$
- **Warmup–Stable–Decay (WSD):** Hold $\eta_0$ constant for $80\%$ of the budget, then decay linearly over the final $20\%$, allowing checkpoints from the plateau to be branched to any horizon $K$.
:::

::: {.col .fig}
@fig:mdl-opt-schedule-zoo
:::
:::
:::

::: {.slide title="Ranking Schedule Shapes via the Exact Recursion"}
[Practical Schedules]{.kicker}

Because $e_{k+1} = (1 - \eta_k\lambda)^2 e_k + \eta_k^2\sigma^2$ holds step-by-step for any time-varying sequence $\eta_k$, we can compute the exact expected loss curve $\mathbb{E}[f(x_k)] = \tfrac12\lambda e_k$ over $K = 2000$ steps with zero sampling noise:

@!adaptive-stochastic-methods-schedules

- WSD sits on the constant-step floor ($0.0882$) for the first $1{,}600$ steps, then drops $14.4\times$ during its $400$-step linear cooldown; cosine and $c/(k+k_0)$ decay continuously to reach $\approx 1.8 \times 10^{-3}$.
:::

::: {.slide title="Fixed Compute Budgets: Transient vs. Noise Balance"}
[Fixed-Budget Rates]{.kicker}

::: {.cols .vc}
::: {.col}
When the compute budget $K$ is fixed in advance and $\mu = 0$ (no strong convexity), what constant learning rate $\eta$ minimizes the final error?

- **Universal two-term bound structure:**
  $$\text{Error}(K, \eta) \le \underbrace{\frac{\text{Initial Gap}}{\eta K}}_{\text{transient progress}} + \underbrace{\eta \cdot (\text{Noise Scale})}_{\text{stationary noise floor}}$$
- **Optimal budget scaling:** Equalizing the two competing terms always selects $\eta^\star \propto 1/\sqrt{K}$, yielding an overall convergence rate of $O(1/\sqrt{K})$.
:::

::: {.col .fig}
@fig:mdl-opt-budget-tradeoff
:::
:::
:::

::: {.slide title="Proving the O(1/√K) Rate for Convex SGD"}
[Convex Budget Proof]{.kicker}

::: {.cols .vc}
::: {.col}
For convex $f$ with $\|\mathbf{x}_0 - \mathbf{x}^\star\| \le D$ and $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le G^2$, telescoping the distance expansion and applying Jensen to $\bar{\mathbf{x}}_K = \frac{1}{K}\sum_{k=0}^{K-1}\mathbf{x}_k$ gives:

$$\mathbb{E}\big[f(\bar{\mathbf{x}}_K)\big] - f^\star \le \frac{D^2}{2\eta K} + \frac{\eta G^2}{2}$$

- **Minimax budget step:** Setting $\eta^\star = \frac{D}{G\sqrt{K}}$ equalizes the two terms and achieves $\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le \frac{DG}{\sqrt{K}}$ without requiring smoothness.
:::

::: {.col .fig}
@fig:ch35-opt-convex-sgd-telescoping
:::
:::
:::

::: {.slide title="Nonconvex SGD: The Ghadimi–Lan Stationarity Bound"}
[Nonconvex Budget Proof]{.kicker}

For nonconvex neural networks, assume $f$ is $L$-smooth with $\Delta = f(\mathbf{x}_0) - f^\star$, and stochastic gradients satisfy $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] = \nabla f(\mathbf{x}_k)$ and $\mathbb{E}[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k] \le \sigma^2$:

- **Step 1 — Stochastic descent lemma:** Plug $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\mathbf{g}_k$ into the $L$-smoothness quadratic ceiling and use $\mathbb{E}[\|\mathbf{g}_k\|^2 \mid \mathbf{x}_k] \le \|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2$:
  $$\mathbb{E}\big[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\big] \le f(\mathbf{x}_k) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(\mathbf{x}_k)\|^2 + \frac{L\eta^2\sigma^2}{2}$$
- **Step 2 — Telescope for $\eta \le 1/L$ (so $1 - L\eta/2 \ge \tfrac12$):** Summing over $k = 0, \ldots, K-1$ and dividing by $\eta K / 2$ bounds the average squared gradient norm:
  $$\frac{1}{K}\sum_{k=0}^{K-1}\mathbb{E}\big[\|\nabla f(\mathbf{x}_k)\|^2\big] \le \frac{2\Delta}{\eta K} + L\eta\sigma^2$$
:::

::: {.slide title="Optimal Nonconvex Step and Sample Complexity"}
[Nonconvex Budget Proof]{.kicker}

::: {.cols .vc}
::: {.col}
Returning a uniformly drawn iterate $\mathbf{x}_R$ ($R \sim \mathrm{Uniform}\{0, \ldots, K-1\}$) with $\eta^\star = \min\!\big(\frac{1}{L}, \sqrt{\frac{2\Delta}{L\sigma^2 K}}\big)$ certifies:

$$\mathbb{E}\big[\|\nabla f(\mathbf{x}_R)\|^2\big] \le \frac{2L\Delta}{K} + 2\sigma\sqrt{\frac{2L\Delta}{K}}$$

- **Why random stopping?** SGD cannot observe exact $\|\nabla f(\mathbf{x}_k)\|$ to pick the best iterate, so sampling $R$ turns the average into an expectation.
- **$O(1/\epsilon^4)$ sample complexity:** Halving $\mathbb{E}\|\nabla f\|^2$ by $100\times$ (so RMS gradient drops $10\times$) requires $10^4\times$ more steps.
:::

::: {.col .fig}
@fig:ch35-opt-ghadimi-lan-random-stop
:::
:::
:::

::: {.slide title="Measuring the Nonconvex Rate on a tanh Neuron"}
[Empirical Verification]{.kicker}

Training a nonconvex single-neuron regression model $f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \big(\tanh(\mathbf{a}_i^\top\mathbf{x}) - y_i\big)^2$ ($n = 64, d = 5$) with $\eta = 0.5/\sqrt{K}$ across budgets $K \in \{125, 500, 2000, 8000\}$ tracks the theoretical $K^{-1/2}$ law:

@!adaptive-stochastic-methods-ghadimi-lan

- The random-iterate metric $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2$ decays with log–log slope $\approx -0.60$, closely tracking the worst-case $-0.50$ guarantee.
:::

::: {.slide title="Finite-Sum Objectives and Interpolation vs. Noise"}
[Variance Reduction]{.kicker}

Why does constant-step SGD stall on a noise floor? At the global minimizer $\mathbf{x}^\star$, the average gradient vanishes ($\nabla f(\mathbf{x}^\star) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x}^\star) = \mathbf{0}$), but individual example gradients $\nabla f_i(\mathbf{x}^\star)$ generally do not:

- **The interpolation regime ($\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ for all $i$):** When an overparameterized network fits every training example with zero loss, every component gradient vanishes at $\mathbf{x}^\star$. The noise variance $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ vanishes automatically at the optimum, and constant-step SGD converges linearly with no noise floor!
- **Non-interpolating finite sums ($\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$):** When individual examples disagree at $\mathbf{x}^\star$, we can cancel their persistent variance using a Monte Carlo **control variate**—subtracting a correlated gradient whose exact mean over the $n$ examples is known.
:::

::: {.slide title="SVRG: Snapshot Control Variates Shrink Variance"}
[Variance Reduction]{.kicker}

::: {.cols .vc}
::: {.col}
**SVRG** periodically saves a snapshot $\tilde{\mathbf{x}}$ with exact full gradient $\nabla f(\tilde{\mathbf{x}})$ and forms the unbiased control-variate estimator:

$$\tilde{\mathbf{g}}_i(\mathbf{x}) = \nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}}) + \nabla f(\tilde{\mathbf{x}})$$

- **Vanishing variance:** By $L_{\max}$-smoothness of each $f_i$, $\mathbb{E}_i\|\tilde{\mathbf{g}}_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2 \le L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2$.
- **Linear convergence at fixed $\eta$:** As $\mathbf{x}, \tilde{\mathbf{x}} \to \mathbf{x}^\star$, the gradient noise vanishes without decaying $\eta$.
:::

::: {.col .fig}
@fig:ch35-opt-svrg-control-variate
:::
:::
:::

::: {.slide title="SVRG vs. SGD on Ridge Regression"}
[Variance Reduction]{.kicker}

Comparing single-example SGD and SVRG at the same constant step $\eta = 0.02$ and the same budget of $3n$ example gradients per epoch on ridge regression ($n = 300, d = 12$):

@!mdl-adaptive-svrg

- **SGD** stalls permanently at the exact theoretical noise floor $\sum_i \frac{\eta\sigma_i^2}{2(2-\eta\lambda_i)} \approx 2 \times 10^{-2}$, whereas **SVRG** contracts geometrically by $\sim 8\times$ per epoch down to machine precision ($10^{-11}$).
:::

::: {.slide title="Concept Check: Noise Floors, Schedules, and Budgets"}
[Concept Check]{.kicker}

Test your understanding of stochastic gradient dynamics:

1. On a scalar quadratic with $\lambda = \sigma = 1$, what is the expected loss floor at $\eta = 0.3$, and by what factor does the floor drop if we simultaneously halve $\eta$ to $0.15$ and double the batch size $b$?
2. Suppose a strongly convex objective has minimum curvature $\mu = 0.25$. What condition must $c$ satisfy for the schedule $\eta_k = c/(k + k_0)$ to achieve an $O(1/k)$ rate, and what choice of $c$ minimizes the asymptotic constant?
3. For a smooth nonconvex loss with $L = \Delta = \sigma = 1$ and budget $K = 10^6$, what step size $\eta$ and stationarity bound does the Ghadimi–Lan theorem prescribe?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** At $\eta = 0.3$, the loss floor is $\frac{\eta\sigma^2}{2(2 - \eta\lambda)} = \frac{0.3}{3.4} \approx 0.0882$. Halving $\eta$ and doubling $b$ cuts $\eta/b$ by a factor of **$4\times$** (and slightly more, $\approx 4.3\times$, because $2 - \eta\lambda$ increases from $1.7$ to $1.85$). **(2)** Requiring $2c\mu > 1$ with $\mu = 0.25$ forces **$c > 2$**; below $c = 2$, convergence slows to $O(k^{-c/2})$. The optimal constant is $c^\star = 1/\mu = \mathbf{4}$. **(3)** $\eta = \sqrt{2\Delta/(L\sigma^2 K)} = \sqrt{2} \times 10^{-3} \approx \mathbf{0.00141}$, guaranteeing $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2 \le 2 \times 10^{-6} + 2\sqrt{2} \times 10^{-3} \approx \mathbf{0.00283}$.
:::
:::

::: {.slide title="Summary: Stochastic Gradient Methods and Rates"}
[Key Takeaways]{.kicker}

| Regime / Algorithm | Step Size / Estimator | Convergence Rate / Floor | Key Mechanism & Practical Takeaway |
| :--- | :--- | :--- | :--- |
| **Constant-Step SGD** | Fixed $\eta \in (0, 2/L)$, minibatch $b$ | Transient $\rho^{2k} \to$ loss floor $\approx \frac{\eta \operatorname{tr}\boldsymbol{\Sigma}}{4b}$ | Fast initial descent, stalls in a noise ball $\propto \eta/b$ |
| **Robbins–Monro Decay** | $\eta_k = \frac{c}{k + k_0}$ with $2c\mu > 1$ | $\mathbb{E}\|\mathbf{x}_k - \mathbf{x}^\star\|^2 \approx \frac{c^2\sigma^2}{(2c\mu - 1)k}$ | Breaks the noise floor; degrades to $O(k^{-2c\mu})$ if $c < \frac{1}{2\mu}$ |
| **Convex Fixed Budget** | Fixed $\eta = \frac{D}{G\sqrt{K}}$, averaged $\bar{\mathbf{x}}_K$ | $\mathbb{E}[f(\bar{\mathbf{x}}_K)] - f^\star \le \frac{DG}{\sqrt{K}}$ | Minimax optimal $O(K^{-1/2})$ without smoothness |
| **Nonconvex Fixed Budget** | Fixed $\eta = \sqrt{\frac{2\Delta}{L\sigma^2 K}}$, random $\mathbf{x}_R$ | $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2 \le \frac{2L\Delta}{K} + 2\sigma\sqrt{\frac{2L\Delta}{K}}$ | Certifies approximate stationarity in $O(1/\epsilon^4)$ steps |
| **SVRG / SAGA** | $\nabla f_i(\mathbf{x}) - \nabla f_i(\tilde{\mathbf{x}}) + \nabla f(\tilde{\mathbf{x}})$ | Variance $\le L_{\max}^2\|\mathbf{x} - \tilde{\mathbf{x}}\|^2 \to 0$; linear rate | Eliminates finite-sum noise floor at constant $\eta$ |
:::

