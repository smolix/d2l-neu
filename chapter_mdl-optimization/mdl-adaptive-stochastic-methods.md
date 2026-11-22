# Adaptive Methods
:label:`sec_mdl-adaptive-stochastic-methods`

The transformers in this book, like most large neural networks, are trained
with Adam or its variant AdamW (:numref:`sec_adamw`) rather than with plain
stochastic gradient descent. A primary reason is the step size. The methods of
:numref:`sec_mdl-stochastic-gradient-methods` apply a single step size to every
coordinate, adjusting it only over time. Yet a modern network has $10^7$ to
$10^{11}$ parameters whose curvatures differ by orders of magnitude and are
unknown in advance. When all coordinates share one step size, the steepest
direction caps the step and the flat directions barely move. This section asks
how an optimizer can choose a separate step size for each coordinate from the
gradients alone—and where that strategy breaks down.

Our running example illustrates how much is at stake. Consider the narrow
valley $f(\mathbf{x}) = \tfrac12\bigl(x_1^2 + 1000\, x_2^2\bigr)$ starting at
$(1, 1)$. Gradient descent with the optimal single step size,
$\eta = 2/1001$, needs 6,160 steps to drive $f$ below $10^{-8}$, because a
step small enough to keep $x_2$ stable is a thousand times too small for
$x_1$. The heavy-ball method of :numref:`sec_mdl-gradient-based-optimization`,
which adds a fraction of the previous step to each new one, cuts that to 296
steps. An optimizer that gives each coordinate its own step size, estimated
purely from the gradients it has observed, needs only a few dozen. Even more
striking is what that count does not depend on: make the valley a hundred
times narrower still, and the iteration count grows by just six steps.

Headline counts like these are easy to produce and easy to misread, so much of
this section is devoted to making the comparison fair. We will run heavy ball
at its theoretically optimal settings, give Adam a simple sweep of step sizes,
and also rotate the valley by $45^\circ$. Under that fairer test, Adam's
advantage looks quite different from what the first count suggests.

The underlying mechanism is simple: divide each coordinate's step by the root
mean square of its past gradients, as AdaGrad, RMSProp, and Adam do. We first
prove that this normalization makes Adam blind to the scale of each
coordinate, and then follow that symmetry to the boundaries where it breaks: a
rotation of the valley, a periodic sequence of convex losses, weight decay,
the opening steps of training, and the matrix structure of a layer. Two
results carry the most weight: the scale-invariance proposition and the
fixed-point analysis of AdamW. The regret bounds for AdaGrad and the
structured preconditioners at the end can be skipped on a first reading.
Throughout, $t$ indexes optimizer steps, $h_i$ denotes the curvature of
coordinate $i$, and $\lambda$ is the weight-decay coefficient.

```{.python .input #adaptive-stochastic-methods-imports}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #adaptive-stochastic-methods-imports}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Per-Coordinate Step Sizes
:label:`subsec_mdl-per-coordinate`

### A Diagonal Quadratic

Gradient descent on a quadratic requires a number of steps proportional to
the condition number $\kappa = h_{\max}/h_{\min}$, because its single step
size must stay below $2/h_{\max}$
(:numref:`sec_mdl-gradient-based-optimization`). The simplest setting where
a remedy becomes visible is the two-variable diagonal quadratic,

$$
f(\mathbf{x}) = \tfrac12\bigl(h_1 x_1^2 + h_2 x_2^2\bigr),
\qquad
\frac{\partial f}{\partial x_i} = h_i x_i .
$$

Because each partial derivative depends only on its own coordinate, nothing
forces the two coordinates to share a step size. With a separate step
$\eta_i$ for each coordinate, $x_i$ updates to
$x_i - \eta_i h_i x_i = (1 - \eta_i h_i)\, x_i$, and choosing
$\eta_i = 1/h_i$ sends every coordinate to zero in a single step. For
$h = (1, 1000)$, the ideal steps are $1$ and $10^{-3}$. This is Newton's
method for a diagonal Hessian, and the condition number vanishes from the
iteration count.

Two obstacles stand between this ideal and a practical optimizer: the
curvatures $h_i$ are unknown and shift over the course of training, and the
gradients we observe are noisy minibatch estimates. We must therefore infer
each coordinate's step size from the gradients alone, using gradient
magnitude as a proxy. On the diagonal quadratic, the gradient entry
$g_i = h_i x_i$ has magnitude $h_i |x_i|$. Whenever coordinates are of
comparable size, a coordinate with large gradients also has high curvature,
so dividing by its gradient magnitude effectively divides by its curvature.
When coordinates differ sharply in scale, however, $g_i/|g_i|$ still has
unit magnitude: normalization fixes the distance traveled per step at $\eta$,
rather than taking the true Newton step $x_i$.

### AdaGrad

Varying curvature is one reason to give each coordinate its own step size. A
second, independent reason is that coordinates are updated at very different
frequencies. A global decaying step size such as $\eta/\sqrt{t}$—the
classical choice in online and nonsmooth convex optimization—treats every
coordinate identically. Suppose coordinate 1 receives a gradient of $\pm 1$
at every step while coordinate 2 receives one only once every $10^4$ steps,
as happens for a rare word's weight in a bag-of-words model. By $t = 10^4$,
a global schedule reduces both step sizes to $\eta/100$, so the single piece
of evidence about the rare coordinate moves it by only a hundredth of
$\eta$. If instead each coordinate accumulates its own squared gradients,
the running sums are $10^4$ and $1$, and dividing $\eta$ by their square
roots gives the busy coordinate $\eta/100$ while leaving the rare coordinate
at $\eta$.

This is **AdaGrad** :cite:`Duchi.Hazan.Singer.2011`. Write
$\mathbf{g}_t \odot \mathbf{g}_t$ for the vector of squared gradient entries
(abbreviated as $\mathbf{g}_t^2$ from here on), with square roots and
divisions of vectors likewise applied elementwise. AdaGrad maintains the
running sum $\mathbf{s}_t$ of squared gradients and divides the step by its
square root:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\, \mathbf{g}_t,
$$
:eqlabel:`eq_mdl-opt-adagrad`

starting from $\mathbf{s}_0 = \mathbf{0}$, where a small constant $\epsilon$
prevents division by zero (:eqref:`eq_adagrad` in the main text places
$\epsilon$ inside the square root, which makes no qualitative difference
here). Each coordinate's step size now decays according to its own gradient
history rather than the global step count.

*The metric view.* Steepest descent depends on how we measure step length.
Under the weighted norm
$\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\, \mathbf{d}}$
for a positive definite matrix $\mathbf{P}$, the steepest descent direction
is $-\mathbf{P}^{-1}\nabla f$ (Exercise
:exref:`ex_mdl-gradient-based-optimization-show-that-is-de` of
:numref:`sec_mdl-gradient-based-optimization`). AdaGrad is steepest descent in
the diagonal metric $\mathbf{P}_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$:
moving along coordinate $i$ is penalized in proportion to the accumulated
evidence $\sqrt{\sum_{\tau \le t} g_{\tau,i}^2}$ that its gradient has been
large. A matrix such as $\mathbf{P}_t$ that rescales the gradient before
taking a step is called a **preconditioner**. Whereas Newton's method uses
the true curvature $\mathrm{diag}(h_i)$, AdaGrad substitutes the gradient
history (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`).

*The regret view.* AdaGrad was originally designed for **online convex
optimization**, in which convex losses $f_1, f_2, \ldots$ arrive one at a
time and the learner must choose $\mathbf{x}_t$ before seeing $f_t$.
Performance over $T$ rounds is measured by the **regret**,
$\sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^T f_t(\mathbf{x})$:
the cumulative excess loss relative to the best fixed point chosen in
hindsight. An algorithm learns in this framework when its average regret
(regret divided by $T$) tends to zero.

On a convex set of Euclidean diameter $D_2$ with bounded gradients
$\|\mathbf{g}_t\| \le G$, online gradient descent with a tuned global step
size achieves regret of order $D_2 G \sqrt{T}$
(proved in Exercise :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`).
AdaGrad, by contrast, achieves regret of order :cite:`Duchi.Hazan.Singer.2011`

$$
D_\infty \sum_{i=1}^{d} \sqrt{\textstyle\sum_{t=1}^{T} g_{t,i}^2} ,
$$

where $D_\infty$ is the diameter of the set in the $\ell_\infty$ norm. Each
of the $d$ coordinates pays only for its own gradient history rather than
for the worst-case gradient norm across all coordinates.

On the box $[-1, 1]^d$, we have $D_2 = \sqrt{d}\, D_\infty$. When gradient
mass concentrates on a small subset of coordinates, AdaGrad's bound is up to
a factor of $\sqrt{d}$ smaller; when the mass is spread evenly across
coordinates, the two bounds match. For one-hot gradients that hit coordinate
$i$ with probability $p_i$, the expected ratio of the two bounds is at most
$\sum_i \sqrt{p_i}/\sqrt{d}$
(Exercise :exref:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`).
For uniform $p_i$ this ratio is $1.00$ even though every individual gradient
is sparse, whereas for word frequencies following Zipf's law with exponent
$1$ at $d = 10^4$ it drops to $0.63$. AdaGrad's advantage therefore comes from
the *concentration* of gradient mass on certain coordinates, not from
sparsity by itself.

The square root in :eqref:`eq_mdl-opt-adagrad` is not arbitrary: it is the
exact exponent that balances the two competing terms in the regret proof
(Exercise :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`c).

Outside online convex optimization, however, unbounded accumulation becomes
AdaGrad's main weakness. Under persistent gradient noise of variance
$\sigma^2$, the sum grows as $s_{t,i} \approx \sigma^2 t$, so the effective
step size shrinks as $\eta/(\sigma\sqrt{t})$. AdaGrad thus carries a
built-in decay schedule tied to the step count rather than to optimization
progress. That $1/\sqrt{t}$ rate is optimal for convex regret, but on long
nonconvex training runs it causes the step size to decay relentlessly whether
or not the loss is still improving, leaving almost no step size by the time
the optimizer reaches a late plateau.

### RMSProp

**RMSProp** :cite:`Tieleman.Hinton.2012` fixes this decay by replacing the
all-time sum with an exponential moving average, keeping the scale estimate
responsive to recent gradients:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon}\, \mathbf{g}_t .
$$
:eqlabel:`eq_mdl-opt-rmsprop`

Unrolling the recursion shows that $\mathbf{v}_t$ assigns the squared
gradient from $k$ steps ago the weight $(1 - \beta_2)\beta_2^k$, so old
gradients halve in weight roughly every $\log 2/(1 - \beta_2)$ steps—every
7 steps at RMSProp's customary $\beta_2 = 0.9$, and every 693 steps at
$\beta_2 = 0.999$. Step sizes no longer vanish by construction. The
trade-off is that any single large gradient is forgotten on that same
timescale, a vulnerability we will exploit in a counterexample below.

### Adam and Its Bias Correction

**Adam** :cite:`Kingma.Ba.2014` combines RMSProp's second-moment estimate
with an exponential moving average of the gradient itself, and corrects both
averages for their initialization at zero. In the notation of the main text
(:eqref:`eq_adam-moments`, :eqref:`eq_adam-update`), the update reads

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t), \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta\; \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\end{aligned}
$$
:eqlabel:`eq_mdl-opt-adam`

with $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$ and the standard defaults
$\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$. The first
moment $\mathbf{m}_t$ is the heavy-ball velocity of
:numref:`subsec_mdl-momentum-acceleration` scaled by $1 - \beta_1$, which
turns a running sum into a weighted average and smooths the minibatch noise
that plain RMSProp steps along directly.

Dividing by $1 - \beta_2^t$ compensates for starting from
$\mathbf{v}_0 = \mathbf{0}$. Unrolling the recursion gives
$\mathbf{v}_t = (1 - \beta_2) \sum_{\tau \le t} \beta_2^{\,t-\tau}\, \mathbf{g}_\tau^2$.
If the squared-gradient scale is stationary, with
$\mathbb{E}[\mathbf{g}_\tau^2] = \overline{\mathbf{g}^2}$ for every $\tau$,
summing the geometric series yields

$$
\mathbb{E}[\mathbf{v}_t]
= (1 - \beta_2)\, \overline{\mathbf{g}^2} \sum_{\tau=1}^{t} \beta_2^{\,t-\tau}
= \left(1 - \beta_2^t\right) \overline{\mathbf{g}^2} .
$$
:eqlabel:`eq_mdl-opt-bias-correction`

Thus $\hat{\mathbf{v}}_t$ is an unbiased estimate of $\overline{\mathbf{g}^2}$
at every step $t$ under stationarity, and the same calculation with $\beta_1$
shows that $\hat{\mathbf{m}}_t$ is unbiased for the mean gradient.

Without this correction, the initialization deficit is both large and
long-lived: at the default $\beta_2 = 0.999$, $\mathbf{v}_{10}$ captures
only about $1\%$ of its stationary target, and even at $t = 1000$ it has
reached just $63\%$ (:numref:`sec_adam`). More striking is how the error
evolves over time. Dropping both bias corrections mis-scales the update
$\mathbf{m}_t/\sqrt{\mathbf{v}_t}$ by the factor
$(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$, which starts at $3.16$ at $t = 1$
and—perhaps surprisingly—climbs to a peak of $6.6$ at $t = 12$. The
first-moment average fills in roughly ten steps, whereas the second-moment
average needs about a thousand. :numref:`fig_mdl-opt-bias-correction` plots
both the deficit and the resulting step inflation.

![Bias correction in Adam. Left: starting from zero, the running average $\mathbf{v}_t$ captures only the fraction $1-\beta_2^t$ of the squared-gradient scale ($63\%$ at $t=1000$ for $\beta_2=0.999$); dividing by $1-\beta_2^t$ restores the full scale at every step. Right: without bias correction, the update $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$ is inflated by $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$, starting at $3.16$ at $t=1$, peaking at $6.6$ at $t=12$ as the numerator fills in ten steps while the denominator takes a thousand, and returning to $1$ only on the $1/(1-\beta_2)$ timescale. Uncorrected Adam thus takes its largest steps when its preconditioner rests on the fewest gradients.](../img/mdl-opt-bias-correction.svg)
:label:`fig_mdl-opt-bias-correction`

Even an unbiased $\hat{\mathbf{v}}_t$ does not make the step size itself
unbiased. Because Adam divides by $\sqrt{\hat{\mathbf{v}}_t}$ rather than
$\hat{\mathbf{v}}_t$, concavity of the square root biases the denominator
downward: at $t = 1$, $\sqrt{\hat{v}_1} = |g_1|$, and by Jensen's inequality
$\mathbb{E}|g| < \sqrt{\mathbb{E}[g^2]}$ unless $|g|$ is constant. This is
why Adam's very first step has magnitude $\eta$ in every coordinate, rather
than $\eta\,|g_1|/\sigma$ for a gradient of scale $\sigma$. We examine how
much this matters in :numref:`subsec_mdl-schedules-warmup`.

### Scale Invariance

In our opening example, $h_2 = 1000\, h_1$, yet Adam crosses the valley as
if that thousandfold ratio did not exist. The reason is an exact symmetry of
the update, and it is the central result of this section.

**Proposition (Adam ignores the scale of each coordinate).** *Let
$\epsilon = 0$ and let $\mathbf{C} = \mathrm{diag}(c_1, \ldots, c_d)$ with
every $c_i > 0$. Suppose two problems have gradients related by
$\nabla \tilde{f}(\mathbf{x}) = \mathbf{C}\, \nabla f(\mathbf{x})$ at every
$\mathbf{x}$. Then Adam with the same $\beta_1$, $\beta_2$, step schedule,
and starting point produces identical iterates on both problems, as long as
the updates remain defined.*

In other words, the second problem is the first with coordinate $i$'s
gradient rescaled by a positive constant $c_i$. We denote its quantities
with a tilde.

**Proof.** We proceed by induction on $t$. Suppose the iterates agree up to
$\mathbf{x}_t$. Then $\tilde{\mathbf{g}}_\tau = \mathbf{C}\mathbf{g}_\tau$
for every $\tau \le t$. Because the recursions for $\mathbf{m}$ and
$\mathbf{v}$ are linear in $\mathbf{g}_\tau$ and $\mathbf{g}_\tau^2$, we
obtain $\tilde{m}_{t,i} = c_i m_{t,i}$ and $\tilde{v}_{t,i} = c_i^2 v_{t,i}$.
The bias-correction denominators are the same scalars in both problems, so
the corrected ratio $\hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$ of the second
problem is
$c_i \hat{m}_{t,i}/(c_i \sqrt{\hat{v}_{t,i}}) = \hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$.
The updates therefore coincide, giving $\tilde{\mathbf{x}}_{t+1} = \mathbf{x}_{t+1}$. $\blacksquare$

On the diagonal quadratic, $\nabla f(\mathbf{x}) = \mathrm{diag}(h)\,\mathbf{x}$
is simply the gradient of the round bowl $\tfrac12\|\mathbf{x}\|^2$
multiplied by $\mathbf{C} = \mathrm{diag}(h)$. Consequently, Adam on the
$1000{:}1$ valley traces, step for step, the exact same trajectory as Adam
on the round bowl—and on the bowl, two coordinates started at equal values
remain equal forever. Each coordinate follows the same one-dimensional
iteration on $\tfrac12 x^2$,

$$
x_{t+1} = x_t \Bigl(1 - \frac{\eta}{\sqrt{\hat{v}_t}}\Bigr)
\qquad (\beta_1 = 0),
$$

where $\hat{v}_t$ averages the past squares $x_\tau^2$ for $\tau \le t$.
Because earlier iterates were larger, $\sqrt{\hat{v}_t}$ exceeds $|x_t|$
from the second step onward, so each update moves by less than $\eta$ and
contracts geometrically by the factor $1 - \eta/\sqrt{\hat{v}_t}$. At
$\eta = 0.1$, this factor tightens from $0.9$ at $t = 1$ to $0.67$ at
$t = 48$ as $\hat{v}_t$ catches up with the shrinking iterates. The iterates
themselves are completely independent of $h_2/h_1$. Only the stopping test
depends on the curvature ratio, because
$f(\mathbf{x}_t) = \tfrac12(h_1 + h_2)\, x_t^2$ weights the common iterate
by the curvatures. Making the valley far narrower should therefore add only
a handful of geometric contraction steps, never a factor of $\kappa$.

What the step count *does* depend on is the starting distance measured in
units of $\eta$, since the early steps have length roughly $\eta$ per
coordinate regardless of curvature. An experiment in the next subsection
checks both predictions directly.

In short, Adam measures every coordinate in units of its own gradient size,
making coordinate-wise scaling invisible to it. Three caveats bound this
invariance. First, $\epsilon > 0$ breaks the symmetry once a coordinate's
gradients shrink to the order of $\epsilon$. Second, the
proposition says nothing about a change of basis that mixes coordinates,
which we test next on the rotated valley. Third, Adam is not in general sign
descent: its update reduces to the **sign step**
$\eta\,\mathrm{sign}(\mathbf{g}_t)$ in only three special cases—at $t = 1$;
when $\beta_1 = \beta_2 = 0$ (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d); and, for
$\beta_1 = 0$, whenever each coordinate's gradient magnitude stays constant
over the memory of $\hat{\mathbf{v}}$.

## Coordinate Scaling and Its Limits

### Aligned and Rotated Valleys

How much of gradient descent's 6,160 steps does coordinate scaling alone
eliminate, and does that gain survive when the valley is rotated? To
separate per-coordinate scaling from momentum, we compare four methods:
gradient descent at its optimal single step size; heavy ball at its optimal
step size and momentum; Adam with $\beta_1 = 0$, which only rescales; and
Adam with its default momentum $\beta_1 = 0.9$. To rotate the problem by an
angle $\theta$, we replace $\mathrm{diag}(h)$ with
$\mathbf{A}_\theta = \mathbf{R}_\theta\, \mathrm{diag}(h)\, \mathbf{R}_\theta^\top$,
where $\mathbf{R}_\theta$ is the $2 \times 2$ rotation matrix, and rotate
the starting point alongside it, $\mathbf{x}_0 = \mathbf{R}_\theta (1, 1)^\top$,
so that the initial point retains the same components along both curvature
axes. At $\theta = \pi/4$, this gives
$\mathbf{A} = \left(\begin{smallmatrix}500.5 & -499.5\\ -499.5 & 500.5\end{smallmatrix}\right)$.
The code below defines the valley, a loop that counts steps to reach the
target tolerance, and the three optimizer updates.

```{.python .input #adaptive-stochastic-methods-adam-vs-gd}
h = np.array([1.0, 1000.0])                      # curvatures h_i, kappa = 1000

def rotated(theta):
    """A = R diag(h) R^T and the start R (1, 1): the valley turned by theta."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(h) @ R.T, R @ np.ones(2)

def run(step, A, x0, K=40000, tol=1e-8):
    """First t with f(x_t) = x_t^T A x_t / 2 < tol, or None within K steps."""
    x, state = x0.copy(), {}
    for t in range(1, K + 1):
        x = step(x, A @ x, t, state)
        if 0.5 * x @ A @ x < tol:
            return t
    return None

def gd(eta):
    return lambda x, g, t, state: x - eta * g

def heavy_ball(eta, beta):
    def step(x, g, t, state):
        state['d'] = beta * state.get('d', 0.0) - eta * g    # velocity
        return x + state['d']
    return step

def adam(eta, beta1=0.9, beta2=0.999, eps=1e-8):
    def step(x, g, t, state):
        state['m'] = beta1 * state.get('m', 0.0) + (1 - beta1) * g
        state['v'] = beta2 * state.get('v', 0.0) + (1 - beta2) * g * g
        m_hat, v_hat = state['m'] / (1 - beta1**t), state['v'] / (1 - beta2**t)
        return x - eta * m_hat / (np.sqrt(v_hat) + eps)
    return step
```

Before reading the step counts, notice what information each method is
given. Gradient descent and heavy ball run at their theoretically optimal
settings, computed in closed form from the true curvatures $1$ and $1000$.
That is a generous handicap, since in practice nobody knows the extreme
curvatures of a neural network. Adam has no closed-form formula, so we give
it what a practitioner would use: a log-spaced sweep of seven step sizes
from $0.002$ to $2$ (spaced by factors of about three), reading each row at
its best entry. The grid deliberately skips $\eta = 1$: because Adam's first
step has length $\eta$ in every coordinate
(:numref:`subsec_mdl-schedules-warmup`), starting at $(1, 1)$ with
$\eta = 1$ would land directly on the origin in one step. That would be a
coincidence of the starting point, not a fair measure of optimizer speed.

```{.python .input #mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-1}
mu, L = h.min(), h.max()
rho = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
tuned = {'GD, eta = 2/(mu + L)': gd(2 / (mu + L)),
         'heavy ball, tuned': heavy_ball(4 / (np.sqrt(L) + np.sqrt(mu))**2,
                                         rho**2)}
valleys = {'aligned': rotated(0.0), 'rotated 45': rotated(np.pi / 4)}
print(f'{"steps to f < 1e-8":28s}{"aligned":>9s}{"rotated 45":>12s}')
for name, step in tuned.items():
    hits = [run(step, *problem) for problem in valleys.values()]
    print(f'{name:28s}{hits[0] or "never":>9}{hits[1] or "never":>12}')
etas = np.geomspace(0.002, 2.0, 7)              # step sizes for Adam
print(f'\n{"Adam, eta =":28s}' + ''.join(f'{eta:>7.2g}' for eta in etas))
for beta1 in (0.0, 0.9):
    for name, problem in valleys.items():
        hits = [run(adam(eta, beta1=beta1), *problem) for eta in etas]
        print(f'{f"beta1 = {beta1}, {name}":28s}'
              + ''.join(f'{k or "never":>7}' for k in hits))
```

Consider the aligned valley first. Per-coordinate scaling alone ($\beta_1 = 0$)
needs just 22 steps at its best step size, compared with 6,160 for gradient
descent and 296 for heavy ball—thirteen times fewer steps than a heavy-ball
method given both true curvatures, from an optimizer told neither. That row
also clarifies what tuning Adam actually adjusts. The iteration count scales
roughly as $1/\eta$, because the early steps have length about $\eta$ and
the total count mainly reflects the distance to the minimizer measured in
step lengths. At the two largest step sizes, however, scaling alone never
converges: the iterate overshoots the minimum and settles into a cycle that
bounces each coordinate between $+\eta/2$ and $-\eta/2$. Adam has thus
traded one unknown—the curvature—for another: the distance to the minimizer
in units of $\eta$.

Adding Adam's default momentum ($\beta_1 = 0.9$) brings the count to between
165 and 205 steps across the entire range of step sizes from $0.02$ to $2$.
Two features stand out. The first is robustness: a hundredfold change in
$\eta$ shifts the iteration count by only a quarter, and the two largest
step sizes that cycled forever without momentum now converge cleanly because
the momentum average damps the alternating signs. Heavy ball enjoys no such
margin: its optimal step size sits within $0.1\%$ of its stability limit
$(2 + 2\beta)/\lambda_{\max}$ (Exercise
:exref:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum`a of
:numref:`sec_mdl-gradient-based-optimization`), so its 296-step result
presupposes near-exact knowledge of the largest curvature. The second
feature is the price of that robustness: momentum makes Adam roughly seven
times slower than scaling alone. The cause is the $\sqrt\beta$ contraction
limit from :numref:`subsec_mdl-momentum-acceleration`: an iteration with
momentum $\beta$ contracts any mode with complex roots at rate $\sqrt\beta$
per step, and no faster. After per-coordinate scaling, both coordinates
effectively run on a round bowl of condition number one, where the optimal
momentum is zero. With $\beta_1 = 0.9$, the contraction factor is
$\sqrt{0.9} \approx 0.95$ per step, which predicts roughly 230 steps to
reach the tolerance, closely matching the observed counts. Momentum that
accelerates progress along an ill-conditioned valley acts as a brake on a
round bowl, yet Adam's default applies it unconditionally.

Now turn to the rotated valley. Gradient descent and heavy ball require the
exact same number of steps as before, because their updates are
rotation-invariant and simply rotate with the problem. Adam's updates are
not. Scaling alone fails to reach the tolerance at every single step size on
the grid. Adam with momentum does converge, but only toward the upper end of
the grid: its best count is 559 at $\eta = 2$, nearly twice heavy ball's
296, and at the smallest step size it needs 12,862 steps—twice as many as
plain gradient descent. Its step-size robustness vanishes too, with counts
now spanning more than a twentyfold range across the grid.

The fair verdict therefore has two halves. When the coordinate axes align
with the principal curvature directions, per-coordinate scaling beats a
perfectly tuned heavy ball by an order of magnitude while requiring far
less prior knowledge. When the valley is rotated, the benefit of diagonal
scaling disappears completely; whatever convergence remains comes from
momentum, and a well-tuned heavy ball performs that job twice as fast. Let
us examine both halves in closer detail.

On the aligned valley, our scale-invariance proposition predicts that both
coordinates follow identical iterates while the stiff coordinate receives an
effective step a thousand times smaller. Below, we print the iterates and
effective step sizes $\eta/\sqrt{\hat{\mathbf{v}}_t}$ of scaling-only Adam
($\epsilon = 0$, $\eta = 0.1$). The last two lines then test what actually
controls the iteration count by varying the curvature ratio across five
orders of magnitude and moving the starting point a hundred times farther
from the origin.

```{.python .input #mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-2}
A, x = rotated(0.0)
step, state, eta, gap = adam(0.1, beta1=0.0, eps=0.0), {}, 0.1, 0.0
for t in range(1, 49):
    x = step(x, A @ x, t, state)
    gap = max(gap, abs(x[0] - x[1]))
    if t in (1, 10, 48):
        eff = eta / np.sqrt(state['v'] / (1 - 0.999**t))
        print(f't = {t:2d}: x = {x[0]:.3g}, {x[1]:.3g};  '
              f'eta/sqrt(v_hat) = {eff[0]:.3g}, {eff[1]:.3g}')
print(f'largest |x_1 - x_2| over the run: {gap:.1e}')
counts = [run(adam(0.1, beta1=0.0), np.diag([1.0, r]), np.ones(2))
          for r in (1.0, 1e3, 1e5)]
print('steps to f < 1e-8 at curvature ratios 1, 1e3, 1e5:', *counts)
print('steps from (100, 100) at ratio 1e3:',
      run(adam(0.1, beta1=0.0), A, 100 * np.ones(2)))
```

From the very first step, the effective step sizes stand in the exact ratio
$1000{:}1$—the true curvature ratio recovered from gradient magnitudes alone,
without ever forming a Hessian. The two coordinates agree to floating-point
precision throughout, and the effective step on the flat coordinate grows
from $0.1$ to $0.33$ as $\hat{v}_t$ adapts to the shrinking gradients.
Raising the curvature ratio from $1$ to $10^3$ to $10^5$ increases the step
count only from 40 to 48 to 54: a hundred-thousandfold worsening in
conditioning costs just fourteen extra steps. Moving the starting point a
hundred times farther away, however, pushes the count to 4,145. The
iteration count measures starting distance in units of $\eta$, and barely
feels the curvature at all.

Why is a $45^\circ$ rotation fatal to all of this? The rotated matrix has
diagonal entries near $500$, and its steep direction $(-1, 1)/\sqrt{2}$
projects equally onto both coordinate axes. Every coordinate's gradient is
therefore dominated by the steep mode, so both coordinates accumulate nearly
identical scale estimates (across the runs, their $\hat{v}$ values agree to
within one percent). Because a per-coordinate preconditioner assigns only
one scalar to each coordinate, it cannot shrink the steep component of a
coordinate's gradient without equally shrinking its flat component. With
equal scales on both coordinates, the preconditioner reduces to a scalar
multiple of the identity. Scaling-only Adam ($\beta_1 = 0$) thus reverts to
gradient descent whose step size is dictated by the steep mode, leaving the
flat mode to crawl. Adam with $\beta_1 = 0.9$ reverts to heavy ball: its
momentum is close to the optimal $0.88$, but its effective step
$\eta/\sqrt{\hat{v}_t}$ is governed by gradient history rather than
curvature, leaving even the best grid point twice as slow as tuned heavy
ball.

Without momentum, the rotated run ends in oscillation. As the steep
component initially shrinks, $\hat{\mathbf{v}}$ decreases and the effective
step size grows until it crosses the stability limit $2/h_{\max}$ of the
steep mode. At $\eta = 0.2$—the optimal step size for the aligned
valley—the loss drops to about $10^{-3}$ around $t = 1{,}700$, at which
point the steep mode begins to oscillate. That steady oscillation locks the
gradient magnitude in place, driving $\hat{\mathbf{v}}_t \to \mathbf{g}_t^2$
so that the update collapses to the sign step
$\eta\,\mathrm{sign}(\mathbf{g}_t)$ of length $\eta\sqrt{2}$ across the
valley. The iterate is trapped forever in a two-cycle of amplitude
$\eta/\sqrt{2}$ with loss $f = 250\,\eta^2 = 10$ (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity` verifies
this formula across the step-size grid). Momentum averages out these
alternating steps, which is why the $\beta_1 = 0.9$ runs eventually reach
the target tolerance. :numref:`fig_mdl-opt-per-coordinate` illustrates all
three behaviors on a valley shallow enough to plot clearly.

![Comparison on a two-dimensional valley ($\kappa=20$; the text's runs use $\kappa=10^3$, whose contours are too thin to draw) from the same starting point. (a) Gradient descent with a single step size near the stability limit oscillates across the steep direction while crawling along the flat one. (b) Per-coordinate normalization (Adam with $\beta_1=0$) moves both coordinates at the same speed and curves smoothly into the minimum. (c) Rotating the valley and starting point by $45^\circ$ misaligns the coordinate axes from the curvature directions, leaving the iterate creeping along the valley floor ($f=0.19$ after 40 steps, versus $4\times10^{-7}$ in (b)).](../img/mdl-opt-per-coordinate.svg)
:label:`fig_mdl-opt-per-coordinate`

**Per-coordinate scaling eliminates ill-conditioning if and only if the
coordinate axes align with the eigenvectors of the curvature.** Scaling
changes the metric by a diagonal matrix, and no diagonal matrix can capture
curvature along directions that mix coordinates. A full matrix built from
outer products $\mathbf{g}_\tau\mathbf{g}_\tau^\top$ rotates with the
problem and avoids this flaw, but storing and inverting it is far too
expensive for large neural networks. The structured preconditioners at the
end of this section address this limitation for weight matrices by using
compact factors that mix coordinates.

### A Convex Counterexample to Adam
:label:`subsec_mdl-adam-counterexample`

RMSProp and Adam forget past gradients, whereas AdaGrad remembers them
indefinitely. Can that forgetting make even a simple convex problem
unsolvable? :citet:`Reddi.Kale.Kumar.2019` showed that it can, using a
one-dimensional periodic sequence of convex losses.

**Proposition (Adam need not converge).** *Let $C > 2$ and consider online
optimization over $x \in [-1, 1]$ with the periodic losses*

$$
f_t(x) =
\begin{cases}
C\,x, & t \bmod 3 = 1, \\
-x, & \textrm{otherwise}.
\end{cases}
$$
:eqlabel:`eq_mdl-opt-reddi`

*Run Adam without bias correction, with $\beta_1 = 0$,
$\beta_2 = 1/(1 + C^2)$, step sizes $\eta_t = \eta/\sqrt{t}$ satisfying
$\eta < \sqrt{1 - \beta_2}$, and projection of each iterate back onto
$[-1, 1]$, starting from $x_1 = 1$. Then $x_t > 0$ for every $t$ and
$x_{3k+1} = 1$ for every $k \ge 0$. Each three-step period adds at least
$2C - 4 > 0$ to the regret, so the average regret never tends to zero.*
:cite:`Reddi.Kale.Kumar.2019`

Here, projection onto $[-1, 1]$ is simply clipping
(:numref:`sec_mdl-constrained-optimization-duality` treats projection in
general). Because the gradients over each three-step period sum to
$C - 2 > 0$, the best fixed point in hindsight is $x^\star = -1$. Yet Adam
returns to $x = +1$—the worst point in the interval—at the start of every
period. Each period costs Adam at least $C - 2$ (it incurs $C$ on the first
step at $x = 1$, and at least $-1$ on each of the other two), whereas
$x^\star = -1$ incurs $-(C - 2)$, producing a regret increment of at least
$2C - 4$ per period. The proof proceeds by induction over periods; Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-reddi-by-hand` works through the
key step, where $\eta < \sqrt{1 - \beta_2}$ ensures that the downward step
stays inside $(0, 1]$. Nor is this failure an artifact of setting
$\beta_1 = 0$: the same paper constructs, for every fixed pair with
$\beta_1 < \sqrt{\beta_2}$, both an online convex sequence and a stochastic
convex problem on which Adam fails to converge.

We can see the mechanism directly in a single period with $C = 4$ and
$\beta_2 = 1/17$, once $v$ has settled into its steady periodic cycle.
Following the gradients $+4, -1, -1$, the second-moment estimate $v$ takes
the values $15.12$, $1.83$, and $1.05$, so $\sqrt{v}$ is $3.89$, $1.35$, and
$1.02$. The three resulting displacements $-\eta g_t/\sqrt{v_t}$ are

$$
-1.03\,\eta, \qquad +0.74\,\eta, \qquad +0.98\,\eta,
$$

yielding a net drift of $+0.69\,\eta$ per period toward $+1$. Normalization
caps the rare large gradient $+C$ to a move of roughly $\eta$—barely larger
than the move from a gradient of size $1$—and $v$ forgets the $C^2$ spike
within two steps. Together, the two $-1$ gradients therefore push $x$
farther upward than the single $+C$ gradient pulls it downward. By contrast,
plain SGD with step $\eta g_t/C$ moves by $-\eta$, $+0.25\,\eta$, and
$+0.25\,\eta$, for a net $-0.5\,\eta$ toward the optimum. Unlike the noise
floor of constant-step SGD (:numref:`sec_mdl-stochastic-gradient-methods`),
Adam's wrong-way drift does not disappear as the step size shrinks: it
points toward $+1$ at every step size.

To repair this flaw, :citet:`Reddi.Kale.Kumar.2019` proposed **AMSGrad**,
which replaces $\mathbf{v}_t$ in the denominator with its running maximum
$\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$ of the
uncorrected average. The effective step $\eta_t/\sqrt{\tilde{v}_t}$ is then
nonincreasing, so a large gradient is never forgotten. In our periodic
example, $\tilde{v}$ remains pinned at $15.12$ throughout, changing the
three displacements to $-1.03\,\eta$, $+0.26\,\eta$, and $+0.26\,\eta$—a net
$-0.51\,\eta$ toward the optimum. Adam should therefore converge to $+1$,
while AMSGrad and SGD converge to $-1$. The cell below tests all three with
the same $1/\sqrt{t}$ decay starting from $x = 0$, the midpoint of the
interval, so that Adam is not even initialized at the bad endpoint.

```{.python .input #adaptive-stochastic-methods-reddi}
def online(update, C=4.0, T=15000, alpha=0.5):
    """Projected iterates on the periodic losses; beta_1 = 0, no bias correction."""
    beta2 = 1 / (1 + C * C)
    x, v, vmax, xs = 0.0, 0.0, 0.0, []
    for t in range(1, T + 1):
        g = C if t % 3 == 1 else -1.0            # rare +C, frequent -1
        v = beta2 * v + (1 - beta2) * g * g
        vmax = max(vmax, v)                      # running maximum (AMSGrad)
        x = min(1.0, max(-1.0, x - alpha / np.sqrt(t) * update(g, v, vmax, C)))
        xs.append(x)
    return xs

updates = {'adam': lambda g, v, vmax, C: g / (np.sqrt(v) + 1e-8),
           'amsgrad': lambda g, v, vmax, C: g / (np.sqrt(vmax) + 1e-8),
           'sgd': lambda g, v, vmax, C: g / C}
print('x_t at t =        30      300     3000    15000   (x* = -1)')
for name, update in updates.items():
    xs = online(update)
    print(f'{name:9s}' + ''.join(f'{xs[t - 1]:9.3f}' for t in (30, 300, 3000, 15000)))
```

Here `alpha` plays the role of $\eta$ in the proposition, and
$\eta = 0.5 < \sqrt{16/17}$, so the step-size hypothesis holds. Even from
the midpoint of the interval, Adam still marches straight to $+1$: from
$t \approx 300$ onward it returns to $+1$ at every period and never drops
below $0.97$. Meanwhile, AMSGrad and SGD steadily approach the optimum $-1$
and agree to two decimal places. This counterexample proves that Adam lacks
a general convex convergence guarantee, though it should be read as an
existence result rather than a prediction of failure on typical deep
learning problems.

## Decoupled Weight Decay
:label:`subsec_mdl-decoupled-weight-decay`

### Coupled and Decoupled Decay

Almost every modern training recipe includes weight decay, which forces an
adaptive optimizer to make a design choice: should the decay term pass
through the preconditioner? Under SGD, the distinction does not exist.
Adding an $\ell_2$ penalty $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss
adds $\lambda \mathbf{w}$ to the gradient, and the resulting update
$\mathbf{w}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{w}_t) = (1 - \eta\lambda)\, \mathbf{w}_t - \eta\, \mathbf{g}_t$
is simultaneously a gradient step on the loss and a uniform shrinkage of the
weights (:eqref:`eq_l2-is-decay` in :numref:`sec_adamw`, which writes
$\mathbf{x}$ for the weights). Inside Adam, however, those two mechanisms
diverge. Feeding the penalized gradient into :eqref:`eq_mdl-opt-adam` and
isolating the decay term (setting $\beta_1 = 0$ for clarity, so the update
is $-\eta\,(\mathbf{g}_t + \lambda\mathbf{w}_t)/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$),
the per-step shrinkage of coordinate $i$ becomes

$$
\textrm{coupled ($\ell_2$ through Adam):}\quad
\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; w_{t,i}
\qquad \textrm{versus} \qquad
\textrm{decoupled:}\quad \eta\,\lambda\, w_{t,i} .
$$
:eqlabel:`eq_mdl-opt-adamw`

In coupled Adam, the preconditioner divides both the loss gradient and the
penalty gradient, so a coordinate's shrinkage rate is inversely proportional
to its gradient scale: weights with large gradients are barely decayed,
while quiet weights are decayed heavily. **AdamW**
:cite:`Loshchilov.Hutter.2019` decouples the two operations so that only the
loss gradient passes through the preconditioner while weight decay acts
directly on the parameters:

$$
\mathbf{w}_{t+1} = (1 - \eta\lambda)\, \mathbf{w}_t - \eta\; \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_mdl-opt-adamw-decoupled`

Every coordinate now shrinks by the same factor $1 - \eta\lambda$ per step,
just as in SGD, and the moment averages track only the loss gradient.

:begin_tab:`pytorch`
`torch.optim.Adam(weight_decay=...)` adds $\lambda\mathbf{w}$ to the gradient:
it implements the coupled rule. `torch.optim.AdamW` implements the decoupled
rule :eqref:`eq_mdl-opt-adamw-decoupled`.
:end_tab:

:begin_tab:`mxnet`
`mx.optimizer.Adam(wd=...)` adds $\lambda\mathbf{w}$ to the gradient: it
implements the coupled rule. `mx.optimizer.AdamW` applies weight decay
outside the preconditioner.
:end_tab:

:begin_tab:`tensorflow`
In Keras, `weight_decay` is an argument of every optimizer and is applied to
the weights outside the preconditioner, so
`keras.optimizers.Adam(weight_decay=...)` and `keras.optimizers.AdamW` both
implement the decoupled rule. The coupled rule requires adding an explicit
$\ell_2$ penalty to the loss.
:end_tab:

:begin_tab:`jax`
`optax.adam` has no weight-decay argument; `optax.adamw` implements the
decoupled rule. The coupled rule requires an explicit penalty in the loss,
or `optax.chain(optax.add_decayed_weights(lam), optax.adam(eta))`, which
adds $\lambda\mathbf{w}$ to the gradient before Adam sees it.
:end_tab:

The sharpest test of :eqref:`eq_mdl-opt-adamw` is a weight whose loss
gradient is pure zero-mean noise, leaving weight decay as the only
systematic drift. If the noise has standard deviation $\sigma$, then
$\sqrt{\hat{v}} \approx \sigma$, so we expect coupled Adam to shrink the
weight to roughly $e^{-\eta\lambda T/\sigma}$ after $T$ steps, whereas AdamW
should shrink every weight to $(1 - \eta\lambda)^T$ regardless of $\sigma$.
Below, we initialize $10^4$ weights with noise scale $\sigma = 10$ and
$10^4$ with $\sigma = 0.1$, all at $w = 1$, run 4,000 steps of each method
with $\eta = 10^{-3}$ and $\lambda = 0.1$, and compare the group means with
these predictions.

```{.python .input #adaptive-stochastic-methods-adamw}
sigma, eta, lam, T, n = np.array([[10.0], [0.1]]), 1e-3, 0.1, 4000, 10_000
rng = np.random.default_rng()

def decay(decoupled, beta1=0.9, beta2=0.999, eps=1e-8):
    """Mean of w over n weights per noise scale after T Adam steps from w = 1."""
    w, m, v = np.ones((2, n)), np.zeros((2, n)), np.zeros((2, n))
    for t in range(1, T + 1):
        g = sigma * rng.standard_normal((2, n))     # loss gradient: pure noise
        if decoupled:
            w = (1 - eta * lam) * w                 # decay outside the preconditioner
        else:
            g = g + lam * w                         # l2 penalty inside the gradient
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        w = w - eta * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    return w.mean(axis=1)

print('per-step decay rates: coupled', eta * lam / sigma.ravel(), ' AdamW', eta * lam)
print('mean w, coupled:', decay(False).round(3),
      '  predictions exp(-eta lam T / sigma):', np.exp(-eta * lam * T / sigma.ravel()).round(3))
print('mean w, AdamW:  ', decay(True).round(3),
      '  prediction (1 - eta lam)^T:', round((1 - eta * lam)**T, 3))
```

A single value of $\lambda$ gives the two groups coupled decay rates of
$10^{-5}$ and $10^{-3}$ per step—a hundredfold difference. The noisy weights
retain $0.96$ of their initial magnitude as predicted, while the quiet
weights drop to about $0.03$, slightly above the $0.018$ estimate. That
small shortfall has a straightforward cause: near $w = 1$, the penalty
gradient $\lambda w = 0.1$ on a quiet weight is comparable to its noise and
inflates its own $\hat{v}$, slowing its initial decay. Under AdamW, by
contrast, both groups finish right at $0.67 = (1 - \eta\lambda)^{4000}$. For
AdamW this prediction is exact in expectation, because the normalized noise
steps have mean zero and the multiplicative decay is identical at every
step.

### Fixed Points

The pure-noise experiment has no stationary point because its gradients
never settle. With exact gradients, however, an optimizer can come to rest,
and at a fixed point $\lambda$ takes on a surprisingly different meaning.

**Proposition (fixed points of coupled and decoupled decay).** *Let the
gradient $\mathbf{g}(\mathbf{w}) = \nabla\ell(\mathbf{w})$ of a loss $\ell$
be exact (noiseless), let $\epsilon > 0$, and call $\mathbf{w}$ a fixed
point of an update if the update vanishes there once the moment estimates
have settled. (a) Adam on $\ell + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$ has a
fixed point at $\mathbf{w}$ if and only if
$\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$, the same
stationary point as SGD on the penalized loss. (b) AdamW has a fixed point
at $\mathbf{w}$ if and only if, for every coordinate $i$,*

$$
\lambda\, w_i \bigl(|g_i| + \epsilon\bigr) = -g_i ,
\qquad g_i = \partial_i \ell(\mathbf{w}).
$$
:eqlabel:`eq_mdl-opt-adamw-fixed-point`

*Every fixed point of AdamW therefore lies strictly inside the box
$|w_i| < 1/\lambda$, and as $\epsilon \to 0$ the condition reduces to a
dichotomy: either $g_i = 0$, or $|w_i| = 1/\lambda$ with $w_i$ opposite in
sign to $g_i$.*

**Proof.** At a fixed point, the vector $\mathbf{u}$ fed into the moment
averages is constant, so $\hat{\mathbf{m}}_t \to \mathbf{u}$ and
$\hat{\mathbf{v}}_t \to \mathbf{u}^2$. (a) For coupled Adam,
$\mathbf{u} = \mathbf{g} + \lambda\mathbf{w}$, and coordinate $i$'s update
approaches $-\eta\, u_i/(|u_i| + \epsilon)$, which vanishes if and only if
$u_i = 0$. (b) For AdamW, $\mathbf{u} = \mathbf{g}$, so coordinate $i$'s
update approaches $-\eta\lambda w_i - \eta\, g_i/(|g_i| + \epsilon)$;
setting this to zero and multiplying through by $(|g_i| + \epsilon)/\eta$
yields :eqref:`eq_mdl-opt-adamw-fixed-point`. Taking absolute values gives
$|w_i| = |g_i|/\bigl(\lambda(|g_i| + \epsilon)\bigr) < 1/\lambda$. Letting
$\epsilon \to 0$ turns the condition into $\lambda w_i |g_i| = -g_i$, which
holds precisely when $g_i = 0$ or $\lambda w_i = -\mathrm{sign}(g_i)$.
$\blacksquare$

This proposition characterizes fixed points; it does not assert that one
exists or that the iterates converge to it. Nor is the dichotomy merely a
limiting curiosity: since $1 - \lambda|w_i| = \epsilon/(|g_i| + \epsilon)$,
at $\epsilon = 10^{-8}$ every coordinate either sits within a fraction
$\epsilon/|g_i|$ of the box boundary or has a gradient of order $\epsilon$.

A one-dimensional example makes the contrast concrete. Consider
$\ell(w) = \tfrac12 (w - 3)^2$, whose unconstrained minimizer is $w = 3$.
With $\lambda = \tfrac14$, coupled Adam rests at the penalized minimizer
$3/(1 + \lambda) = 2.4$, whereas AdamW rests at $3.0$: the box $|w| \le 4$
does not bind, so decoupled decay leaves the minimizer untouched. With
$\lambda = 1$, coupled Adam rests at $1.5$, while AdamW is pinned at $1.0$,
right on the boundary of the box.

The two interpretations of $\lambda$ are thus completely distinct. In the
noise-dominated regime of training, AdamW applies uniform per-step shrinkage
across all coordinates while coupled Adam distorts it by gradient scale. At
exact-gradient fixed points, the roles flip: coupled Adam preserves the
maximum a posteriori (MAP) solution of a Gaussian prior on the weights
(:numref:`sec_mdl-maximum_likelihood`), whereas AdamW turns $\lambda$ into
the radius $1/\lambda$ of an $\ell_\infty$ constraint box
:cite:`Xie.Li.2024`. Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-adamw-linf` connects this box
directly to constrained optimization.

## Warmup
:label:`subsec_mdl-schedules-warmup`

At step $t = 1$, Adam's bias-corrected moments are
$\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and
$\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the update is simply
$\eta\,\mathrm{sign}(\mathbf{g}_1)$—a step of full size $\eta$ in every
coordinate. A coordinate whose gradient is $10^{-6}$ moves just as far as
one whose gradient is $10^3$, on the evidence of a single minibatch.
**Warmup** ramps the step size up from near zero over the opening steps of
training :cite:`Goyal.Dollar.Girshick.ea.2017`, keeping these early updates
small. It is tempting to blame this early instability on the preconditioner
$\hat{\mathbf{v}}_t$, which at $t = 1$ rests on a single sample. A simple
experiment shows that something else is at work.

Is a step of size $\eta$ actually too large? Not for a coordinate with a
consistent gradient, whose steady-state step is also $\eta$. For a
coordinate whose gradient is pure noise, however, it is wasted motion. To
measure how large that wasted motion is, we draw $2 \times 10^4$ independent
gradient coordinates from $\mathcal{N}(0, 1)$ at each step, run default
Adam, and track the root-mean-square update in units of $\eta$. We compare
four curves: standard Adam; Adam with $\hat{\mathbf{v}}_t$ replaced by the
exact variance $\sigma^2 = 1$ (eliminating preconditioner estimation error
entirely); Adam with both bias corrections removed; and Adam multiplied by a
linear warmup over the first $20$ steps.

```{.python .input #mdl-adaptive-stochastic-methods-warmup}
n, T, beta1, beta2, eps, T_w = 20_000, 3000, 0.9, 0.999, 1e-8, 20
m, v = np.zeros(n), np.zeros(n)
rms = {'Adam': [], 'exact variance': [], 'no bias correction': [],
       f'warmup over {T_w} steps': []}
for t in range(1, T + 1):
    g = rng.standard_normal(n)                      # pure noise, sigma = 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat, v_hat = m / (1 - beta1**t), v / (1 - beta2**t)
    step = m_hat / (np.sqrt(v_hat) + eps)          # Adam's update / eta
    for r, u in zip(rms.values(), (step, m_hat, m / (np.sqrt(v) + eps),
                                   min(1.0, t / T_w) * step)):
        r.append(np.sqrt(np.mean(u**2)))
d2l.plot(np.arange(1, T + 1), list(rms.values()), 't', 'RMS update / eta',
         legend=list(rms), xscale='log')
ts = (1, 2, 3, 10, 20, 30, 100, 1000, 3000)
print(f'{"t:":24s}' + ''.join(f'{t:>6d}' for t in ts))
for name, r in rms.items():
    print(f'{name + ":":24s}' + ''.join(f'{r[t - 1]:6.2f}' for t in ts))
```

Standard Adam starts with an RMS update of $1.00\,\eta$, drops to
$0.33\,\eta$ by $t = 10$, and settles within about fifty steps at
$\sqrt{(1 - \beta_1)/(1 + \beta_1)}\,\eta = 0.23\,\eta$—the steady-state
size of an exponential moving average of independent unit noise. On a pure
noise coordinate, Adam's opening steps are therefore roughly four times
larger than their steady-state size, and this transient lasts about ten
steps.

Is the noisy preconditioner to blame? The second curve says no. Replacing
$\hat{\mathbf{v}}_t$ with the exact variance leaves the RMS update unchanged
to within $0.01$ at every step (the sampling noise of $2 \times 10^4$
coordinates). The fourfold inflation comes from the *momentum* average,
which needs roughly $1/(1 - \beta_1) = 10$ steps to average out independent
noise. What the preconditioner actually does at $t = 1$ is equalize
coordinates of different scales, an effect hidden when every coordinate
shares $\sigma = 1$.

Bias correction matters far more. Without it, the opening update jumps to
$3.16\,\eta$—matching the $(1 - \beta_1)/\sqrt{1 - \beta_2}$ factor in
:numref:`fig_mdl-opt-bias-correction`—and the transient stretches from ten
steps to hundreds: the update is still around $0.75\,\eta$ at $t = 100$ and
$0.29\,\eta$ at $t = 1000$, because the uncorrected $\mathbf{v}_t$ fills on
the slow $1/(1 - \beta_2) = 1000$ timescale. An uncorrected optimizer would
need a thousand-step warmup merely to undo its own zero initialization. With
bias correction in place, a linear ramp over just $20$ steps tames what
remains, keeping the warmup curve below $0.26\,\eta$ throughout.

That accounts for warmups of tens of steps, not the thousands of steps often
used in large-scale training. Longer warmups respond to a second mechanism
absent from this noise model: curvature. A step size is stable only below
the ceiling $2/\lambda_{\max}$ imposed by the steepest curvature direction
(:numref:`subsec_mdl-quadratic-model`). Immediately after random
initialization, that curvature can be unusually large and shift rapidly, so
a step size that is safe later in training may initially be unstable.
:numref:`subsec_scheduler-warmup` revisits both mechanisms on a real neural
network and measures the opening step sizes directly.

## Structured Preconditioners
:label:`subsec_mdl-preconditioning-ladder`

### Kronecker Factors

Our rotated valley defeated every diagonal preconditioner. All of the
methods in this section share the preconditioned form
$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{P}_t^{-1} \mathbf{g}_t$
for a positive definite matrix $\mathbf{P}_t$: gradient descent uses
$\mathbf{P} = \mathbf{I}$, Newton's method uses the Hessian $\nabla^2 f$,
and Adam uses a diagonal matrix estimated from past gradients. A full
curvature matrix captures steepness along any direction—aligned with the
axes or not—but it is far too large to store. A single $4096 \times 4096$
weight matrix already has $1.7 \times 10^7$ parameters; its full curvature
matrix would contain $2.8 \times 10^{14}$ entries, requiring roughly two
petabytes in double precision.

What makes a middle ground possible is that neural network parameters are
organized into matrices. Consider a linear layer
$\mathbf{y} = \mathbf{W}\mathbf{a}$ with weight matrix
$\mathbf{W} \in \mathbb{R}^{m \times n}$, input $\mathbf{a} \in \mathbb{R}^n$,
and output gradient
$\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y} \in \mathbb{R}^m$.
The gradient with respect to $\mathbf{W}$ is the rank-one outer product
$\boldsymbol{\delta}\mathbf{a}^\top$. Writing $\mathrm{vec}(\mathbf{V})$
for the vector obtained by stacking the columns of a matrix $\mathbf{V}$, we
have
$\mathrm{vec}(\boldsymbol{\delta}\mathbf{a}^\top) = \mathbf{a} \otimes \boldsymbol{\delta}$,
where the **Kronecker product** $\mathbf{B} \otimes \mathbf{D}$ is the block
matrix whose $(i, j)$ block is $b_{ij}\mathbf{D}$.

The outer-product curvature of this layer—the block of the Fisher
information matrix $\mathbb{E}[\nabla\log p\,\nabla\log p^\top]$ (which we will
study in :numref:`sec_mdl-maximum_likelihood`) corresponding to $\mathbf{W}$—is
therefore
$\mathbb{E}\bigl[(\mathbf{a}\mathbf{a}^\top) \otimes (\boldsymbol{\delta}\boldsymbol{\delta}^\top)\bigr]$.
**K-FAC** :cite:`Martens.Grosse.2015` approximates this expectation by the
Kronecker product $\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}$,
where $\mathbf{S}_{\mathrm{in}} = \mathbb{E}[\mathbf{a}\mathbf{a}^\top]$
is $n \times n$ and
$\mathbf{S}_{\mathrm{out}} = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top]$
is $m \times m$. This factorization would be exact if inputs and output
gradients were statistically independent; nobody expects that to hold in a
trained network, but the approximation is adopted because it can be inverted
cheaply.

Specifically, the inverse of a Kronecker product acts directly on the
unstacked gradient matrix. For symmetric invertible factors,

$$
\bigl(\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}\bigr)^{-1} \mathrm{vec}(\mathbf{V})
= \mathrm{vec}\bigl(\mathbf{S}_{\mathrm{out}}^{-1}\, \mathbf{V}\, \mathbf{S}_{\mathrm{in}}^{-1}\bigr).
$$
:eqlabel:`eq_mdl-opt-kronecker`

To verify this identity, write column $j$ of the product
$\mathbf{D}\mathbf{V}\mathbf{B}$ as $\sum_k b_{kj}\, \mathbf{D}\mathbf{v}_k$,
where $\mathbf{v}_k$ is column $k$ of $\mathbf{V}$. Stacking these columns
produces the block product $(\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$,
so $\mathrm{vec}(\mathbf{D}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$.
Using $(\mathbf{B} \otimes \mathbf{D})^{-1} = \mathbf{B}^{-1} \otimes \mathbf{D}^{-1}$
together with the symmetry of the factors, setting
$\mathbf{B} = \mathbf{S}_{\mathrm{in}}^{-1}$ and
$\mathbf{D} = \mathbf{S}_{\mathrm{out}}^{-1}$ yields
:eqref:`eq_mdl-opt-kronecker`. In this way, a curvature matrix with
$(mn)^2$ entries is inverted and applied through two modest inverses of
sizes $n$ and $m$. The cell below compares the storage and operation counts
for our $4096 \times 4096$ layer.

```{.python .input #mdl-adaptive-stochastic-methods-kronecker-factors}
m = n = 4096
print(f'stored entries: full {(m * n)**2:.1e}, two factors {m**2 + n**2:.1e}')
print(f'flops to invert: full {(m * n)**3:.1e}, two factors {m**3 + n**3:.1e}')
```

Factoring cuts storage from $2.8 \times 10^{14}$ entries down to
$3.4 \times 10^7$, and cuts the inversion cost from $4.7 \times 10^{21}$
operations to roughly $1.4 \times 10^{11}$.

### K-FAC, Shampoo and Muon

The Kronecker identity leaves two choices open: which matrices to factor,
and what exponent to apply to them. Modern structured optimizers share the
two-sided preconditioning form
$\mathbf{G} \mapsto \mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\,\mathbf{P}_{\mathrm{r}}^{-a}$
applied to a layer's gradient matrix $\mathbf{G}$ (bold, distinguishing it
from the scalar gradient bound $G$ of the regret analysis). Together, they
form a natural progression from Adam's diagonal scaling to matrix-level
normalization.

K-FAC constructs its factors from the second moments of layer inputs and
output gradients with exponent $a = 1$. It approximates preconditioning by
the Fisher information matrix—the **natural gradient** :cite:`Amari.1998`,
which is the metric view of AdaGrad using the Fisher matrix as the metric.
For softmax cross-entropy and squared loss, the Fisher matrix (with labels
sampled from the model) coincides with the Gauss–Newton matrix, a positive
semidefinite approximation to the Hessian that drops the term involving
second derivatives of the network. For these losses, K-FAC therefore
approximates genuine curvature.

**Shampoo** :cite:`Gupta.Koren.Singer.2018` instead builds its left and
right factors directly from accumulated gradient matrices in the spirit of
AdaGrad, setting
$\mathbf{P}_{\mathrm{l}} = \sum_t \mathbf{G}_t\mathbf{G}_t^\top$ and
$\mathbf{P}_{\mathrm{r}} = \sum_t \mathbf{G}_t^\top\mathbf{G}_t$ with
$a = \tfrac14$. **Muon** :cite:`Jordan.Jin.Boza.ea.2024` can be viewed as
memoryless Shampoo: using only the current matrix
$\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (assumed square
and invertible),

$$
\bigl(\mathbf{G}\mathbf{G}^\top\bigr)^{-1/4}\, \mathbf{G}\, \bigl(\mathbf{G}^\top\mathbf{G}\bigr)^{-1/4}
= \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\, \boldsymbol{\Sigma}\, \boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top
= \mathbf{U}\mathbf{V}^\top ,
$$

which is the polar factor of $\mathbf{G}$. It preserves the singular vectors
of $\mathbf{G}$ while setting every singular value to one. Where Adam
equalizes the coordinates of the update, Muon equalizes its singular
directions—and unlike per-coordinate normalization, this map commutes with
orthogonal rotations of the rows and columns (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d). Muon as
a practical optimizer, which applies this orthogonalization to the momentum
matrix, is covered in the main text (:numref:`sec_muon`), while the
Newton–Schulz iteration that computes the polar factor using matrix
multiplications alone is analyzed in
:numref:`subsec_mdl-orthogonalized-updates`.

None of these three methods reproduces full Newton's method. Muon stores no
second-moment matrices and requires only a handful of matrix products per
step, whereas K-FAC and Shampoo invert or take matrix roots of their two
factors, typically amortizing that cost over several steps. Moreover, every
positive result in this section assumes that curvature aligns either with
the coordinate axes or with the Kronecker structure of individual layers.
Because the true curvature of a deep neural network is neither diagonal nor
fixed, explaining why these adaptive methods perform as well as they do in
practice remains an active open question.

## Summary

Dividing each coordinate's update by the root mean square of its past
gradients—the common thread linking AdaGrad, RMSProp, and Adam—removes the
condition number whenever curvature aligns with the coordinate axes. With
$\epsilon = 0$, Adam's iterates are strictly invariant to per-coordinate
rescaling, allowing scaling alone to beat an optimally tuned heavy-ball
method by an order of magnitude on an aligned valley. Rotate the valley by
$45^\circ$, however, and that advantage vanishes: scaling alone never
converges, and Adam with momentum takes roughly twice as many steps as heavy
ball.

Forgetting past gradients introduces further subtleties. On a periodic
sequence of convex losses, Adam's short second-moment memory can drive it to
the worst point of the domain, a failure fixed by AMSGrad's running maximum.
With weight decay, passing an $\ell_2$ penalty through Adam scales each
coordinate's shrinkage inversely with its gradient size while preserving the
MAP fixed point; AdamW decouples the decay so that every coordinate shrinks
uniformly per step, turning $\lambda$ into the radius $1/\lambda$ of an
$\ell_\infty$ box at exact-gradient fixed points.

Finally, Adam's opening updates on pure-noise coordinates are roughly four
times larger than their steady-state size because of the momentum average
rather than the preconditioner, and dropping bias correction inflates them
much further for hundreds of steps; longer practical warmups address early
curvature instead. Beyond diagonal scaling, Kronecker-factored
preconditioners such as K-FAC, Shampoo, and Muon capture cross-coordinate
structure within weight matrices using two compact factors, with Muon
equalizing the singular directions of a layer's update.

## Exercises

1. **AdaGrad from the metric view.** Exercise
   :exref:`ex_mdl-gradient-based-optimization-show-that-is-de`b of
   :numref:`sec_mdl-gradient-based-optimization` showed that the steepest
   descent direction under the norm
   $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P} \mathbf{d}}$
   is proportional to $-\mathbf{P}^{-1}\nabla f$.
    1. Set $\mathbf{P}_t = \mathrm{diag}\bigl(\sqrt{\mathbf{s}_t}\bigr) + \epsilon \mathbf{I}$
       with $\mathbf{s}_t = \sum_{\tau \le t} \mathbf{g}_\tau^2$ and
       recover :eqref:`eq_mdl-opt-adagrad`.
    1. Replace the objective $f$ by $cf$ for a constant $c > 0$ and set
       $\epsilon = 0$. How does the AdaGrad update change? How would it
       change if $\mathbf{P}_t$ were built from $\mathbf{s}_t$ instead of
       $\sqrt{\mathbf{s}_t}$? Compare both with gradient descent and with
       Newton's method.
    1. Let the gradients of one coordinate be i.i.d. with
       $\mathbb{E}[g_{t,i}^2] = \sigma^2$. How fast does the effective step
       of that coordinate decay in $t$ under each of the two choices, and how
       does it depend on $\sigma$?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`

1. **Concentrated gradients and AdaGrad's bound.** Up to constant factors,
   projected online gradient descent with a tuned global step has regret
   $D_2 G \sqrt{T}$ on a convex set of Euclidean diameter $D_2$ when
   $\|\mathbf{g}_t\| \le G$
   (Exercise :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`).
   Diagonal AdaGrad has regret
   $D_\infty \sum_{i=1}^{d} \sqrt{\sum_{t=1}^{T} g_{t,i}^2}$, where
   $D_\infty$ is the diameter of the set in the $\ell_\infty$ norm
   :cite:`Duchi.Hazan.Singer.2011`. Take the box $[-1, 1]^d$ and one-hot
   gradients: at each step one coordinate $i$, drawn independently with
   probability $p_i$, receives the gradient $+1$ or $-1$, and all other
   coordinates receive $0$.
    1. Compute both bounds. For AdaGrad, bound the expectation of each square
       root with Jensen's inequality, and show that the ratio of the AdaGrad
       bound to the gradient-descent bound is at most $\sum_i \sqrt{p_i}/\sqrt{d}$.
    1. Evaluate the ratio for uniform $p_i$ and for Zipf frequencies
       $p_i \propto i^{-\alpha}$ with $d = 10^4$ and $\alpha \in \{1, 2, 3\}$.
    1. Word frequencies in natural text follow Zipf's law with
       $\alpha \approx 1$, and your ratio there is about $0.63$. How much of
       the factor $\sqrt{d}$ of :numref:`subsec_mdl-per-coordinate` does a
       bag-of-words model obtain? Which distribution of gradient mass would
       obtain all of it?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`

1. **The Reddi example by hand.** Take $C = 4$, $\beta_1 = 0$, the
   proposition's $\beta_2 = 1/(1 + C^2) = 1/17$, and no bias correction, as
   in `online`.
    1. Assume a fixed step $\eta$, an iterate in the interior of $[-1, 1]$,
       and $v_t$ settled into its periodic pattern. Write $v_a$ for the value
       of $v$ after the $+C$ step, solve the three-step recursion for $v_a$
       in closed form, and derive the periodic values $15.12$, $1.83$, $1.05$
       and the three displacements of the text. Compute SGD's three steps
       $\eta g_t/C$. Which steps does Adam enlarge relative to SGD, and why?
    1. AMSGrad replaces $v_t$ by
       $\tilde{v}_t = \max(\tilde{v}_{t-1}, v_t)$. Show that its effective
       step $\eta_t/\sqrt{\tilde{v}_t}$ is nonincreasing in $t$ whenever
       $\eta_t$ is. Compute its three displacements in the periodic regime
       and show that the net displacement is $-0.51\,\eta$.
    1. The proposition needs $\eta < \sqrt{1 - \beta_2}$ for the steps
       $\eta_t = \eta/\sqrt{t}$. Show where this is used: starting a period
       at $x = 1$, use $v \ge (1-\beta_2)C^2$ after the $+C$ gradient to show
       that the first step of the period lands in $(0, 1)$, so that the
       projection does not act on it.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-reddi-by-hand`

1. **Regret of online gradient descent.** ● Projected online gradient
   descent plays
   $\mathbf{x}_{t+1} = \Pi_{\mathcal{C}}(\mathbf{x}_t - \eta_t \mathbf{g}_t)$
   with $\mathbf{g}_t = \nabla f_t(\mathbf{x}_t)$, where $\Pi_{\mathcal{C}}$
   is the Euclidean projection onto a closed convex set $\mathcal{C}$ of
   diameter $D$, $\|\mathbf{g}_t\| \le G$, and the steps $\eta_t$ are
   nonincreasing :cite:`Zinkevich.2003`.
    1. Prove that the regret satisfies

       $$\sum_{t=1}^{T} f_t(\mathbf{x}_t) - \min_{\mathbf{x} \in \mathcal{C}} \sum_{t=1}^{T} f_t(\mathbf{x}) \;\le\; \frac{D^2}{2\eta_T} + \frac{G^2}{2} \sum_{t=1}^{T} \eta_t.$$

       Start from $f_t(\mathbf{x}_t) - f_t(\mathbf{x}) \le \mathbf{g}_t^\top(\mathbf{x}_t - \mathbf{x})$
       and use that the projection does not increase distances to points
       of $\mathcal{C}$.
    1. Show that $\eta_t = D/(G\sqrt{t})$ gives regret at most
       $\tfrac32 DG\sqrt{T}$, and that for $\eta_t = c/t$ the bound grows
       linearly in $T$ for every $c > 0$.
    1. Combine these two bounds with the decay rates of Exercise
       :exref:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`c to
       explain why AdaGrad divides by $\sqrt{\mathbf{s}_t}$ rather than
       by $\mathbf{s}_t$.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-regret-ogd`

1. [code] **Rotation sensitivity.** ● Use `rotated`, `run` and the update
   factories of the `#adaptive-stochastic-methods-adam-vs-gd` cell, where the
   Hessian $\mathbf{A}_\theta$ has eigenvalues $\lambda_{\min} = 1$ and
   $\lambda_{\max} = 1000$ and condition number
   $\kappa = \lambda_{\max}/\lambda_{\min} = 1000$.
    1. Show that the iterates of gradient descent and of heavy ball on
       $\mathbf{A}_\theta$ from $\mathbf{R}_\theta(1, 1)^\top$ are
       $\mathbf{R}_\theta$ times their iterates on the aligned valley, so
       that their step counts do not depend on $\theta$.
    1. For $\beta_1 = 0$ at $\theta = \pi/4$, print $f$ after $40{,}000$
       steps at each $\eta$ of the table's grid and compare it with the
       value $250\,\eta^2$ of the text. Which component of Adam supplies
       what survives of its advantage over gradient descent?
    1. Print Adam's effective per-coordinate steps at $\theta = \pi/16$.
       Why does even this small rotation remove most of the step-size ratio
       that Adam builds on the aligned valley?
    1. Exercise
       :exref:`ex_mdl-gradient-based-optimization-steepest-other-norms` of
       :numref:`sec_mdl-gradient-based-optimization` identifies
       $-\mathrm{sign}(\mathbf{g})$ as the steepest direction under the
       $\ell_\infty$ norm and $-\mathbf{U}\mathbf{V}^\top$ as the steepest
       direction under the spectral norm, where
       $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. Show that
       Adam with $\beta_1 = \beta_2 = 0$ and $\epsilon = 0$ steps along
       $-\mathrm{sign}(\mathbf{g}_t)$. Then let $\mathbf{Q}_1$ and
       $\mathbf{Q}_2$ be orthogonal, show that the map
       $\mathbf{G} \mapsto \mathbf{U}\mathbf{V}^\top$ commutes with
       $\mathbf{G} \mapsto \mathbf{Q}_1 \mathbf{G} \mathbf{Q}_2$, and give a
       $2 \times 2$ example in which the sign map does not. What does this
       suggest about the rotation sensitivity of Muon compared with Adam?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`

1. [extended] **AdamW's implicit $\ell_\infty$ constraint.** ●● The
   proposition of :numref:`subsec_mdl-decoupled-weight-decay` found that the
   fixed points of AdamW lie in the box $|w_i| < 1/\lambda$ and, as
   $\epsilon \to 0$, on its boundary wherever the gradient does not vanish.
   :citet:`Xie.Li.2024` prove that if full-batch AdamW with
   $\beta_1 \le \beta_2$ and a nonincreasing step schedule with divergent
   sum converges, its limit is a KKT point of
   $\min_{\mathbf{w}} \ell(\mathbf{w})$ subject to
   $\|\mathbf{w}\|_\infty \le 1/\lambda$.
    1. A feasible $\mathbf{w}$ is a KKT point of this problem
       (:numref:`sec_mdl-constrained-optimization-duality` derives the
       conditions in general) when every
       coordinate satisfies $\partial_i \ell(\mathbf{w}) = 0$ if
       $|w_i| < 1/\lambda$ and $w_i\, \partial_i \ell(\mathbf{w}) \le 0$ if
       $|w_i| = 1/\lambda$. Show that at such a
       point no feasible direction decreases $\ell$ to first order.
    1. Set $\epsilon = 0$, the setting of the theorem, in the AdamW
       condition :eqref:`eq_mdl-opt-adamw-fixed-point`. Show that a coordinate with
       $\partial_i \ell \ne 0$ satisfies the KKT conditions. Why does the
       condition alone not show $|w_i| \le 1/\lambda$ for a coordinate whose
       gradient vanishes?
    1. Generate linearly separable logistic-regression data with $n = 100$,
       $d = 5$, standard normal features, and labels
       $y_i = \mathrm{sign}(\mathbf{x}_i^\top \mathbf{w}_{\mathrm{true}})$,
       so that the unregularized loss has no finite minimizer. Run
       full-batch AdamW with $\epsilon = 10^{-16}$ for $5 \times 10^4$
       steps, with $\eta$ decaying from $10^{-2}$ to $0$ by a cosine
       schedule, for $\lambda \in \{0.05, 0.1, 0.2\}$. Compare the result
       with the solution of the box-constrained problem computed by
       projected gradient descent, which here clips every coordinate to
       $[-1/\lambda, 1/\lambda]$ after each gradient step, and with Adam on
       $\ell + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$.
    1. Repeat the AdamW runs with minibatch gradients of size $10$ and $50$.
       Where does the iterate end up relative to the box, and why?
    1. What does $\lambda$ control in AdamW according to these results, and
       how does that differ from the Gaussian-prior reading of $\lambda$ in
       :numref:`sec_mdl-maximum_likelihood`?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-adamw-linf`

1. **The Kronecker identity and its cost.** Use column stacking for
   $\mathrm{vec}$.
    1. Prove $\mathrm{vec}(\mathbf{D}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$
       by writing out column $j$ of the product, and prove
       $(\mathbf{B} \otimes \mathbf{D})(\mathbf{B}' \otimes \mathbf{D}') = (\mathbf{B}\mathbf{B}') \otimes (\mathbf{D}\mathbf{D}')$.
    1. Deduce :eqref:`eq_mdl-opt-kronecker` for symmetric invertible
       factors. Which formula replaces it for factors that are not
       symmetric, and which for row stacking?
    1. For an $m \times n$ layer, count the operations needed to invert the
       full $mn \times mn$ matrix and apply it, against inverting the two
       factors and applying them by :eqref:`eq_mdl-opt-kronecker`
       ($O(m^3 + n^3 + mn(m + n))$). Evaluate both counts at $m = n = 4096$
       and at $m = 4096$, $n = 16384$.
    1. For a single gradient $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$,
       set $\mathbf{P}_{\mathrm{l}} = \mathbf{G}\mathbf{G}^\top$ and
       $\mathbf{P}_{\mathrm{r}} = \mathbf{G}^\top\mathbf{G}$. Show that the
       Shampoo step $\mathbf{P}_{\mathrm{l}}^{-1/4}\mathbf{G}\,\mathbf{P}_{\mathrm{r}}^{-1/4}$
       equals $\mathbf{U}\mathbf{V}^\top$. What must the inverse fourth root
       mean when $\mathbf{G}$ is not square, and which singular vectors does
       the result keep?

[Discussions](https://d2l.discourse.group/t/36829)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §35.5]{.kicker}

Adaptive Methods<br>**AdaGrad · RMSProp · Adam & bias correction · scale invariance · rotation sensitivity · AdamW · warmup · K-FAC & Muon**
:::
:::

::: {.slide title="Why One Global Learning Rate Bottlenecks Networks"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Modern neural networks have $10^7$ to $10^{11}$ parameters whose curvatures differ by orders of magnitude across layers and coordinates. On $f(\mathbf{x}) = \tfrac12(x_1^2 + 1000\,x_2^2)$ from $(1, 1)$ to $f(\mathbf{x}) < 10^{-8}$:

- **Optimal single-step GD ($\eta = 2/1001$):** $6{,}160$ steps.
- **Tuned Heavy-Ball momentum:** $296$ steps.
- **Per-coordinate RMS scaling (aligned):** **$22$ steps**—and increasing $\kappa$ from $10^3$ to $10^5$ adds only $6$ steps!
- **Rotated by $45^\circ$:** Scaling alone **never** reaches $10^{-8}$.
:::

::: {.col .fig}
@fig:mdl-opt-per-coordinate
:::
:::
:::

::: {.slide title="Ideal Diagonal Preconditioning vs. Gradient Proxies"}
[Per-Coordinate Scaling]{.kicker}

::: {.cols .vc}
::: {.col}
On a diagonal bowl $f(\mathbf{x}) = \tfrac12\sum_i h_i x_i^2$, each coordinate updates via $x_{t+1,i} = (1 - \eta_i h_i)x_{t,i}$:

- **Diagonal Newton ideal:** Choosing $\eta_i = 1/h_i$ reaches zero in 1 step and eliminates $\kappa = h_{\max}/h_{\min}$.
- **Unknown & noisy curvatures:** In deep learning, $h_i$ is unknown and masked by minibatch noise.
- **Gradient RMS proxy:** Because $|g_i| = h_i|x_i|$, dividing each step by a running RMS of past gradients $\sqrt{v_{t,i}}$ automatically assigns smaller steps to steeper axes.
:::

::: {.col .fig .big}
@fig:ch35-opt-diagonal-precond-proxy
:::
:::
:::

::: {.slide title="AdaGrad: Cumulative Per-Coordinate Normalization"}
[AdaGrad]{.kicker}

::: {.cols .vc}
::: {.col}
**AdaGrad** accumulates the sum of squared gradients $\mathbf{s}_t$ elementwise and divides each step by $\sqrt{s_{t,i}}$:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t, \qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\,\mathbf{g}_t$$

- **Rare vs. frequent features:** After $10^4$ steps, a busy feature ($g_{t,1} = \pm 1$) receives step $\eta/100$, while a rare token embedding firing once receives a full step $\eta$.
- **Diagonal metric:** Steepest descent under $\mathbf{P}_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$.
:::

::: {.col .fig}
@fig:ch35-opt-adagrad-rare-frequent
:::
:::
:::

::: {.slide title="Online Convex Regret and AdaGrad's Built-In Decay"}
[AdaGrad]{.kicker}

AdaGrad was designed for **online convex optimization**, minimizing cumulative regret $\sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^T f_t(\mathbf{x})$:

- **Coordinate-adaptive regret bound:** On a box of $\ell_\infty$ diameter $D_\infty$ ($D_2 = \sqrt{d}\,D_\infty$), online gradient descent achieves regret $O(D_2 G\sqrt{T})$, whereas AdaGrad achieves:
  $$\text{Regret}_T(\text{AdaGrad}) = O\!\left(D_\infty \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2}\right)$$
  When gradient mass concentrates on a small subset of coordinates (e.g., Zipf word frequencies), AdaGrad's bound is up to $\sqrt{d}$ smaller.
- **Why AdaGrad stalls on deep networks:** Under persistent minibatch noise of variance $\sigma^2$, the cumulative sum grows linearly ($s_{t,i} \approx \sigma^2 t$), forcing the effective step size $\eta/(\sigma\sqrt{t})$ to decay relentlessly to zero even when the optimizer is far from a minimum.
:::

::: {.slide title="RMSProp: Exponential Moving Average of Squares"}
[RMSProp]{.kicker}

::: {.cols .vc}
::: {.col}
**RMSProp** replaces AdaGrad's all-time sum with an exponential moving average (EMA) of squared gradients:

$$\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2, \qquad \mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon}\,\mathbf{g}_t$$

- **Finite memory window:** Past squared gradients halve in weight every $k_{1/2} = \frac{\log 2}{1 - \beta_2}$ steps ($\approx 7$ steps at $\beta_2 = 0.9$, $\approx 693$ steps at $\beta_2 = 0.999$).
- **Non-vanishing steps:** Effective step sizes stay bounded away from zero under stationary noise and track changing curvature.
:::

::: {.col .fig}
@fig:ch35-opt-rmsprop-ema-window
:::
:::
:::

::: {.slide title="Adam: Combining Momentum with RMS Normalization"}
[Adam]{.kicker}

::: {.cols .vc}
::: {.col}
**Adam** pairs RMSProp's second-moment EMA $\mathbf{v}_t$ with a first-moment momentum EMA $\mathbf{m}_t$, correcting both for zero initialization:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}, \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

- **Standard defaults:** $\beta_1 = 0.9$ (half-life $\approx 7$ steps), $\beta_2 = 0.999$ (half-life $\approx 693$ steps), $\epsilon = 10^{-8}$.
- **Dual action:** $\hat{\mathbf{m}}_t$ averages out high-frequency oscillations before rescaling by $\sqrt{\hat{\mathbf{v}}_t}$.
:::

::: {.col .fig}
@fig:ch35-opt-adam-dual-moment-flow
:::
:::
:::

::: {.slide title="Why Zero Initialization Demands Bias Correction"}
[Adam]{.kicker}

::: {.cols .vc}
::: {.col}
If the squared gradient scale is stationary ($\mathbb{E}[\mathbf{g}_\tau^2] = \overline{\mathbf{g}^2}$), summing the geometric series gives:

$$\mathbb{E}[\mathbf{v}_t] = (1 - \beta_2)\,\overline{\mathbf{g}^2}\sum_{\tau=1}^t \beta_2^{t-\tau} = \big(1 - \beta_2^t\big)\,\overline{\mathbf{g}^2}$$

- At $\beta_2 = 0.999$, $\mathbf{v}_{10}$ holds only $1\%$ of $\overline{\mathbf{g}^2}$, and $\mathbf{v}_{1000}$ holds just $63\%$.
- **Step inflation without correction:** Dropping both corrections multiplies the update by $\frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}$, starting at $3.16\times$ ($t=1$) and peaking at **$6.6\times$** ($t=12$) because $\mathbf{m}_t$ fills $100\times$ faster than $\mathbf{v}_t$.
:::

::: {.col .fig}
@fig:mdl-opt-bias-correction
:::
:::
:::

::: {.slide title="Exact Coordinate Scale Invariance of Adam"}
[Scale Invariance]{.kicker}

::: {.cols .vc}
::: {.col}
When $\epsilon = 0$ and $\nabla\tilde{f}(\mathbf{x}) = \mathrm{diag}(c_1, \ldots, c_d)\nabla f(\mathbf{x})$ ($c_i > 0$), linearity of the EMAs gives $\tilde{m}_{t,i} = c_i m_{t,i}$ and $\tilde{v}_{t,i} = c_i^2 v_{t,i}$:

$$\frac{\hat{\tilde{m}}_{t,i}}{\sqrt{\hat{\tilde{v}}_{t,i}}} = \frac{c_i\,\hat{m}_{t,i}}{\sqrt{c_i^2\,\hat{v}_{t,i}}} = \frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}}}$$

- **Identical trajectories:** Adam on $\tfrac12(x_1^2 + 1000\,x_2^2)$ from $(1, 1)$ follows the exact same path $x_{t,1} = x_{t,2}$ as on the round sphere $\tfrac12(x_1^2 + x_2^2)$.
:::

::: {.col .fig}
@fig:ch35-opt-adam-scale-invariance
:::
:::
:::

::: {.slide title="Consequences and Limits of Scale Invariance"}
[Scale Invariance]{.kicker}

Applying the scale-invariance theorem to $f(\mathbf{x}) = \tfrac12\sum_{i=1}^d h_i x_i^2$ explains both Adam's strength on aligned valleys and its structural boundaries:

- **Every aligned valley is a round bowl:** Because $\nabla f(\mathbf{x}) = \mathrm{diag}(h)\,\mathbf{x}$ is the gradient of the isotropic sphere $\tfrac12\|\mathbf{x}\|^2$ rescaled by $c_i = h_i$, Adam on $\tfrac12(x_1^2 + 1000\,x_2^2)$ from $(1, 1)$ follows the *exact same trajectory* $x_{t,1} = x_{t,2}$ as Adam on $\tfrac12(x_1^2 + x_2^2)$.
- **Distance replaces curvature:** Because early steps move each coordinate by roughly $\pm\eta$ regardless of $h_i$, the iteration count scales with the initial distance $\|\mathbf{x}_0\|_\infty / \eta$ rather than $\kappa = h_{\max}/h_{\min}$.
- **Three boundaries of invariance:**
  1. $\epsilon > 0$ breaks invariance once $|g_{t,i}| \lesssim \epsilon$.
  2. Adam is *not* sign descent in general: it reduces to $-\eta\,\mathrm{sign}(\mathbf{g}_t)$ only at $t=1$, when $\beta_1 = \beta_2 = 0$, or when $|g_{t,i}|$ is constant over $\hat{v}_t$'s memory.
  3. Invariance holds only for *diagonal* coordinate rescalings, failing under orthogonal rotations.
:::

::: {.slide title="Fair Benchmark: Aligned vs. 45° Rotated Valley"}
[Rotation Sensitivity]{.kicker}

We compare optimally tuned GD and Heavy Ball (given the exact eigenvalues $\mu = 1, L = 1000$) against a 7-point learning-rate sweep for Adam ($\beta_1 = 0$ scaling-only vs. $\beta_1 = 0.9$) on the aligned valley and the valley rotated by $\theta = 45^\circ$:

@!mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-1
:::

::: {.slide title="Dissecting the Benchmark: Robustness vs. Rotation"}
[Rotation Sensitivity]{.kicker}

::: {.cols .vc}
::: {.col}
- **Aligned ($\theta = 0^\circ$):** Scaling-only Adam ($\beta_1 = 0$) finishes in **$22$ steps**, and default Adam ($\beta_1 = 0.9$) finishes in $165$–$205$ steps across a $100\times$ range of $\eta$.
- **Rotated ($\theta = 45^\circ$):** GD ($6{,}160$ steps) and Heavy Ball ($296$ steps) are rotation-invariant, whereas scaling-only Adam locks into a 2-cycle and default Adam slows to $559$–$12{,}862$ steps.
- **Why $45^\circ$ breaks diagonal scaling:** The steep mode $(-1, 1)/\sqrt{2}$ projects equally onto both axes, making $\hat{v}_{t,1} \approx \hat{v}_{t,2}$ so the diagonal preconditioner collapses to a scalar!
:::

::: {.col .fig .big}
@fig:ch35-opt-rotated-valley-failure
:::
:::
:::

::: {.slide title="Effective Step Sizes Across Curvature Ratios"}
[Rotation Sensitivity]{.kicker}

On the aligned valley ($\beta_1 = 0, \eta = 0.1, \epsilon = 0$), Adam discovers the $1000{:}1$ curvature ratio on step 1 from gradient magnitudes alone:

@!mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-2

- Increasing $\kappa$ by $10^5\times$ ($1$ $\to$ $10^3$ $\to$ $10^5$) adds only $14$ steps ($40$ $\to$ $48$ $\to$ $54$), whereas moving $\mathbf{x}_0$ $100\times$ farther away ($1$ $\to$ $100$) multiplies the step count by $\approx 86\times$ ($48$ $\to$ $4{,}145$).
:::

::: {.slide title="Reddi's Convex Counterexample: When Adam Diverges"}
[Convergence Limits]{.kicker}

::: {.cols .vc}
::: {.col}
On $x \in [-1, 1]$, consider the 3-periodic convex losses $(+4x, -x, -x)$ with average slope $+2$ and unique optimum $x^\star = -1$:

- **Short EMA memory ($\beta_2 = 1/17, \beta_1 = 0$):** Across $(+4, -1, -1)$, $\sqrt{v_t}$ settles to $(3.89, 1.35, 1.02)$, producing normalized steps $-1.03\eta, +0.74\eta, +0.98\eta$ (net drift **$+0.69\eta > 0$** toward the worst point $x = +1$!).
- **AMSGrad fix:** Using the running maximum $\tilde{v}_t = \max(\tilde{v}_{t-1}, v_t)$ pins $\sqrt{\tilde{v}_t} = 3.89$, restoring net drift $-0.51\eta$ to $x^\star = -1$.
:::

::: {.col .fig .big}
@fig:ch35-opt-reddi-cycle-drift
:::
:::
:::

::: {.slide title="AMSGrad Restores Convergence via Running Maxima"}
[Convergence Limits]{.kicker}

**AMSGrad** repairs the Reddi counterexample by replacing $\mathbf{v}_t$ in the denominator with its running maximum $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$, guaranteeing nonincreasing effective steps $\eta_t / \sqrt{\tilde{\mathbf{v}}_t}$:

@!adaptive-stochastic-methods-reddi

- With $\tilde{v}_t$ pinned at $15.12$, the three steps become $-1.03\eta, +0.26\eta, +0.26\eta$ (net drift $-0.51\eta$ toward $x^\star = -1$), matching SGD while standard Adam locks onto $x = +1$.
:::

::: {.slide title="Weight Decay in Adaptive Methods: Coupled vs. AdamW"}
[Decoupled Weight Decay]{.kicker}

In plain SGD, adding an $\ell_2$ penalty $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss ($\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\mathbf{w}_t$) is identical to multiplying weights by $(1 - \eta\lambda)$ at every step. In Adam, passing $\lambda\mathbf{w}_t$ through the preconditioner $\sqrt{\hat{\mathbf{v}}_t}$ couples decay to gradient scale:

$$\text{Coupled ($\ell_2$ inside Adam):}\quad \Delta w_{t,i}^{\mathrm{decay}} = -\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\,w_{t,i} \qquad \text{vs.} \qquad \text{Decoupled (AdamW):}\quad \Delta w_{t,i}^{\mathrm{decay}} = -\eta\lambda\,w_{t,i}$$

- **The flaw of coupled $\ell_2$ decay:** Coordinates with large historical gradients ($\sqrt{\hat{v}_{t,i}} \gg 1$) receive almost zero weight decay, while quiet coordinates ($\sqrt{\hat{v}_{t,i}} \ll 1$) are shrunk aggressively.
- **The AdamW update (Loshchilov and Hutter, 2019):**
  $$\mathbf{w}_{t+1} = (1 - \eta\lambda)\,\mathbf{w}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$
  Only the data-loss gradient enters $\mathbf{m}_t$ and $\mathbf{v}_t$, while multiplicative shrinkage $(1 - \eta\lambda)$ acts uniformly on all parameters.
:::

::: {.slide title="How Coupled Decay Distorts Shrinkage Under Noise"}
[Decoupled Weight Decay]{.kicker}

Simulate two groups of $10^4$ weights initialized at $w_0 = 1$ receiving pure zero-mean Gaussian gradient noise at scales $\sigma_1 = 10$ and $\sigma_2 = 0.1$ over $T = 4{,}000$ steps ($\eta = 10^{-3}, \lambda = 0.1$):

@!adaptive-stochastic-methods-adamw

- Coupled Adam decays the quiet weights $100\times$ faster ($10^{-3}$ vs $10^{-5}$ per step, ending at $0.031$ vs $0.961$), whereas AdamW shrinks both groups uniformly to $(1 - \eta\lambda)^T \approx \exp(-0.4) \approx 0.670$.
:::

::: {.slide title="Exact-Gradient Fixed Points: MAP vs. ℓ∞ Box"}
[Decoupled Weight Decay]{.kicker}

::: {.cols .vc}
::: {.col}
When exact gradients settle ($\hat{\mathbf{m}}_t \to \mathbf{u}, \hat{\mathbf{v}}_t \to \mathbf{u}^2$), coupled Adam and AdamW converge to distinct stationary points:

$$\lambda\,w_i\big(|g_i| + \epsilon\big) = -g_i \implies |w_i| = \frac{|g_i|}{\lambda(|g_i| + \epsilon)} < \frac{1}{\lambda}$$

- **Coupled Adam:** Vanishes iff $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$ (Gaussian MAP).
- **Decoupled AdamW ($\epsilon \to 0$):** Enforces the KKT conditions for an implicit $\ell_\infty$ box constraint $\min_{\|\mathbf{w}\|_\infty \le 1/\lambda} \ell(\mathbf{w})$!
:::

::: {.col .fig}
@fig:ch35-opt-adamw-linf-box
:::
:::
:::

::: {.slide title="Anatomy of Opening Steps and Learning-Rate Warmup"}
[Warmup Dynamics]{.kicker}

At step $t = 1$, bias correction gives $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so Adam's very first update is a full sign step in every coordinate:

$$\mathbf{x}_2 = \mathbf{x}_1 - \eta\,\frac{\mathbf{g}_1}{|\mathbf{g}_1| + \epsilon} \approx \mathbf{x}_1 - \eta\,\mathrm{sign}(\mathbf{g}_1)$$

- **Why pure-noise coordinates jump $4\times$ farther at $t = 1$:** On a coordinate whose gradient is independent zero-mean noise $g_t \sim \mathcal{N}(0, \sigma^2)$, the step at $t = 1$ has RMS magnitude $1.00\,\eta$.
- **Steady-state variance of the momentum EMA:** Once $\mathbf{m}_t = (1 - \beta_1)\sum_{\tau \le t}\beta_1^{t-\tau}g_\tau$ has averaged $\sim \frac{1}{1-\beta_1} = 10$ independent samples, its variance drops to $\frac{1 - \beta_1}{1 + \beta_1}\sigma^2$, so the steady-state RMS update is only:
  $$\text{RMS}\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}\right) \to \sqrt{\frac{1 - \beta_1}{1 + \beta_1}} = \sqrt{\frac{0.1}{1.9}} \approx 0.23$$</
:::

::: {.slide title="Decomposing Early Instability: Momentum vs. Bias"}
[Warmup Dynamics]{.kicker}

Tracking the RMS update (in units of $\eta$) across $20{,}000$ independent noise coordinates separates the role of the momentum window, bias correction, and linear warmup:

@!mdl-adaptive-stochastic-methods-warmup
:::

::: {.slide title="Decomposing Early Instability: Momentum vs. Bias (cont.)"}
[Warmup Dynamics]{.kicker}

- Replacing $\hat{\mathbf{v}}_t$ with the exact variance $\sigma^2 = 1$ leaves the $1.00 \to 0.23$ curve unchanged: early step inflation on noisy coordinates comes from the $10$-step momentum window (tamed by a $20$-step warmup), whereas longer warmups in deep nets guard against high initial Hessian sharpness $\lambda_{\max}$.
:::

::: {.slide title="Beyond Diagonal Scaling: Kronecker-Factored Metrics"}
[Structured Preconditioners]{.kicker}

To handle rotated valleys where curvature mixes coordinates without paying $O(d^2)$ storage for a full $d \times d$ preconditioner, structured methods exploit the **matrix shape** of neural network layers:

- **Rank-one layer gradients:** For a linear layer $\mathbf{y} = \mathbf{W}\mathbf{a}$ with $\mathbf{W} \in \mathbb{R}^{m \times n}$, input $\mathbf{a} \in \mathbb{R}^n$, and output gradient $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y} \in \mathbb{R}^m$, the weight gradient is $\mathbf{G} = \boldsymbol{\delta}\mathbf{a}^\top$, or in stacked vector form $\mathrm{vec}(\mathbf{G}) = \mathbf{a} \otimes \boldsymbol{\delta}$.
- **Kronecker factorization:** Approximating the $(mn) \times (mn)$ outer-product curvature $\mathbb{E}[(\mathbf{a}\mathbf{a}^\top) \otimes (\boldsymbol{\delta}\boldsymbol{\delta}^\top)]$ by $\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}$ (where $\mathbf{S}_{\mathrm{in}} = \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \in \mathbb{R}^{n \times n}$ and $\mathbf{S}_{\mathrm{out}} = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top] \in \mathbb{R}^{m \times m}$) lets us invert and apply the preconditioner via two small matrix products:
  $$\big(\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}\big)^{-1}\mathrm{vec}(\mathbf{V}) = \mathrm{vec}\big(\mathbf{S}_{\mathrm{out}}^{-1}\,\mathbf{V}\,\mathbf{S}_{\mathrm{in}}^{-1}\big)$$
:::

::: {.slide title="Storage and FLOP Savings of Kronecker Factoring"}
[Structured Preconditioners]{.kicker}

For a single $4096 \times 4096$ Transformer weight matrix ($d = mn \approx 1.68 \times 10^7$ parameters), compare a full $(mn) \times (mn)$ curvature matrix against two $4096 \times 4096$ Kronecker factors:

@!mdl-adaptive-stochastic-methods-kronecker-factors

- Kronecker factoring reduces storage by **$8.4 \times 10^6\times$** (from $2.8 \times 10^{14}$ entries to $3.4 \times 10^7$) and inversion cost by **$3.4 \times 10^{10}\times$** (from $4.7 \times 10^{21}$ FLOPs to $1.4 \times 10^{11}$ FLOPs).
:::

::: {.slide title="The Preconditioning Ladder: K-FAC, Shampoo, Muon"}
[Structured Preconditioners]{.kicker}

::: {.cols .vc}
::: {.col}
Modern matrix optimizers apply two-sided preconditioning $\mathbf{G} \mapsto \mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\mathbf{P}_{\mathrm{r}}^{-a}$ to an $m \times n$ gradient/momentum matrix $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$:

$$\big(\mathbf{G}\mathbf{G}^\top\big)^{-1/4}\,\mathbf{G}\,\big(\mathbf{G}^\top\mathbf{G}\big)^{-1/4} = \mathbf{U}\mathbf{V}^\top$$

- **K-FAC ($a = 1$) & Shampoo ($a = 1/4$):** Approximate the Fisher / matrix-AdaGrad factors via row and column covariances.
- **Muon (polar factor $\mathbf{U}\mathbf{V}^\top$):** Equalizes all singular values to $1$ via fast Newton–Schulz iterations and commutes with arbitrary orthogonal rotations.
:::

::: {.col .fig}
@fig:ch35-opt-preconditioning-ladder
:::
:::
:::

::: {.slide title="Concept Check: Scale Invariance, AdamW, and Muon"}
[Concept Check]{.kicker}

Test your understanding of adaptive and matrix-preconditioned optimizers:

1. At step $t = 1$, how far does bias-corrected Adam ($\epsilon = 10^{-8}$) move a coordinate with gradient $g_{1,i} = 10^{-6}$, and after how many steps does $\beta_2 = 0.999$ halve the weight of an old squared gradient?
2. On the 1D loss $\ell(w) = \tfrac12(w - 3)^2$ with exact gradients and $\lambda = 1$, what is the stationary fixed point of coupled Adam ($\ell_2$ penalty) versus decoupled AdamW ($\epsilon \to 0$)?
3. Why does rotating the $1000{:}1$ valley by $45^\circ$ defeat Adam's diagonal scaling while leaving Muon's polar factor $\mathbf{U}\mathbf{V}^\top$ equivariant?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** At $t = 1$, $\hat{m}_{1,i}/\sqrt{\hat{v}_{1,i}} = \mathrm{sign}(g_{1,i})$, so the step size is $\eta \frac{10^{-6}}{10^{-6} + 10^{-8}} \approx \mathbf{0.99\,\eta}$. The half-life for $\beta_2 = 0.999$ is $\frac{\log 2}{1 - 0.999} \approx \mathbf{693\text{ steps}}$. **(2) Coupled Adam** solves $(w - 3) + \lambda w = 0 \implies w^\star = \frac{3}{1+1} = \mathbf{1.5}$. **AdamW** clips the unconstrained minimizer $w = 3$ to the $\ell_\infty$ box $|w| \le 1/\lambda = 1$, resting at $w^\star = \mathbf{1.0}$. **(3)** A $45^\circ$ rotation mixes coordinates so both axes see the steep mode ($\hat{v}_1 \approx \hat{v}_2$), reducing Adam's diagonal preconditioner to a scalar identity. By contrast, under orthogonal rotations $\mathbf{Q}_1\mathbf{G}\mathbf{Q}_2^\top = (\mathbf{Q}_1\mathbf{U})\boldsymbol{\Sigma}(\mathbf{Q}_2\mathbf{V})^\top$, the polar factor maps to $\mathbf{Q}_1(\mathbf{U}\mathbf{V}^\top)\mathbf{Q}_2^\top$, rotating cleanly with the basis.
:::
:::

::: {.slide title="Summary: Adaptive and Structured Preconditioners"}
[Key Takeaways]{.kicker}

| Optimizer | Preconditioner / Update Form | Invariance Property | Memory Overhead | Primary Strength & Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **AdaGrad** | $\mathbf{s}_t = \sum_{\tau \le t}\mathbf{g}_\tau^2$, step $\propto \mathbf{s}_t^{-1/2}\mathbf{g}_t$ | Diagonal coordinate scale | $1$ vector ($d$) | Optimal sparse convex regret; $1/\sqrt{t}$ decay stalls on deep nets |
| **RMSProp** | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | Diagonal coordinate scale | $1$ vector ($d$) | Non-vanishing steps; noisy without momentum smoothing |
| **Adam / AMSGrad** | Bias-corrected $\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ | Exact diagonal scale ($\epsilon=0$) | $2$ vectors ($2d$) | Fast & robust on axis-aligned valleys; fails under $45^\circ$ rotation |
| **AdamW** | Decoupled $(1-\eta\lambda)\mathbf{w}_t - \eta\hat{\mathbf{m}}_t/\sqrt{\hat{\mathbf{v}}_t}$ | Uniform decay across scales | $2$ vectors ($2d$) | Default for Transformers; fixed points lie in $\ell_\infty$ box $\|\mathbf{w}\|_\infty \le 1/\lambda$ |
| **K-FAC / Shampoo** | Two-sided $\mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\mathbf{P}_{\mathrm{r}}^{-a}$ ($m \times n$ layer) | Layer row/column transforms | $m^2 + n^2$ per layer | Captures cross-coordinate correlations; periodic matrix roots/inverses |
| **Muon** | Orthogonal polar factor $\mathbf{U}\mathbf{V}^\top$ of $\mathbf{M}_t$ | Orthogonal row/col rotations | $1$ matrix ($mn$) | Equalizes singular values via fast Newton–Schulz matmuls |
:::

