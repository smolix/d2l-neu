# Adaptive Methods
:label:`sec_mdl-adaptive-stochastic-methods`

The transformers in this book, like most large networks, are trained with
Adam or its variant AdamW (:numref:`sec_adam`) rather than with plain
stochastic gradient descent. One reason is the step size. The methods of
:numref:`sec_mdl-stochastic-gradient-methods` use one step size for every
coordinate, and their schedules change that step only over time. A network
has $10^7$ to $10^{11}$ coordinates whose curvatures differ by orders of
magnitude and are not known in advance. With one step size for all of them,
the stiffest coordinate sets the step and the others crawl. This section
asks how an optimizer can choose a step size for each coordinate from the
gradients alone, and when that choice fails.

The chapter's running example shows how much is at stake. Take the valley
$f(\mathbf{x}) = \tfrac12\bigl(x_1^2 + 1000\, x_2^2\bigr)$ and start at
$(1, 1)$. Gradient descent with the best single step size, $\eta = 2/1001$,
needs 6,160 steps to bring $f$ below $10^{-8}$, because a step small enough
to keep $x_2$ stable is a thousand times too small for $x_1$. The heavy-ball
method of :numref:`sec_mdl-gradient-based-optimization`, which adds a
fraction of the previous step to each new one, needs 296 steps. An optimizer
that gives each coordinate its own step size, estimated only from the
gradients it has seen, needs 48.

The remedy is to divide each coordinate's step by the root mean square of its
own past gradients. AdaGrad, RMSProp and Adam, which this section derives, all
do this, so each of them multiplies the gradient by a diagonal matrix before
the step. That is also the limit of the remedy: it works when the axes of the
curvature are the coordinate axes. Rotate the valley by $45^\circ$ and the
same optimizer never reaches $f < 10^{-8}$. A periodic sequence of convex
losses even drives it to the worst feasible point.

Dividing by a gradient scale has three further consequences, which the rest
of the section follows. Weight decay that passes through the division no
longer shrinks all weights alike. The first steps are taken before the scale
has been estimated. And the rescaling can use the matrix structure of a layer
instead of a diagonal. Throughout, $t$ counts optimizer steps, $h_i$ is the
curvature of coordinate $i$, and $\lambda$ is the weight-decay coefficient.

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

Gradient descent on a quadratic is stable only when the step is below
$2/h_{\max}$, twice the inverse of the largest curvature. The flattest
direction then shrinks by a factor of only about $1 - 2/\kappa$ per step, where
$\kappa = h_{\max}/h_{\min}$ is the condition number, so the number of steps
grows in proportion to $\kappa$ (:numref:`sec_mdl-gradient-based-optimization`).
The smallest case in which a cure is visible is the diagonal quadratic in two
variables,

$$
f(\mathbf{x}) = \tfrac12\bigl(h_1 x_1^2 + h_2 x_2^2\bigr),
\qquad
\frac{\partial f}{\partial x_i} = h_i x_i .
$$

Each partial derivative depends on its own coordinate only, so nothing forces
the two coordinates to share a step. With a separate step $\eta_i$, coordinate
$i$ moves to $x_i - \eta_i h_i x_i = (1 - \eta_i h_i)\, x_i$, and the choice
$\eta_i = 1/h_i$ sends every coordinate to zero in one step. For
$h = (1, 1000)$ the two steps are $1$ and $10^{-3}$. This is Newton's method
for a diagonal Hessian, and the condition number has disappeared from the
iteration count.

Two obstacles stand between this observation and an algorithm. The curvatures
$h_i$ are not known, and away from a fixed quadratic they change along the
run. And in training the gradients are noisy minibatch estimates. The step of
each coordinate must therefore be estimated from the gradients themselves. The
available estimate is the size of the gradient. On the diagonal quadratic the
gradient entry $g_i = h_i x_i$ has size $h_i |x_i|$. When the coordinates have
comparable magnitudes, a coordinate with large gradients therefore has large
curvature, and dividing by the gradient size divides by the curvature.

### AdaGrad

A global decaying step, $\eta/\sqrt{t}$, is the classical choice for noisy
convex problems. It treats every coordinate alike, which fails when the
coordinates are used at very different rates. Suppose coordinate 1 receives a
gradient of $\pm 1$ at every step and coordinate 2 receives $\pm 1$ once every
$10^4$ steps and zero otherwise, as the weight of a rare word does in a
bag-of-words model. At $t = 10^4$ the global schedule gives both coordinates
the step $\eta/100$, so the one piece of evidence about the rare coordinate
moves it by one hundredth of $\eta$. If instead each coordinate accumulates its
own squared gradients, the sums are $10^4$ and $1$, and dividing $\eta$ by
their square roots gives the busy coordinate $\eta/100$ and the rare one
$\eta$.

This is **AdaGrad** :cite:`Duchi.Hazan.Singer.2011`. Write
$\mathbf{g}_t \odot \mathbf{g}_t$ for the vector of squared entries of the
gradient (we abbreviate it as $\mathbf{g}_t^2$ from here on); square roots and
divisions of vectors are also taken entry by entry. AdaGrad keeps the running
sum $\mathbf{s}_t$ and divides by its square root:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\, \mathbf{g}_t,
$$
:eqlabel:`eq_mdl-opt-adagrad`

with $\mathbf{s}_0 = \mathbf{0}$ and a small constant $\epsilon$ that keeps the
division finite. A coordinate's step decays with its own activity, not with
the step count.

*The metric view.* Steepest descent depends on how step length is measured.
Measuring it by $\|\mathbf{d}\|_{\mathbf{P}} = \sqrt{\mathbf{d}^\top \mathbf{P}\, \mathbf{d}}$
for a positive definite $\mathbf{P}$ makes the steepest direction
$-\mathbf{P}^{-1}\nabla f$ (Exercise
:exref:`ex_mdl-gradient-based-optimization-show-that-is-de` of
:numref:`sec_mdl-gradient-based-optimization`). AdaGrad is steepest descent in
the metric $\mathbf{P}_t = \mathrm{diag}(\sqrt{\mathbf{s}_t})$: moving along
coordinate $i$ costs in proportion to the evidence
$\sqrt{\sum_{\tau \le t} g_{\tau,i}^2}$ that its gradient has been large. A
matrix such as $\mathbf{P}_t$, which rescales the gradient before the step, is
called a **preconditioner**. Newton's method would use the true curvature $\mathrm{diag}(h_i)$; AdaGrad
replaces it by the gradient history (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-adagrad-metric-view`).

*The regret view.* AdaGrad was designed for **online convex optimization**.
Convex losses $f_1, f_2, \ldots$ arrive one at a time, and the algorithm must
choose $\mathbf{x}_t$ before it sees $f_t$. Its performance after $T$ rounds
is the **regret**,
$\sum_{t=1}^T f_t(\mathbf{x}_t) - \min_{\mathbf{x}} \sum_{t=1}^T f_t(\mathbf{x})$,
the excess loss over the best fixed point chosen in hindsight. An algorithm
learns, in this sense, when its average regret (the regret divided by $T$)
tends to zero.

The regret bounds of online gradient descent and of AdaGrad count the
gradients differently. On a convex set of Euclidean diameter
$D_2$, with every gradient bounded by $\|\mathbf{g}_t\| \le G$, online
gradient descent with a tuned global step has regret of order $D_2 G \sqrt{T}$
(Exercise :exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd` proves it).
AdaGrad has regret of order

$$
D_\infty \sum_{i=1}^{d} \sqrt{\textstyle\sum_{t=1}^{T} g_{t,i}^2}
$$

:cite:`Duchi.Hazan.Singer.2011`, where $D_\infty$ is the diameter of the set
in the $\ell_\infty$ norm and the sum runs over the $d$ coordinates. The bound
adds up the gradient history of each coordinate separately, instead of
charging every coordinate for the largest gradient.

On the box $[-1, 1]^d$, $D_2 = \sqrt{d}\, D_\infty$. When the gradient mass
sits on a few coordinates, the AdaGrad sum is about $G\sqrt{T}$, and AdaGrad's
bound is up to $\sqrt{d}$ times smaller. When the mass is spread evenly, the
two bounds agree. Take one-hot gradients that fall on coordinate $i$ with
probability $p_i$. The expected ratio of the two bounds is then at most
$\sum_i \sqrt{p_i}/\sqrt{d}$, and close to it for large $T$
(Exercise :exref:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`).
The ratio is $1.00$ when the $p_i$ are uniform, even though every gradient is
sparse, and $0.64$ for word frequencies that follow Zipf's law with exponent
$1$ at $d = 10^4$. The gain therefore comes from concentration of the gradient
mass, not from sparsity as such.

The square root in :eqref:`eq_mdl-opt-adagrad` is the choice that balances
the two terms of the regret proof, the distance term $D^2/\eta_T$ against the
accumulated step term $\sum_t \eta_t g_t^2$ (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-regret-ogd`c). Steps that decay
faster than $1/\sqrt{t}$ inflate the distance term, and steps that decay more
slowly inflate the step term.

The accumulation is also AdaGrad's weakness outside this setting. Under
persistent gradient noise of variance $\sigma^2$, $s_{t,i} \approx \sigma^2 t$,
so the effective step is about $\eta/(\sigma\sqrt{t})$, and the rule carries
its own decay schedule. The Robbins–Monro conditions of :numref:`sec_mdl-stochastic-gradient-methods`
ask for steps whose sum diverges, so that the iterate can travel any distance,
and whose squares have a finite sum, so that the injected noise stays bounded.
The squares of $1/\sqrt{t}$ sum to $\sum_t 1/t = \infty$, so AdaGrad's decay is
not of that kind. It suffices for $\sqrt{T}$ regret on convex problems. On long
nonconvex runs it is a drawback: the steps keep shrinking whether or not the
loss is still falling, and a plateau met late is crossed with almost no step
left.

### RMSProp

**RMSProp** :cite:`Tieleman.Hinton.2012` replaces the all-time sum by an
exponential moving average, so that the scale estimate follows the recent
gradients:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon}\, \mathbf{g}_t .
$$
:eqlabel:`eq_mdl-opt-rmsprop`

Unrolled, $\mathbf{v}_t$ gives the squared gradient of $k$ steps ago the weight
$(1 - \beta_2)\beta_2^k$, so the weights halve about every
$\log 2/(1 - \beta_2)$ steps: every 7 steps at RMSProp's customary
$\beta_2 = 0.9$, every 693 at $\beta_2 = 0.999$. The preconditioner now
estimates the current gradient scale of each coordinate rather than the
lifetime total, and the steps no longer decay to zero by construction. But a
gradient seen once is now forgotten on that timescale, which the
counterexample below exploits.

### Adam and Its Bias Correction

RMSProp divides by a smoothed gradient scale but steps along the raw
minibatch gradient, noise included. Averaging the gradient itself over recent
steps, as heavy-ball momentum does, smooths that noise and keeps the iterate
moving along directions that persist.

**Adam** :cite:`Kingma.Ba.2014` applies the moving average of RMSProp to the
gradient itself and corrects both averages for their zero start. The chapter on
optimization algorithms runs it on real models (:numref:`sec_adam`). The
update is

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

with $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$ and the defaults
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Each coordinate moves
by a momentum-averaged gradient divided by its root-mean-square gradient. The
average $\mathbf{m}_t$ is the heavy-ball velocity of
:numref:`subsec_mdl-momentum-acceleration` in averaged form: heavy ball adds
gradients, $\mathbf{d}_t = \beta\mathbf{d}_{t-1} - \eta\,\mathbf{g}_t$, and the
factor $1 - \beta_1$ turns the sum into a weighted average. For a constant
$\eta$ and without the correction $\hat{\mathbf{m}}_t$, the two differ only by
a rescaling of the step.

The division by $1 - \beta_2^t$ corrects for the zero start. We show that, in
expectation, $\mathbf{v}_t$ is a known fraction of the true squared-gradient
scale, so that dividing by the fraction removes the bias. Unroll the recursion from $\mathbf{v}_0 = \mathbf{0}$:

$$
\mathbf{v}_t = (1 - \beta_2) \sum_{\tau=1}^{t} \beta_2^{\,t-\tau}\, \mathbf{g}_\tau^2 .
$$

Assume that the squared-gradient scale is stationary,
$\mathbb{E}[\mathbf{g}_\tau^2] = \overline{\mathbf{g}^2}$ for every $\tau$.
Taking expectations and summing the geometric series
$\sum_{\tau=1}^{t} \beta_2^{\,t-\tau} = (1 - \beta_2^t)/(1 - \beta_2)$ gives

$$
\mathbb{E}[\mathbf{v}_t]
= (1 - \beta_2)\, \overline{\mathbf{g}^2} \sum_{\tau=1}^{t} \beta_2^{\,t-\tau}
= \left(1 - \beta_2^t\right) \overline{\mathbf{g}^2} .
$$
:eqlabel:`eq_mdl-opt-bias-correction`

The zero start leaves $\mathbf{v}_t$ with only the fraction $1 - \beta_2^t$ of
the scale it estimates, and $\hat{\mathbf{v}}_t$ is unbiased for
$\overline{\mathbf{g}^2}$ at every $t$ under stationarity. The same
computation with $\beta_1$ handles $\hat{\mathbf{m}}_t$ (Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-bias-correction-first-moment`).

The deficit is large. At the defaults $\mathbf{v}_{10}$ holds $1 - 0.999^{10}
\approx 1\%$ of its target, and $\mathbf{v}_{1000}$ still only $63\%$. Without
both corrections the update $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$ is mis-scaled
by the factor $(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$, which is $3.16$ at
$t = 1$ and peaks at $6.6$ at $t = 12$. The factor exceeds one because the
momentum average fills in about ten steps, while the second-moment average
needs about a thousand.
:numref:`fig_mdl-opt-bias-correction` plots the deficit and the mis-scaling.

![Bias correction in Adam. Left: with zero initialization, the running average $\mathbf{v}_t$ holds only the fraction $1-\beta_2^t$ of the squared-gradient scale; at $\beta_2=0.999$ it reaches $63\%$ after $1000$ steps, and dividing by $1-\beta_2^t$ restores the full scale at every $t$. Right: the mis-scaling of the uncorrected update $\mathbf{m}_t/\sqrt{\mathbf{v}_t}$, the factor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$. It is $3.16$ at $t=1$, peaks at $6.6$ at $t=12$ (the numerator fills in about ten steps, the denominator needs about a thousand), and returns to $1$ only on the $1/(1-\beta_2)$ timescale. Uncorrected Adam takes its largest steps when its preconditioner rests on the fewest gradients.](../img/mdl-opt-bias-correction.svg)
:label:`fig_mdl-opt-bias-correction`

An unbiased $\hat{\mathbf{v}}_t$ still leaves the step size biased. Adam
divides by $\sqrt{\hat{\mathbf{v}}_t}$, not by $\hat{\mathbf{v}}_t$, and the
square root of an unbiased estimate is biased low: at $t = 1$,
$\sqrt{\hat{v}_1} = |g_1|$, and $\mathbb{E}|g| < \sqrt{\mathbb{E}[g^2]}$
unless $|g|$ is constant. The warmup subsection below returns to these first
steps.

### Scale Invariance

The valley of the opening has $h_2 = 1000\, h_1$, yet Adam crosses it in 48
steps, as if the ratio were not there. The reason is a symmetry of the update.

**Proposition (Adam ignores the scale of each coordinate).** *Let
$\epsilon = 0$ and let $\mathbf{C} = \mathrm{diag}(c_1, \ldots, c_d)$ with
every $c_i > 0$. Suppose two problems have gradients related by
$\nabla f'(\mathbf{x}) = \mathbf{C}\, \nabla f(\mathbf{x})$ at every
$\mathbf{x}$. Then Adam with the same $\beta_1$, $\beta_2$, step schedule and
start produces the same iterates on both problems, as long as the updates are
defined.*

**Proof.** By induction on $t$. Suppose the iterates agree up to
$\mathbf{x}_t$. Then $\mathbf{g}'_\tau = \mathbf{C}\mathbf{g}_\tau$ for every
$\tau \le t$, and since the recursions for $\mathbf{m}$ and $\mathbf{v}$ are
linear in $\mathbf{g}_\tau$ and $\mathbf{g}_\tau^2$, we get
$m'_{t,i} = c_i m_{t,i}$ and $v'_{t,i} = c_i^2 v_{t,i}$. The bias corrections
are the same scalars on both problems. Hence
$\hat{m}'_{t,i}/\sqrt{\hat{v}'_{t,i}} = c_i \hat{m}_{t,i}/(c_i \sqrt{\hat{v}_{t,i}})
= \hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$, the updates coincide, and
$\mathbf{x}'_{t+1} = \mathbf{x}_{t+1}$. $\blacksquare$

On the diagonal quadratic, $\nabla f(\mathbf{x}) = \mathrm{diag}(h)\,\mathbf{x}$
is the gradient of the round bowl $\tfrac12\|\mathbf{x}\|^2$ multiplied by
$\mathbf{C} = \mathrm{diag}(h)$. So Adam on the $1000{:}1$ valley follows,
step for step, the path of Adam on the round bowl, and on the bowl the two
coordinates, started equal, stay equal. Each coordinate runs the same
one-dimensional iteration on $\tfrac12 x^2$,

$$
x_{t+1} = x_t \Bigl(1 - \frac{\eta}{\sqrt{\hat{v}_t}}\Bigr)
\qquad (\beta_1 = 0),
$$

where $\hat{v}_t$ averages the squares $x_\tau^2$, $\tau \le t$. Since the earlier
iterates were larger, $\sqrt{\hat{v}_t}$ exceeds $|x_t|$ from the second step
on, and the step is much smaller than $\eta$. For $\eta = 0.1$ the
contraction factor is $0.9$ at $t = 1$ and $0.67$ at $t = 48$, as
$\hat{v}_t$ slowly follows the shrinking iterates. These iterates are the same
for every ratio $h_2/h_1$. Only the stopping test depends on the ratio,
because $f(\mathbf{x}_t) = \tfrac12(h_1 + h_2)\, x_t^2$ weighs the common
iterate by the curvatures. The test $f < 10^{-8}$ is passed after 40 steps on the round bowl, 48 at $h_2/h_1 = 10^3$ and 54 at $10^5$. The
ratio costs a few steps of a geometric contraction, not a factor $\kappa$.

Adam measures each coordinate in units of its own gradient size, so the scale
of a coordinate is invisible to it. Three limits bound this statement. The
constant $\epsilon$ breaks the invariance once a coordinate's gradients fall to
the order of $\epsilon$ (Exercise :exref:`ex_mdl-adaptive-stochastic-methods-epsilon`). The proposition says nothing about a
change of basis that mixes coordinates, which the rotated valley below tests.
And the update is not sign descent: it becomes the sign step
$\eta\,\mathrm{sign}(\mathbf{g}_t)$ only for $\beta_1 = \beta_2 = 0$
(Exercise :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d), or when the gradient magnitude stays constant over the memory of
$\hat{\mathbf{v}}$.

## Coordinate Scaling and Its Limits

### Aligned and Rotated Valleys

How much of gradient descent's 6,160 steps does coordinate scaling alone
remove, and does the gain survive a rotation of the valley? To separate
scaling from momentum we run four methods: gradient descent at the best single
step; heavy ball with its tuned step and momentum; Adam with $\beta_1 = 0$,
which only rescales; and Adam with its default $\beta_1 = 0.9$. The rotated
problem replaces $\mathrm{diag}(h)$ by
$\mathbf{A}_\theta = \mathbf{R}_\theta\, \mathrm{diag}(h)\, \mathbf{R}_\theta^\top$,
where $\mathbf{R}_\theta$ is the rotation by $\theta$, and rotates the start
with it, $\mathbf{x}_0 = \mathbf{R}_\theta (1, 1)^\top$, so that the start has
the same components along the two curvature directions. At
$\theta = \pi/4$,
$\mathbf{A} = \left(\begin{smallmatrix}500.5 & -499.5\\ -499.5 & 500.5\end{smallmatrix}\right)$.
The first cell defines the problem, a loop that counts steps to the tolerance,
and the three updates as functions of the iterate, the gradient, the step
count and a state dictionary.

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

The next cell runs each method on the aligned and on the rotated valley and
counts its steps.

```{.python .input #mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-1}
mu, L = h.min(), h.max()
rho = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
methods = {'GD, eta = 2/(mu + L)': gd(2 / (mu + L)),
           'heavy ball, tuned': heavy_ball(4 / (np.sqrt(L) + np.sqrt(mu))**2,
                                           rho**2),
           'Adam, beta1 = 0, eta = 0.1': adam(0.1, beta1=0.0),
           'Adam, beta1 = 0.9, eta = 0.01': adam(0.01)}
print(f'{"steps to f < 1e-8":31s}{"aligned":>9s}{"rotated 45":>12s}')
for name, step in methods.items():
    hits = [run(step, *rotated(theta)) for theta in (0.0, np.pi / 4)]
    print(f'{name:31s}{hits[0] or "never":>9}{hits[1] or "never":>12}')
```

On the aligned valley, scaling alone needs 48 steps against gradient descent's
6,160, six times fewer than the tuned heavy ball's 296, which uses no scaling
at all. Gradient descent and heavy ball take the same number of steps on the
rotated valley, because their iterates are rotated with the problem. Scaling alone never reaches the tolerance on the rotated valley, and
Adam with momentum needs 7,908 steps at $\eta = 0.01$, more than gradient
descent. Retuning $\eta$ brings full Adam down to about 620 steps on the
rotated valley (Exercise :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`), but the run without momentum never converges at
any step size in that range, so that gain belongs to momentum, not to scaling.

Does the stiff coordinate receive a thousandfold smaller step, and do the two
coordinates run the same iteration, as the proposition predicts?
The next cell runs scaling-only Adam with $\epsilon = 0$ on the aligned valley
and prints the iterate and the effective steps $\eta/\sqrt{\hat{\mathbf{v}}_t}$.

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
```

From the first step the effective steps stand in the ratio $1000{:}1$, the
curvature ratio recovered from gradient sizes alone, with no Hessian. The two
coordinates agree to rounding error at every step, and the effective steps
grow from $0.1$ to $0.33$ on the flat coordinate as $\hat{v}_t$ follows the
shrinking gradients.

Consider now the rotated valley. Its matrix has entries of size $500$, and its
stiff direction $(-1, 1)/\sqrt{2}$ loads equally on both coordinates. The
gradient of each coordinate is therefore dominated by the stiff mode, and the
two coordinates receive almost the same scale (their $\hat{v}$ agree to within
one percent along the run). A per-coordinate scale is one number per
coordinate, so it cannot divide the stiff and the flat component of the same
coordinate by different amounts. With equal scales on both coordinates the
preconditioner is a multiple of the identity. Adam with $\beta_1 = 0$ then
becomes gradient descent whose step adapts to the stiff mode, and the flat
mode crawls as it does under gradient descent.

As the stiff component dies out, $\hat{\mathbf{v}}$ shrinks, and the effective
step grows until it reaches the stability limit $2/h_{\max}$ of the stiff
mode. The loss reaches its smallest value, $f \approx 6 \times 10^{-5}$, near
$t = 2{,}600$, and then the stiff mode starts to oscillate. The oscillation
holds the gradient magnitude constant, so $\hat{\mathbf{v}}_t \to \mathbf{g}_t^2$
and the update becomes the sign step $\eta\,\mathrm{sign}(\mathbf{g}_t)$, of
length $\eta\sqrt{2}$ along the stiff direction. The run ends in a two-cycle
of amplitude $\eta/\sqrt{2}$ across the valley, with $f = 250\,\eta^2 = 2.5$,
however long it runs. Momentum averages the alternating steps, which is why
the $\beta_1 = 0.9$ run eventually reaches the tolerance.
:numref:`fig_mdl-opt-per-coordinate` shows the three behaviors on a valley
shallow enough to draw.

![The same valley ($\kappa=20$; the text's runs use $\kappa=10^3$, whose level sets are too thin to draw) from the same start. (a) Gradient descent with one step size near the stability limit oscillates across the stiff direction and crawls along the flat one. (b) Per-coordinate normalization (Adam with $\beta_1=0$) moves both coordinates at equal speed and curves into the minimum. (c) The same update on the valley rotated by $45^\circ$, with the start rotated too: the coordinate scales no longer match the curvature directions, and the path creeps along the valley floor ($f=0.19$ after 40 steps, against $4\times10^{-7}$ in (b)).](../img/mdl-opt-per-coordinate.svg)
:label:`fig_mdl-opt-per-coordinate`

**Coordinate scaling removes ill-conditioning exactly when the coordinate axes
are the eigenvectors of the curvature.** It is a change of metric by a
diagonal matrix, and no diagonal matrix represents curvature along a
direction that mixes coordinates. The structured preconditioners at the end
of the section address this limit.

### A Convex Counterexample to Adam
:label:`subsec_mdl-adam-counterexample`

RMSProp and Adam forget old gradients; AdaGrad does not. Can forgetting make a
convex problem unsolvable? :citet:`Reddi.Kale.Kumar.2019` construct a
one-dimensional sequence of convex losses on which it does.

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
$\beta_2 = 1/(1 + C^2)$, steps $\eta_t = \eta/\sqrt{t}$ with
$\eta < \sqrt{1 - \beta_2}$, and projection of each iterate back onto
$[-1, 1]$, from $x_1 = 1$. Then $x_t > 0$ for every $t$ and $x_{3k+1} = 1$ for
every $k \ge 0$. Every period adds at least $2C - 4 > 0$ to the regret, so the
average regret does not tend to zero.* :cite:`Reddi.Kale.Kumar.2019`

The gradients of one period sum to $C - 2 > 0$, so the best fixed point is
$x^\star = -1$, while Adam returns to $x = +1$, the worst feasible point, at
the start of every period. The proof is an induction over periods. Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-reddi-by-hand` works through its
key step, where the hypothesis $\eta < \sqrt{1 - \beta_2}$ keeps the downward
step of each period from leaving $(0, 1]$. The authors note that the argument
also holds with bias correction and with a constant step. They give a
different periodic sequence for every $\beta_1 < \sqrt{\beta_2}$ and another
that defeats Adam for every $\epsilon > 0$.

The mechanism is visible in one period at $C = 4$, $\beta_2 = 1/17$, once
$v$ has settled into its periodic pattern. After the gradients $+4, -1, -1$,
$v$ takes the values $15.12$, $1.83$ and $1.05$, so $\sqrt{v}$ is $3.89$,
$1.35$ and $1.02$. The three displacements $-\eta g_t/\sqrt{v_t}$ are

$$
-1.03\,\eta, \qquad +0.74\,\eta, \qquad +0.98\,\eta,
$$

a net $+0.69\,\eta$ per period toward $+1$. Normalization gives the rare
gradient $C$ a displacement of about $\eta$, barely more than a gradient of size
$1$ receives, and $C^2$ leaves $v$ within two steps. Together the two $-1$
steps therefore move $x$ farther up than the one $+C$ step moves it down.
Plain SGD with step $\eta g_t/C$ moves by $-\eta$, $+0.25\,\eta$,
$+0.25\,\eta$, a net $-0.5\,\eta$ toward the optimum. The noise floor of
constant-step SGD (:numref:`sec_mdl-stochastic-gradient-methods`) shrinks with
the step and has no preferred direction. Adam's drift points toward $+1$ at
every step size.

The repair of :citet:`Reddi.Kale.Kumar.2019`, **AMSGrad**, divides by the
running maximum $\tilde{\mathbf{v}}_t = \max(\tilde{\mathbf{v}}_{t-1}, \mathbf{v}_t)$
of the uncorrected average instead of by $\mathbf{v}_t$. The per-coordinate
step $\eta_t/\sqrt{\tilde{v}_t}$ can then only fall, and the large gradient is
never forgotten. In the periodic regime $\tilde{v} = 15.12$ throughout, and
the displacements become $-1.03\,\eta$, $+0.26\,\eta$, $+0.26\,\eta$, a net
$-0.51\,\eta$. Where do Adam, AMSGrad and SGD end on this sequence? The cell
runs all three with the same $1/\sqrt{t}$ decay from $x = 0$, the middle of
the interval, passing each update as a function of the gradient, the average,
its running maximum and $C$.

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

The step constant `alpha` of the cell is the $\eta$ of the proposition, and
$\eta = 0.5 < \sqrt{16/17}$, so the step hypothesis holds. The cell starts at
$x = 0$ instead of $x = 1$, and Adam reaches $+1$ anyway. From
$t \approx 300$ on it returns to $+1$ in every period and never drops below
$0.97$, while AMSGrad and SGD approach the optimum $-1$ and agree to two
decimals. The construction shows that Adam has no general convergence
guarantee. It does not predict failure on ordinary training problems.

## Decoupled Weight Decay
:label:`subsec_mdl-decoupled-weight-decay`

### Coupled and Decoupled Decay

Nearly every training recipe adds weight decay, so an adaptive method has to
decide whether the decay term passes through its division.

Under SGD, weight decay has two equivalent implementations. Adding the penalty
$\tfrac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss adds $\lambda \mathbf{w}$ to
the gradient, and the update can be read as a gradient step followed by
shrinkage of the weights:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{w}_t)
= (1 - \eta\lambda)\, \mathbf{w}_t - \eta\, \mathbf{g}_t .
$$

Penalty and shrinkage are the same update. Through Adam they are not. Feed the
penalized gradient into :eqref:`eq_mdl-opt-adam` and look at the decay term
alone (take $\beta_1 = 0$ for clarity, so that the update is
$-\eta\,(\mathbf{g}_t + \lambda\mathbf{w}_t)/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$).
The shrinkage of coordinate $i$ in one step is

$$
\textrm{coupled ($\ell_2$ through Adam):}\quad
\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; w_{t,i}
\qquad \textrm{versus} \qquad
\textrm{decoupled:}\quad \eta\,\lambda\, w_{t,i} .
$$
:eqlabel:`eq_mdl-opt-adamw`

The preconditioner divides the penalty gradient as well as the loss gradient,
so the coupled shrinkage of a coordinate is inversely proportional to that
coordinate's gradient scale. A weight whose loss gradients are large or noisy
is barely decayed; a weight with small gradients is decayed strongly. For
gradient scales $\sigma = (10, 0.1)$, $\eta = 10^{-3}$ and $\lambda = 0.1$, the
per-step rates are $10^{-5}$ and $10^{-3}$, a factor of $100$ apart for the
same $\lambda$.

**AdamW** :cite:`Loshchilov.Hutter.2019` decouples the two: the loss gradient
goes through the preconditioner, the decay does not,

$$
\mathbf{w}_{t+1} = (1 - \eta\lambda)\, \mathbf{w}_t - \eta\; \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_mdl-opt-adamw-decoupled`

Every coordinate is shrunk by the same factor $1 - \eta\lambda$ per step, as
under SGD, and the moment estimates see only the loss gradient. Loshchilov and
Hutter attribute the gap in generalization between $\ell_2$-regularized Adam
and SGD with weight decay to the distortion of :eqref:`eq_mdl-opt-adamw`.

How much does each weight decay under the two rules when its loss gradient is
pure noise, so that decay is the only systematic force? The cell gives
$10^4$ weights the noise scale $\sigma = 10$ and $10^4$ others the scale
$\sigma = 0.1$, and starts all of them at $w = 1$. It runs 4,000 steps of
each method and prints the mean of $w$ over each group next to the
predictions.

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

Under coupled decay the noisy weights keep $0.96$ of their size, as
$e^{-\eta\lambda T/\sigma_1}$ predicts, while the quiet weights fall to about $0.03$,
against the prediction $e^{-\eta\lambda T/\sigma_2} = 0.018$. The quiet
weights decay at a rate near $10^{-3}$, but a little slower at first,
because at $w = 1$ their penalty gradient $\lambda w = 0.1$ is as large as
their noise and inflates their own $\hat{v}$. Under AdamW both groups end at
$0.67 = (1 - \eta\lambda)^{4000}$. For AdamW this prediction is exact in
expectation, because the normalized noise steps have mean zero and the decay is a fixed factor per
step.

:begin_tab:`pytorch`
`torch.optim.Adam(weight_decay=...)` adds $\lambda\mathbf{w}$ to the gradient:
it is the coupled rule. `torch.optim.AdamW` is the decoupled rule
:eqref:`eq_mdl-opt-adamw-decoupled`.
:end_tab:

:begin_tab:`mxnet`
`mx.optimizer.Adam(wd=...)` adds $\lambda\mathbf{w}$ to the gradient: it is the
coupled rule. `mx.optimizer.AdamW` applies the decay outside the
preconditioner.
:end_tab:

:begin_tab:`tensorflow`
In Keras, `weight_decay` is an argument of every optimizer and is applied to
the weights outside the preconditioner, so `keras.optimizers.Adam(weight_decay=...)`
and `keras.optimizers.AdamW` both implement the decoupled rule. The coupled rule
needs an explicit penalty in the loss.
:end_tab:

:begin_tab:`jax`
`optax.adam` has no weight-decay argument; `optax.adamw` is the decoupled rule.
The coupled rule needs an explicit penalty in the loss, or
`optax.chain(optax.add_decayed_weights(lam), optax.adam(eta))`, which adds
$\lambda\mathbf{w}$ to the gradient before Adam sees it.
:end_tab:

### Fixed Points

The cell above runs in a regime that never becomes stationary, because the
loss gradient is pure noise. What does
$\lambda$ mean at a fixed point of each method? The answer reverses the
conclusion of the previous subsection.

**Proposition (fixed points of coupled and decoupled decay).** *Let the
gradient $\mathbf{g}(\mathbf{w}) = \nabla\ell(\mathbf{w})$ of a loss $\ell$ be
exact (no noise), let $\epsilon > 0$, and call $\mathbf{w}$ a fixed point of an
update if the update vanishes there once the moment estimates have settled.
(a) Adam on $\ell + \tfrac{\lambda}{2}\|\mathbf{w}\|^2$ has a fixed point at
$\mathbf{w}$ if and only if $\nabla\ell(\mathbf{w}) + \lambda\mathbf{w} = \mathbf{0}$,
the stationary point of SGD on the same penalized loss. (b) AdamW has a fixed
point at $\mathbf{w}$ if and only if, for every coordinate $i$,*

$$
\lambda\, w_i \bigl(|g_i| + \epsilon\bigr) = -g_i ,
\qquad g_i = \partial_i \ell(\mathbf{w}).
$$
:eqlabel:`eq_mdl-opt-adamw-fixed-point`

*Every fixed point of AdamW therefore lies strictly inside the box
$|w_i| < 1/\lambda$, and as $\epsilon \to 0$ the condition becomes: either
$g_i = 0$, or $|w_i| = 1/\lambda$ with $w_i$ opposite in sign to $g_i$.*

**Proof.** At a fixed point the vector $\mathbf{u}$ that enters the moment
averages is constant, so $\hat{\mathbf{m}}_t \to \mathbf{u}$ and
$\hat{\mathbf{v}}_t \to \mathbf{u}^2$. (a) For coupled Adam
$\mathbf{u} = \mathbf{g} + \lambda\mathbf{w}$, and the update tends to
$-\eta\, u_i/(|u_i| + \epsilon)$ in coordinate $i$, which vanishes exactly
when $u_i = 0$. (b) For AdamW $\mathbf{u} = \mathbf{g}$, and the update of
coordinate $i$ tends to $-\eta\lambda w_i - \eta\, g_i/(|g_i| + \epsilon)$;
setting it to zero and multiplying by $(|g_i| + \epsilon)/\eta$ gives
:eqref:`eq_mdl-opt-adamw-fixed-point`. Taking absolute values,
$|w_i| = |g_i|/\bigl(\lambda(|g_i| + \epsilon)\bigr) < 1/\lambda$. As
$\epsilon \to 0$ the condition reads $\lambda w_i |g_i| = -g_i$, which holds
if $g_i = 0$ or if $\lambda w_i = -\mathrm{sign}(g_i)$. $\blacksquare$

Take the one-dimensional loss $\ell(w) = \tfrac12 (w - 3)^2$, whose minimizer
is $w = 3$. With $\lambda = \tfrac14$, coupled Adam settles at the penalized
minimizer $3/(1 + \lambda) = 2.4$, while AdamW settles at $3.0$: the box
$|w| \le 4$ does not bind, and the decay leaves no trace at the fixed point.
With $\lambda = 1$, coupled Adam settles at $1.5$ and AdamW at $1.0$, on the
edge of the box. (Running both methods with a cosine-decayed step reproduces
these four values to four decimals.)

The two readings of $\lambda$ are now separate. During training, in the noisy
regime of the cell above, AdamW applies the same per-step shrinkage to every
coordinate, and coupled Adam does not. At a fixed point the roles reverse:
coupled Adam keeps the maximum a posteriori (MAP) point of a Gaussian prior on
the weights (:numref:`sec_mdl-maximum_likelihood`), while AdamW turns
$\lambda$ into the radius $1/\lambda$ of an $\ell_\infty$ box
:cite:`Xie.Li.2024`. Exercise
:exref:`ex_mdl-adaptive-stochastic-methods-coupled-decoupled-fixed-points`
checks the proposition on a two-dimensional quadratic, and Exercise :exref:`ex_mdl-adaptive-stochastic-methods-adamw-linf` derives
the box from the conditions of constrained optimization.

## Warmup
:label:`subsec_mdl-schedules-warmup`

The preconditioner must act before it has data. At $t = 1$,
$\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$,
so the first step is $\eta\,\mathrm{sign}(\mathbf{g}_1)$ on every coordinate.
A coordinate whose gradient is $10^{-6}$ moves by $\eta$, just like one whose
gradient is $10^3$, on the evidence of a single sample (with
$\epsilon = 10^{-8}$ it moves by $0.99\,\eta$). **Warmup** ramps the step size
from about zero over the first steps of training
:cite:`Goyal.Dollar.Girshick.ea.2017`, which keeps these early updates small.

How large are Adam's first updates on gradients that are pure noise, and does
the estimated preconditioner make them large? The cell draws $2 \times 10^4$
independent gradient coordinates from $\mathcal{N}(0, 1)$ at every step, runs
Adam with the default parameters, and records the root-mean-square update in
units of $\eta$. A second curve replaces $\hat{\mathbf{v}}_t$ by the exact
variance $\sigma^2 = 1$, which removes the estimation of the preconditioner
altogether.

```{.python .input #mdl-adaptive-stochastic-methods-warmup}
n, T, beta1, beta2, eps = 20_000, 3000, 0.9, 0.999, 1e-8
m, v, rms_est, rms_exact = np.zeros(n), np.zeros(n), [], []
for t in range(1, T + 1):
    g = rng.standard_normal(n)                      # pure noise, sigma = 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1**t)
    rms_est.append(np.sqrt(np.mean((m_hat / (np.sqrt(v / (1 - beta2**t)) + eps))**2)))
    rms_exact.append(np.sqrt(np.mean(m_hat**2)))   # v_hat replaced by sigma^2 = 1
d2l.plot(np.arange(1, T + 1), [rms_est, rms_exact], 't', 'RMS update / eta',
         legend=['estimated v', 'exact variance'], xscale='log')
ts = (1, 2, 3, 10, 30, 100, 1000, 3000)
print('t:              ', ''.join(f'{t:>6d}' for t in ts))
print('estimated v:    ', ''.join(f'{rms_est[t - 1]:6.2f}' for t in ts))
print('exact variance: ', ''.join(f'{rms_exact[t - 1]:6.2f}' for t in ts))
```

The update starts at $1.00\,\eta$, is down to $0.33\,\eta$ at $t = 10$, and
settles within about fifty steps at
$\sqrt{(1 - \beta_1)/(1 + \beta_1)}\,\eta = 0.23\,\eta$, the stationary size
of an exponential average of independent unit noise. The curve with the exact
variance agrees with it to about $0.01$ at every $t$, the sampling noise of
$2 \times 10^4$ coordinates. The fourfold inflation of the first steps
therefore comes from the momentum average, which needs about
$1/(1 - \beta_1) = 10$ steps to average away the noise, and not from the
preconditioner. The preconditioner's own
effect at $t = 1$ is the equalization above: every coordinate moves by about
$\eta$, whatever its gradient scale (Exercise :exref:`ex_mdl-adaptive-stochastic-methods-first-steps-noise` shows this with unequal
scales). A linear ramp over the first tens of steps removes this transient.

The long warmups used in practice, thousands of steps, rest on a second
mechanism that this noise model does not contain. Right after initialization
the curvature of a network can change quickly, so a step that is stable later
in training may be unstable at first. :numref:`sec_scheduler` measures this in
a real model.

## Structured Preconditioners
:label:`subsec_mdl-preconditioning-ladder`

### Kronecker Factors

The rotated valley defeated every diagonal preconditioner. Every method of
this section has the form
$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{P}_t^{-1} \mathbf{g}_t$ for
a positive definite $\mathbf{P}_t$: gradient descent takes
$\mathbf{P} = \mathbf{I}$, Newton's method the Hessian $\nabla^2 f$, and Adam
a diagonal estimated from gradients. A full matrix captures the curvature in
every direction, but it is too large to store. One layer with a $4096 \times 4096$ weight matrix has
$1.7 \times 10^7$ weights, and its curvature matrix has $2.8 \times 10^{14}$
entries, about 2 petabytes in double precision.

The structural fact that helps is that parameters come in matrices. Take a
linear layer $\mathbf{y} = \mathbf{W}\mathbf{a}$ with
$\mathbf{W} \in \mathbb{R}^{m \times n}$, input $\mathbf{a} \in \mathbb{R}^n$
and output gradient $\boldsymbol{\delta} = \partial\ell/\partial\mathbf{y} \in \mathbb{R}^m$.
Its gradient is the outer product $\boldsymbol{\delta}\mathbf{a}^\top$. Write
$\mathrm{vec}(\mathbf{V})$ for the vector that stacks the columns of a matrix
$\mathbf{V}$; then $\mathrm{vec}(\boldsymbol{\delta}\mathbf{a}^\top) = \mathbf{a} \otimes \boldsymbol{\delta}$,
where the **Kronecker product** $\mathbf{B} \otimes \mathbf{D}$ is the block
matrix whose $(i, j)$ block is $b_{ij}\mathbf{D}$.

The outer-product curvature of the layer, the block of the Fisher information matrix
$\mathbb{E}[\nabla\log p\,\nabla\log p^\top]$ of
:numref:`sec_mdl-maximum_likelihood` that belongs to $\mathbf{W}$, is
$\mathbb{E}\bigl[(\mathbf{a}\mathbf{a}^\top) \otimes (\boldsymbol{\delta}\boldsymbol{\delta}^\top)\bigr]$.
**K-FAC** :cite:`Martens.Grosse.2015` approximates it by
$\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}$ with
$\mathbf{S}_{\mathrm{in}} = \mathbb{E}[\mathbf{a}\mathbf{a}^\top]$
($n \times n$) and $\mathbf{S}_{\mathrm{out}} = \mathbb{E}[\boldsymbol{\delta}\boldsymbol{\delta}^\top]$
($m \times m$). The approximation is exact when inputs and output gradients
are independent, which they are not in a trained network.

The factored form is useful because its inverse acts on the gradient matrix
directly. For symmetric invertible factors,

$$
\bigl(\mathbf{S}_{\mathrm{in}} \otimes \mathbf{S}_{\mathrm{out}}\bigr)^{-1} \mathrm{vec}(\mathbf{V})
= \mathrm{vec}\bigl(\mathbf{S}_{\mathrm{out}}^{-1}\, \mathbf{V}\, \mathbf{S}_{\mathrm{in}}^{-1}\bigr).
$$
:eqlabel:`eq_mdl-opt-kronecker`

To see it, write column $j$ of a product $\mathbf{D}\mathbf{V}\mathbf{B}$ as
$\sum_k b_{kj}\, \mathbf{D}\mathbf{v}_k$, where $\mathbf{v}_k$ is column $k$
of $\mathbf{V}$; stacked, these columns are the blocks of
$(\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$, so
$\mathrm{vec}(\mathbf{D}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\top \otimes \mathbf{D})\,\mathrm{vec}(\mathbf{V})$.
Since $(\mathbf{B} \otimes \mathbf{D})^{-1} = \mathbf{B}^{-1} \otimes \mathbf{D}^{-1}$
and the factors are symmetric, taking $\mathbf{B} = \mathbf{S}_{\mathrm{in}}^{-1}$
and $\mathbf{D} = \mathbf{S}_{\mathrm{out}}^{-1}$ gives
:eqref:`eq_mdl-opt-kronecker`. A curvature with $(mn)^2$ entries is applied
through two inverses of sizes $n$ and $m$. The cell checks the identity on a
$3 \times 2$ matrix and counts what it saves for $m = n = 4096$.

```{.python .input #mdl-adaptive-stochastic-methods-kronecker-factors}
def spd(k, rng):
    """A random k x k symmetric positive definite matrix."""
    B = rng.standard_normal((k, k))
    return B @ B.T + k * np.eye(k)

S_in, S_out, V = spd(2, rng), spd(3, rng), rng.standard_normal((3, 2))  # n=2, m=3
full = np.linalg.solve(np.kron(S_in, S_out), V.flatten('F'))   # vec: stack columns
two = (np.linalg.inv(S_out) @ V @ np.linalg.inv(S_in)).flatten('F')
print(f'max difference: {np.abs(full - two).max():.1e}')
m = n = 4096
print(f'stored entries: full {(m * n)**2:.1e}, two factors {m**2 + n**2:.1e}')
print(f'flops to invert: full {(m * n)**3:.1e}, two factors {m**3 + n**3:.1e}')
```

The two sides agree to rounding error. The factors need $3.4 \times 10^7$
stored numbers instead of $2.8 \times 10^{14}$, and inverting them costs about
$1.4 \times 10^{11}$ operations instead of $4.7 \times 10^{21}$.

### K-FAC, Shampoo and Muon

The Kronecker identity makes a preconditioner per layer affordable. It leaves
open which matrix to factor and which power of it to apply, and the methods
in current use differ in these two choices.

Preconditioning by the Fisher matrix is called the **natural gradient**
:cite:`Amari.1998`. It is steepest descent when the length of a step is
measured by how much it changes the distribution that the parameters define:
the metric view of AdaGrad, with the Fisher matrix as the metric. For softmax cross-entropy and squared loss, the Fisher (with labels drawn
from the model) equals the Gauss–Newton matrix, the positive semidefinite part
of the Hessian, so for these losses K-FAC approximates the curvature itself.

Three methods share the two-sided form $\mathbf{G} \mapsto \mathbf{P}_{\mathrm{l}}^{-a}\mathbf{G}\,\mathbf{P}_{\mathrm{r}}^{-a}$
on a layer's gradient matrix $\mathbf{G}$ (bold, to distinguish it from the
gradient bound $G$ of the regret analysis), and they differ in where the
factors come from. K-FAC takes them from the second moments of the inputs and
output gradients, with $a = 1$. **Shampoo** :cite:`Gupta.Koren.Singer.2018`
builds them AdaGrad-style from the accumulated gradients,
$\mathbf{P}_{\mathrm{l}} = \sum_t \mathbf{G}_t\mathbf{G}_t^\top$ and
$\mathbf{P}_{\mathrm{r}} = \sum_t \mathbf{G}_t^\top\mathbf{G}_t$, with
$a = \tfrac14$. **Muon** :cite:`Jordan.Jin.Boza.ea.2024` is the case without
memory: with the current matrix alone, and
$\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ square and
invertible,

$$
\bigl(\mathbf{G}\mathbf{G}^\top\bigr)^{-1/4}\, \mathbf{G}\, \bigl(\mathbf{G}^\top\mathbf{G}\bigr)^{-1/4}
= \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\, \boldsymbol{\Sigma}\, \boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top
= \mathbf{U}\mathbf{V}^\top ,
$$

the polar factor of $\mathbf{G}$. It keeps the singular vectors of
$\mathbf{G}$ and sets every singular value to one. Muon applies it to the
momentum matrix and computes it by the Newton–Schulz iteration, which reaches
the polar factor with matrix products alone (:numref:`sec_mdl-svd-low-rank`).
Muon thus equalizes the singular directions of the update where Adam
equalizes its coordinates. Unlike Adam's sign-like normalization, it commutes
with rotations of the rows and columns (Exercise :exref:`ex_mdl-adaptive-stochastic-methods-rotation-sensitivity`d).

The three methods differ in cost. Muon keeps no second-moment state and needs
a few matrix products per step. K-FAC and Shampoo invert or take roots of
$m \times m$ and $n \times n$ factors, usually every few steps. None of them
reaches Newton's method, which would use the full curvature matrix.

We can now derive a coordinate's step from its gradient history, prove that
the result ignores the coordinate's scale, read off the fixed points of
coupled and decoupled decay, and apply a layer's curvature through two small
inverses. All of these results assume curvature aligned with the coordinates,
or with the Kronecker structure of a layer. The curvature of a deep network is
neither diagonal nor fixed, so why these methods work as well as they do on
deep networks is an open question. :numref:`sec_mdl-constrained-optimization-duality` asks what
changes when the variables must stay in a set.

## Summary

A single step size is set by the stiffest direction, so gradient descent needs
6,160 steps on a $1000{:}1$ valley. Dividing each coordinate's step by the root
mean square of its own gradients removes the condition number when the
curvature is aligned with the coordinates. AdaGrad sums the squared gradients,
RMSProp averages them, and Adam adds momentum and a bias correction that
cancels the zero start in expectation. With $\epsilon = 0$ Adam is blind to
each coordinate's scale and needs 48 steps on that valley. The same optimizer
never converges on the valley rotated by $45^\circ$, and a periodic convex
sequence drives it to the worst feasible point, because it forgets a rare
large gradient. Through Adam, an $\ell_2$ penalty decays coordinates at rates set by
their gradient scales but keeps the MAP fixed point. AdamW's decoupled decay
is uniform per step and turns $\lambda$ into the radius of an $\ell_\infty$
box at its fixed points. Kronecker-factored preconditioners
apply a layer's curvature through two small inverses, and Muon equalizes the
singular directions of a layer's update.

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
       $\alpha \approx 1$, and your ratio there is about $0.64$. How much of
       the factor $\sqrt{d}$ of :numref:`subsec_mdl-per-coordinate` does a
       bag-of-words model obtain? Which distribution of gradient mass would
       obtain all of it?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-concentrated-gradients`

1. **Bias correction and step inflation.** Let
   $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$
   with $\mathbf{m}_0 = \mathbf{0}$.
    1. Repeat the computation of :eqref:`eq_mdl-opt-bias-correction` under
       $\mathbb{E}[\mathbf{g}_\tau] = \bar{\mathbf{g}}$ and show
       $\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1^t)\,\bar{\mathbf{g}}$.
    1. Without both corrections, Adam scales step $t$ by the factor
       $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$
       of :numref:`fig_mdl-opt-bias-correction`. For $t \ll 1/(1-\beta_2)$,
       replace $1-\beta_2^t$ by $t\,(1-\beta_2)$. Show that the maximizer
       of the approximate factor satisfies $e^u = 1 + 2u$ with
       $u = -t\log\beta_1$, and solve this equation numerically.
    1. Express the location and the height of the peak through
       $1-\beta_1$ and $1-\beta_2$. How does each of the two depend on
       $\beta_1$ and on $\beta_2$? What does the result imply for running
       uncorrected Adam with a longer second-moment memory?
    1. Compare your approximation with the exact factor for
       $\beta_1 \in \{0.9, 0.99\}$ and
       $\beta_2 \in \{0.99, 0.999, 0.9999\}$. For which pair does the
       approximation fail, and why?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-bias-correction-first-moment`

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

1. [code] **Where $\epsilon$ matters.** Use `run` and `adam` of the
   `#adaptive-stochastic-methods-adam-vs-gd` cell with $\beta_1 = 0$ and
   $\eta = 0.1$ on the aligned valley.
    1. Sweep
       $\epsilon \in \{10^{-12}, 10^{-8}, 10^{-4}, 10^{-2}, 10^{-1}, 1\}$,
       record the step at which $f < 10^{-8}$, and print
       $\sqrt{\hat{\mathbf{v}}_t}$ at that step (let `run` also return the
       state). Explain the pattern of step counts with the printed values.
    1. Multiply both curvatures `h` and the tolerance by a common scale
       $s \in \{1, 10^{-4}, 10^{-8}, 10^{-10}\}$, raise the cap `K` to
       $10^5$, and compare $\epsilon = 0$ with $\epsilon = 10^{-8}$. Which
       statement of the scale-invariance proposition does $\epsilon$ break,
       and below which gradient scale?
    1. In a deep network, some layers receive gradients several orders of
       magnitude smaller than others. What does the gradient scale you found
       imply for the updates of those layers under the default $\epsilon$?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-epsilon`

1. [code] **Adam's first steps on noise.** Let every coordinate of the
   gradient be i.i.d. zero-mean noise with variance $\sigma^2$, as for a
   parameter that the loss does not yet constrain.
    1. Show that the bias-corrected first moment has variance

       $$\textrm{Var}(\hat{m}_t) = \sigma^2\, \frac{(1-\beta_1)(1+\beta_1^t)}{(1+\beta_1)(1-\beta_1^t)}.$$

       If $\hat{v}_t$ were replaced by the exact $\sigma^2$, what would the
       root-mean-square update be, in units of $\eta$, at $t = 1$ and as
       $t \to \infty$ at the default $\beta_1$? Compare with the warmup cell.
    1. The warmup cell gives every coordinate the same scale, which hides
       what the preconditioner does. Draw the scales $\sigma_i$
       log-uniformly from $[10^{-3}, 1]$ instead, and compare, across
       coordinates, the root-mean-square displacement
       $\eta\,\hat{m}_{t,i}/\sqrt{\hat{v}_{t,i}}$ at $t = 1$ and $t = 1000$
       with that of SGD, $\eta\, g_{t,i}$. Which optimizer moves the
       small-scale coordinates as far as the large-scale ones, from which
       step on, and is that what you want for a coordinate that the loss does
       not constrain?
    1. Which timescale, $1/(1-\beta_1)$ or $1/(1-\beta_2)$, governs the
       transient of the warmup cell? Relate the answer to the two reasons for
       warmup given in :numref:`subsec_mdl-schedules-warmup`.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-first-steps-noise`

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
   factories of the `#adaptive-stochastic-methods-adam-vs-gd` cell.
    1. Show that the iterates of gradient descent and of heavy ball on
       $\mathbf{A}_\theta$ from $\mathbf{R}_\theta(1, 1)^\top$ are
       $\mathbf{R}_\theta$ times their iterates on the aligned valley, so
       that their step counts do not depend on $\theta$.
    1. Tune $\eta$ over $\{0.03, 0.1, 0.3, 1, 3\}$ at $\theta = \pi/4$ for
       $\beta_1 = 0.9$ and for $\beta_1 = 0$. How much of Adam's advantage
       over gradient descent survives the rotation, and which component of
       Adam supplies it? For $\beta_1 = 0$, print $f$ after $40{,}000$ steps
       and explain the value $250\,\eta^2$.
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

1. [code] **Coupled versus decoupled fixed points.** ● Let
   $\ell(\mathbf{w}) = \tfrac12 (\mathbf{w} - \mathbf{w}_0)^\top
   \mathrm{diag}(h_1, h_2) (\mathbf{w} - \mathbf{w}_0)$ with $h = (100, 1)$,
   $\mathbf{w}_0 = (1, 3)$, and exact gradients.
    1. For $\lambda = 0.1$, compute the fixed point of coupled Adam from the
       proposition of :numref:`subsec_mdl-decoupled-weight-decay`.
    1. For $\lambda = 0.1$ and $\epsilon = 10^{-8}$, solve the AdamW condition
       :eqref:`eq_mdl-opt-adamw-fixed-point` and show that the solution
       differs from $\mathbf{w}_0$ by less than $10^{-8}$. What does weight
       decay contribute at this fixed point, and why?
    1. Choose $\lambda$ so that the box $|w_i| \le 1/\lambda$ binds for
       coordinate 2 but not for coordinate 1. Predict both fixed points, and
       confirm all four predictions of parts (a)–(c) by running each method
       for $2 \times 10^5$ steps, with $\eta$ decaying from $0.01$ to $0$ by
       a cosine schedule.
    1. Reconcile these fixed points with the `decay` cell. In the
       noise-dominated regime, which of the two updates applies the same
       per-step shrinkage to every coordinate, and why is that the property
       that matters during training?
   :exlabel:`ex_mdl-adaptive-stochastic-methods-coupled-decoupled-fixed-points`

1. [extended] **AdamW's implicit $\ell_\infty$ constraint.** ●● The
   proposition of :numref:`subsec_mdl-decoupled-weight-decay` found that the
   fixed points of AdamW lie in the box $|w_i| < 1/\lambda$ and, as
   $\epsilon \to 0$, on its boundary wherever the gradient does not vanish.
   :citet:`Xie.Li.2024` prove that if full-batch AdamW with
   $\beta_1 \le \beta_2$ and a nonincreasing step schedule with divergent
   sum converges, its limit is a KKT point of
   $\min_{\mathbf{w}} \ell(\mathbf{w})$ subject to
   $\|\mathbf{w}\|_\infty \le 1/\lambda$.
    1. A feasible $\mathbf{w}$ is a KKT point of this problem when every
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
[Dive into Deep Learning · §26.4]{.kicker}

Adaptive Methods<br>**one step size per coordinate · scaling and its limits · AdamW · warmup · Kronecker factors**
:::
:::

::: {.slide title="A 1000:1 Valley Costs Gradient Descent 6,160 Steps"}
[The problem]{.kicker}

::: {.cols .vc}
::: {.col}
$f=\tfrac12(x_1^2+1000\,x_2^2)$ from $(1,1)$, until $f<10^{-8}$:

- best single step $2/1001$: **6,160** steps
- heavy ball, tuned: **296** steps
- one step per coordinate, learned from gradients: **48** steps
- the same valley rotated by $45^\circ$: **never**
:::

::: {.col .fig}
@fig:mdl-opt-per-coordinate
:::
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive AdaGrad, RMSProp and Adam from one idea, and Adam's bias correction;
- prove that Adam cannot see a coordinate's scale, and say when that helps;
- follow a convex sequence on which Adam ends at the worst point;
- tell coupled from decoupled weight decay and find their fixed points;
- apply a layer's curvature through two small inverses.

**Running example:** $\tfrac12(x_1^2+1000\,x_2^2)$, aligned and rotated.

**The idea:** learn one step per coordinate from past gradient sizes.
:::

::: {.slide title="Adaptive Methods Rescale the Gradient Coordinate by Coordinate"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gradient Descent and Curvature | safe steps, $\kappa$, momentum, Newton |
| Convexity | local = global, rates, Jensen |
| Stochastic Gradient Methods | noise floors, step decay |
| **Adaptive Methods** | **per-coordinate steps** |
| Constrained Optimization | KKT, projections, prox |
| Duality | lower bounds, prices |
| Numerical Stability and Conditioning | rounding, conditioning |
:::

::: {.col .narrow}
- **Uses:** $\kappa$, heavy ball, the noise floor.
- **Skips:** schedule shapes, constraints.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Per-coordinate step sizes]{.dtitle}

[AdaGrad → RMSProp → Adam]{.dsub}
:::
:::

::: {.slide title="Separate Steps Solve a Diagonal Quadratic in One Step"}
$f=\tfrac12(h_1x_1^2+h_2x_2^2)$, $\;\partial_if=h_ix_i$, $\;h_i$ = curvature

- step $\eta_i$ per coordinate: $x_i\leftarrow(1-\eta_ih_i)\,x_i$
- $\eta_i=1/h_i$: both coordinates reach $0$ at once (diagonal Newton)
- $h=(1,1000)$: steps $1$ and $10^{-3}$; $\kappa$ is gone

::: {.d2l-note .warn}
In training, $h_i$ is unknown and changes, and gradients are noisy. Use
$|g_i|=h_i|x_i|$ as the stand-in for $h_i$.
:::
:::

::: {.slide title="AdaGrad Decays Each Coordinate's Step with Its Own Activity"}
$$\mathbf s_t=\mathbf s_{t-1}+\mathbf g_t\odot\mathbf g_t,\qquad
\mathbf x_{t+1}=\mathbf x_t-\frac{\eta}{\sqrt{\mathbf s_t}+\epsilon}\,\mathbf g_t$$

- coordinate 1: gradient $\pm1$ every step; coordinate 2: once per $10^4$ steps
- at $t=10^4$: $\mathbf s=(10^4,1)$, steps $\eta/100$ and $\eta$
- a global $\eta/\sqrt t$: $\eta/100$ for both

::: {.d2l-note}
Regret up to a constant: $D_\infty\sum_i\sqrt{\sum_tg_{t,i}^2}$, up to $\sqrt d$
below a global step's when the gradient mass is concentrated.
:::
:::

::: {.slide title="AdaGrad's Built-In 1/√t Decay Is Not Robbins–Monro"}
- persistent noise $\sigma^2$: $s_{t,i}\approx\sigma^2t$, step $\approx\eta/(\sigma\sqrt t)$
- $\sum_t1/\sqrt t=\infty$, but also $\sum_t1/t=\infty$: not square-summable
- steps shrink whether or not the loss still falls

**RMSProp:** average instead of sum,
$\mathbf v_t=\beta_2\mathbf v_{t-1}+(1-\beta_2)\,\mathbf g_t^2$. Weights halve
every $\log 2/(1-\beta_2)$ steps: **7** at $\beta_2=0.9$, **693** at $0.999$.
:::

::: {.slide title="Adam Divides a Momentum Average by the RMS Gradient"}
$$\mathbf m_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t,\quad
\mathbf v_t=\beta_2\mathbf v_{t-1}+(1-\beta_2)\mathbf g_t^2$$

$$\mathbf x_{t+1}=\mathbf x_t-\eta\,\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon},\qquad
\hat{\mathbf m}_t=\frac{\mathbf m_t}{1-\beta_1^t},\ \ \hat{\mathbf v}_t=\frac{\mathbf v_t}{1-\beta_2^t}$$

- defaults $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\mathbf m_0=\mathbf v_0=\mathbf 0$
- at $t=1$: $\hat{\mathbf m}_1=\mathbf g_1$, $\hat{\mathbf v}_1=\mathbf g_1^2$, so the step is $\eta\,\mathrm{sign}(\mathbf g_1)$
:::

::: {.slide title="Zero Initialization Leaves v at the Fraction 1 − β₂ᵗ of Its Scale"}
::: {.cols .vc}
::: {.col .narrow}
Unroll and assume $\mathbb{E}[\mathbf g_\tau^2]=\overline{\mathbf g^2}$:
$$\mathbb{E}[\mathbf v_t]=(1-\beta_2^t)\,\overline{\mathbf g^2}$$

- $1\%$ at $t=10$, $63\%$ at $t=1000$
- uncorrected update off by $3.16\times$ at $t=1$, $6.6\times$ at $t=12$
:::

::: {.col .fig}
@fig:mdl-opt-bias-correction
:::
:::
:::

::: {.slide title="Quick Check: Adam's First Step and Its Memory"}
[Check]{.kicker}

1. At $t=1$, how far does Adam move a coordinate whose gradient is $10^{-6}$?
2. After how many steps has $\beta_2=0.999$ halved the weight of a gradient?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) About $\eta$: the first step is $\eta\,\mathrm{sign}(g_1)$
($0.99\,\eta$ at $\epsilon=10^{-8}$). (2) $\log 2/(1-\beta_2)\approx693$ steps.
:::
:::
:::

::: {.slide title="With ε = 0, Adam Cannot See a Coordinate's Scale"}
**Claim.** If $\nabla f'=\mathbf C\,\nabla f$ with $\mathbf C=\mathrm{diag}(c_i)$,
$c_i>0$, Adam takes the same steps on $f$ and $f'$.

**Proof.** If the iterates agree so far, then $m'_i=c_im_i$ and
$v'_i=c_i^2v_i$, so $\hat m'_i/\sqrt{\hat v'_i}=\hat m_i/\sqrt{\hat v_i}$.

::: {.d2l-note .rule}
$\nabla\tfrac12\sum_ih_ix_i^2=\mathrm{diag}(h)\,\mathbf x$: Adam on the
$1000{:}1$ valley follows Adam on the round bowl, step for step.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Scaling and its limits]{.dtitle}

[the valley, rotated · a convex counterexample]{.dsub}
:::
:::

::: {.slide title="Scaling Alone Solves the Aligned Valley in 48 Steps, the Rotated One Never"}
Rotated: $\mathbf A=\mathbf R\,\mathrm{diag}(1,1000)\,\mathbf R^\top$, start
$\mathbf R(1,1)^\top$, $\mathbf R$ = rotation by $45^\circ$.

@!mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-1

::: {.d2l-note}
GD and heavy ball ignore the rotation. Rotated, the stiff mode dominates both
coordinates, the two scales become equal, and scaling-only Adam ends in a
sign-step two-cycle across the valley with $f=250\,\eta^2$.
:::
:::

::: {.slide title="The Stiff Coordinate Gets a Thousandfold Smaller Step from Step One"}
Adam with $\beta_1=0$, $\eta=0.1$, $\epsilon=0$ on the aligned valley:

@!mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-2

::: {.d2l-note}
$x_1=x_2$ at every step, as the scale-invariance claim predicts. The
contraction factor $1-\eta/\sqrt{\hat v_t}$ goes from $0.9$ to $0.67$.
:::
:::

::: {.slide title="A Repeating Triple of Convex Losses Drives Adam to the Worst Point"}
$x\in[-1,1]$; losses $4x,\,-x,\,-x,\,4x,\ldots$; best point $x^\star=-1$.
Adam with $\beta_1=0$, $\beta_2=1/17$:

- $v$ per period: $15.1,\ 1.83,\ 1.05$
- moves $-\eta g/\sqrt v$: $-1.03\eta,\ +0.74\eta,\ +0.98\eta$, net $+0.69\eta$
- SGD $-\eta g/4$: $-\eta,\ +0.25\eta,\ +0.25\eta$, net $-0.5\eta$

::: {.d2l-note .warn}
Normalization caps the large gradient's move at about $\eta$, and $v$ forgets
it in two steps: two $-1$ steps outweigh one $+4$.
:::

[Proof setting: no bias correction, $\eta_t=\eta/\sqrt t$, $\eta<\sqrt{1-\beta_2}$.]{.small}
:::

::: {.slide title="AMSGrad's Running Maximum Never Forgets the Large Gradient"}
$\tilde v_t=\max(\tilde v_{t-1},v_t)=15.12$: moves $-1.03\eta,\ +0.26\eta,\
+0.26\eta$, net $-0.51\eta$. From $x=0$, step $0.5/\sqrt t$:

@!adaptive-stochastic-methods-reddi

::: {.d2l-note}
An existence result: Adam has no general convergence guarantee. It does not
predict failure on ordinary training problems.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Weight decay, warmup, beyond the diagonal]{.dtitle}

[coupled and decoupled decay · the first steps · Kronecker factors]{.dsub}
:::
:::

::: {.slide title="Through Adam, One λ Decays Two Weights at Rates 100× Apart"}
Shrinkage of $w_i$ per step ($\lambda$ = weight decay):
$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat v_{t,i}}+\epsilon}\,w_{t,i}}_{\text{coupled}}
\qquad\text{vs.}\qquad
\underbrace{\eta\lambda\,w_{t,i}}_{\text{AdamW}}$$

Pure-noise gradients at scales $\sigma=(10,\,0.1)$, $10^4$ weights each:

@!adaptive-stochastic-methods-adamw
:::

::: {.slide title="Coupled Adam Keeps the MAP Point, AdamW Settles on the Box |wᵢ| = 1/λ"}
Fixed points, exact gradient $g_i=\partial_i\ell$:

- coupled: $\nabla\ell+\lambda\mathbf w=\mathbf 0$
- AdamW: $\lambda w_i(|g_i|+\epsilon)=-g_i$; as $\epsilon\to0$: $g_i=0$ or $|w_i|=1/\lambda$
- $\ell=\tfrac12(w-3)^2$, $\lambda=\tfrac14$: $2.4$ vs. $3.0$; $\lambda=1$: $1.5$ vs. $1.0$

[Coupled: PyTorch `Adam(weight_decay)`, MXNet `Adam(wd)`. Decoupled: every
`AdamW`, Keras `Adam(weight_decay)`. Optax `adam`: no decay.]{.small}
:::

::: {.slide title="Adam's First Steps Are Inflated by Momentum, Not by the Preconditioner"}
RMS update on pure noise, in units of $\eta$:

@!mdl-adaptive-stochastic-methods-warmup

::: {.d2l-note}
$1.00\to0.23=\sqrt{(1-\beta_1)/(1+\beta_1)}$ within tens of steps.
:::
:::

::: {.slide title="Quick Check: Fixed Points and Rotations"}
[Check]{.kicker}

1. $\ell=\tfrac12(w-3)^2$, $\lambda=\tfrac12$: fixed points of coupled Adam and of AdamW?
2. Which rows of the step-count table do not change when the valley is rotated?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) Both $2$: coupled $3/1.5$, AdamW on the box $1/\lambda=2$, a
coincidence of this $\lambda$. (2) Gradient descent and heavy ball.
:::
:::
:::

::: {.slide title="Kronecker Factors Replace a 10¹⁴-Entry Inverse by Two 4096² Inverses"}
Layer $\mathbf y=\mathbf W\mathbf a$, gradient $\boldsymbol\delta\mathbf a^\top$;
curvature $\approx\mathbf S_{\rm in}\otimes\mathbf S_{\rm out}$,
$\mathbf S_{\rm in}=\mathbb{E}[\mathbf a\mathbf a^\top]$,
$\mathbf S_{\rm out}=\mathbb{E}[\boldsymbol\delta\boldsymbol\delta^\top]$:

$$(\mathbf S_{\rm in}\otimes\mathbf S_{\rm out})^{-1}\mathrm{vec}\,\mathbf V
=\mathrm{vec}(\mathbf S_{\rm out}^{-1}\mathbf V\,\mathbf S_{\rm in}^{-1})$$

@!mdl-adaptive-stochastic-methods-kronecker-factors
:::

::: {.slide title="K-FAC, Shampoo and Muon Differ in Where the Factors Come From"}
Two-sided step $\mathbf P_{\rm l}^{-a}\,\mathbf G\,\mathbf P_{\rm r}^{-a}$ on a
layer's gradient matrix $\mathbf G=\mathbf U\boldsymbol\Sigma\mathbf V^\top$:

- **K-FAC** ($a=1$): moments of inputs and output gradients
- **Shampoo** ($a=\tfrac14$): accumulated $\mathbf G\mathbf G^\top$, $\mathbf G^\top\mathbf G$
- **Muon**: the current $\mathbf G$ alone, giving $\mathbf U\mathbf V^\top$
- **Newton**: the full matrix, out of reach

::: {.d2l-note}
Muon is the cheapest: no statistics, a few matrix products per step.
:::
:::

::: {.slide title="Scaling Removes κ Exactly When the Axes Are the Eigenvectors"}
[Wrap-up]{.kicker}

- valley $1000{:}1$: GD 6,160, heavy ball 296, scaling alone 48; rotated: never
- Adam with $\epsilon=0$ is blind to each coordinate's scale
- periodic convex losses: Adam net $+0.69\eta$ per period, AMSGrad $-0.51\eta$
- one $\lambda$: coupled rates $10^{-5}$ and $10^{-3}$; AdamW $10^{-4}$ for all
- fixed points: coupled keeps MAP, AdamW the box $|w_i|=1/\lambda$
- first step $\eta\,\mathrm{sign}(\mathbf g_1)$; RMS update $1.00\to0.23$
- two $4096^2$ factors instead of $2.8\times10^{14}$ entries

**Next:** optimization when the variables must stay in a set.
:::
