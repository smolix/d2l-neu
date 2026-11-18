# Momentum
:label:`sec_momentum`

Several of the convolutional networks in this book were trained by SGD with
`momentum=0.9` (:numref:`sec_training_recipes`). AdamW, which trained
`TinyLM` in the previous section, averages its gradients over steps, and
that averaging is momentum again. Almost every optimizer in deep learning
has it built in. We have used the setting without asking what it does.

Momentum answers the weakness of gradient descent that the valley of
:numref:`sec_optimization-intro` exposed. The steep direction of the valley
sets a ceiling on the learning rate, and below the ceiling the flat
direction shrinks by less than a tenth per step. Even the best learning
rate needed 47 steps to bring both coordinates within 1% of the minimum,
and the count grows in proportion to the condition number, the ratio of
the largest curvature to the smallest.

Gradient descent is this slow because it looks only at the current
gradient. The gradients of the last few steps say more. Along the floor of
the valley they point the same way, step after step; across the valley
they alternate in sign, because each step overshoots the opposite wall.
:citet:`Polyak.1964` proposed to step along a running sum of past
gradients, in which old gradients fade, instead of along the latest one.
The sum grows along the floor and cancels across the walls, much as a
heavy ball rolling down the valley gathers speed along it while its swings
from wall to wall die out. The method is called *momentum*.

With tuned settings momentum brings the valley within 1% in 18 steps
instead of 47. On any quadratic it needs a number of steps proportional to
the square root of the condition number rather than to the condition
number itself. All it asks for is one extra buffer per parameter and one
more hyperparameter.

We derive the method on the valley and learn to read its hyperparameter in
two ways: as the length of a memory and as a multiplier of the learning
rate. The valley has an exact gradient, however, and a network has one
estimated from a minibatch. We therefore go on to Fashion-MNIST and train
two models there, a linear one on all the data at once and the
`FashionCNN` of :numref:`sec_test_models` on small minibatches, to find
out where the saving survives.

```{.python .input #momentum}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #momentum}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #momentum-1}
%%tab pytorch
import random
```

## The Momentum Method

Let us first watch gradient descent fail in the way that momentum is meant
to repair. A learning rate above the ceiling of $0.5$ would be welcome
along the floor of the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$: at
$\eta = 0.6$ a step multiplies $x_1$ by $1 - 0.6 \cdot 0.2 = 0.88$. Across
the valley the same step multiplies $x_2$ by $1 - 0.6 \cdot 4 = -1.4$. We
run it with the helpers of :numref:`sec_gd`: `d2l.train_2d` applies an
update rule for 20 steps from $(-5, -2)$ and prints where it ends, and
`d2l.gd_2d` is the rule of gradient descent.

```{.python .input #momentum-an-ill-conditioned-problem-2}
%%tab pytorch, jax
h1, h2 = d2l.VALLEY_CURVATURES  # Curvatures of the valley along x1 and x2
f_valley, valley_grad = d2l.f_valley, d2l.valley_grad
d2l.train_2d(d2l.gd_2d, valley_grad, eta=0.6);
```

After 20 steps $x_1$ stands at $-0.39$, closer to the minimum than any
learning rate below the ceiling could have brought it, and $x_2$ at about
$-1673$. The run fails in one coordinate only, and there the gradient
changes sign at every step.

### The Velocity

A minibatch averages gradients over examples (:numref:`sec_sgd`). Momentum
goes on to add them up over time, in a memory that leaks: at every step it
keeps a fraction $\beta$ of what it holds and adds the new gradient. The
memory is called the *velocity* $\mathbf{v}_t$, and the update steps along
it instead of along the gradient,

$$
\begin{aligned}
\mathbf{v}_t &= \beta\, \mathbf{v}_{t-1} + \mathbf{g}_t, \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta\, \mathbf{v}_t.
\end{aligned}
$$
:eqlabel:`eq_momentum`

Here $\mathbf{g}_t$ is the gradient at $\mathbf{x}_t$, exact or estimated
from a minibatch, the velocity starts at $\mathbf{v}_0 = \mathbf{0}$, and
the hyperparameter $\beta \in [0, 1)$ is itself called the momentum. For
$\beta = 0$ the update is gradient descent. Inserting the first line into
itself, again and again, shows what the velocity holds:

$$\mathbf{v}_t = \mathbf{g}_t + \beta\, \mathbf{g}_{t-1} + \beta^2\, \mathbf{g}_{t-2} + \cdots + \beta^{t-1}\, \mathbf{g}_1.$$

The gradient of $k$ steps ago counts with weight $\beta^k$: with
$\beta = 0.5$ the latest gradient counts in full, the one before it half,
and the one three steps back an eighth. Note that the velocity is a sum
and not an average. Its weights add up to more than one, which will matter
as soon as we ask how long a step is. For the heavy ball of the opening,
$1 - \beta$ is the friction, and the method is also known as *heavy-ball
momentum*.

In code, the velocity occupies the two state variables that `train_2d`
hands to an update rule. We return to the learning rate $0.6$, at which
gradient descent diverged, and add momentum $\beta = 0.5$.

```{.python .input #momentum-the-momentum-method-1}
%%tab pytorch, jax
def momentum_2d(x1, x2, v1, v2, f_grad, eta, beta):
    g1, g2 = f_grad(x1, x2)
    v1, v2 = beta * v1 + g1, beta * v2 + g2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

d2l.show_trace_2d(f_valley, d2l.train_2d(momentum_2d, valley_grad, eta=0.6,
                                         beta=0.5))
```

The first step is the one that gradient descent would take, since the
velocity starts at zero, and it throws $x_2$ from $-2$ to $2.8$. But the
swings die out within about ten steps while the iterate runs down the
floor, and after 20 steps both coordinates are within $0.01$ of the
minimum. Along $x_2$ a new gradient usually met a velocity of the opposite
sign, and the two cancelled in part. Along $x_1$ the gradients piled up.

That the run converges at all should puzzle us. Gradients that pile up
make for longer steps than gradient descent takes at the same $\eta$, and
yet momentum is stable where gradient descent diverged. How long, then, is
a step of momentum?

### The Effective Step

The length of a step depends on what the gradients do from one step to the
next. Suppose first that successive gradients agree, as they do along the
floor of the valley. The velocity then builds up to
$1 + \beta + \beta^2 + \cdots = 1/(1-\beta)$ times a single gradient, and
the iterate moves as if gradient descent had the learning rate

$$\eta_\textrm{eff} = \frac{\eta}{1-\beta},$$

which we call the *effective step*. Raising $\beta$ from $0.9$ to $0.99$
at a fixed $\eta$ therefore makes the steps ten times longer, not only
smoother.

Now suppose that the gradients alternate in sign, as they do across the
valley when every step overshoots. The velocity then amounts to only
$1 - \beta + \beta^2 - \cdots = 1/(1+\beta)$ times a single gradient, and
the step shrinks to $\eta/(1+\beta)$. This resolves the puzzle. At
$\eta = 0.6$ and $\beta = 0.5$ the step along the floor is $1.2$ times the
gradient. Across the valley, as long as the iterate keeps changing sides,
it is only $0.4$ times the gradient, less than the $0.5$ that gradient
descent could afford there. Momentum lengthens the step where gradient
descent was too slow and shortens it where gradient descent overshot.

The alternating sum also gives the new ceiling. Gradient descent is at the
edge of divergence when a step flips the sign of a coordinate and
preserves its size, at $\eta\lambda = 2$ in a direction of curvature
$\lambda$. Under momentum such an iterate moves with the step
$\eta/(1+\beta)$, so the edge lies at $\eta\lambda = 2(1+\beta)$. For the
valley at $\beta = 0.5$ the ceiling on $\eta$ rises from $0.5$ to $0.75$,
and the run at $0.6$ lies below it (Exercise :exref:`ex_momentum-stability`
derives the condition from the update itself).

This is the mechanism of the speed-up. Where gradients persist, momentum
steps as gradient descent would at the learning rate
$\eta_\textrm{eff}$, and that rate may exceed anything gradient descent
could have used. The steepest direction, of curvature $\lambda_{\max}$,
held gradient descent to $\eta < 2/\lambda_{\max}$. The same direction
holds momentum to
$\eta_\textrm{eff} < 2(1+\beta)/\left((1-\beta)\lambda_{\max}\right)$,
which on the valley at $\beta = 0.5$ is $1.5$ against $0.5$. Our run used
$1.2$.

### The Timescale of $\beta$
:label:`subsec_momentum-timescale`

The same geometric sum measures the memory of the velocity. A plain sum of
the last $n$ gradients gives them a total weight of $n$. The weights
$1, \beta, \beta^2, \ldots$ of the velocity add up to $1/(1-\beta)$, as
much as a window of the last $1/(1-\beta)$ gradients would carry, and in
this sense momentum $\beta$ looks back over about $1/(1-\beta)$ steps. We
compute the window for three common values. Next to it we print the
half-life $\log \frac{1}{2} / \log \beta$, the number of steps after
which a gradient's weight has halved, and the share of the total weight
that falls on the gradients inside the window.

```{.python .input #momentum-effective-sample-weight}
%%tab pytorch, jax
print('beta   window   half-life   weight inside the window')
for beta in [0.5, 0.9, 0.99]:
    window = 1 / (1 - beta)
    half_life = np.log(0.5) / np.log(beta)
    inside = 1 - beta ** window
    print(f'{beta:4}   {window:6.0f}   {half_life:9.1f}   {inside:24.2f}')
```

The value $\beta = 0.9$ of the recipes looks back about 10 steps, and
$\beta = 0.99$ about 100. "About" is the right word. The gradients inside
the window carry between 63% and 75% of the total weight, and a gradient
loses half of its weight well before the window ends, after fewer than 7
steps at $\beta = 0.9$. The values $0.9$ and $0.99$ look close and are far
apart: what counts is $1 - \beta$, and between them lies a factor of ten
in memory, as in the effective step.

### Minibatch Noise

What does the velocity do to the noise of a minibatch gradient? At a
constant learning rate SGD never comes to rest. It fluctuates in a cloud
around the minimum, the noise ball of :numref:`sec_sgd`, which is the
wider the larger the learning rate. The velocity pools about
$1/(1-\beta)$ noisy gradients, and it is tempting to conclude that
momentum shrinks the noise ball. It does not.

Follow a single noise term $\boldsymbol{\xi}_t$, the error of the gradient
at step $t$. It enters the velocity with weight 1 and stays in it with
weights $\beta, \beta^2, \ldots$, so over the following steps it
displaces the iterate by
$\eta\, (1 + \beta + \beta^2 + \cdots)\, \boldsymbol{\xi}_t = \eta_\textrm{eff}\, \boldsymbol{\xi}_t$
in all. One step of plain SGD at the learning rate $\eta_\textrm{eff}$
would have displaced it as far at once. Momentum delays the noise and does
not remove it: at equal effective step, SGD with and without momentum
fluctuate in noise balls of nearly the same size
(Exercise :exref:`ex_momentum-momentum-under-minibatch-noise`).

For the noise, as for the progress in directions where gradients persist,
it is the effective step that counts. Hence the rule for changing $\beta$
in a run that works: $\eta$ comes down when $\beta$ goes up, so that
$\eta/(1-\beta)$ stays where it was. The rule preserves a run. It does
not say where the best effective step lies, and on that our two findings
pull apart. Curvature permits a larger effective step under momentum than
without it, and noise does not. Which of the two decides depends on what
holds the run back, and we shall meet both cases when we come to data.

## Acceleration and Damping
:label:`subsec_momentum_acceleration`

The setting $\eta = 0.6$, $\beta = 0.5$ was a guess. What are the best
settings, and how much do they gain over gradient descent?

### The Square-Root Law

Consider a quadratic whose curvatures, the eigenvalues of its Hessian, lie
between $\lambda_{\min}$ and $\lambda_{\max}$, with condition number
$\kappa = \lambda_{\max} / \lambda_{\min}$. At its best learning rate,
gradient descent shrinks the error by the factor
$(\kappa - 1)/(\kappa + 1)$ per step (:numref:`sec_optimization-intro`),
and so needs about $\frac{\kappa}{2} \log 100 \approx 2.3\, \kappa$ steps
to reduce it a hundredfold. Heavy-ball momentum with the settings

$$
\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2},
\qquad
\beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2}
$$
:eqlabel:`eq_momentum-optimal`

shrinks it by the factor
$\sqrt{\beta^\star} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ per step,
and needs about $2.3 \sqrt{\kappa}$ steps for the same reduction
:cite:`Polyak.1964`. The square root of the condition number takes the
place of the condition number. This is the central result of the section.
For $\kappa = 100$ it saves a factor of 10, with
$\beta^\star \approx 0.67$, and the worse the conditioning, the closer the
best $\beta$ moves to 1. Read backwards, the formula says what the recipes
assume: $\beta = 0.9$ is the best momentum for a quadratic with
$\kappa \approx 1{,}400$, and $\beta = 0.99$ for
$\kappa \approx 160{,}000$. A default of $0.9$ is a statement about the
problems we expect to meet.

### The Recurrence of a Single Direction

To see where these settings come from, we look at a single direction. On
a quadratic every eigenvector of the Hessian evolves on its own, under
momentum as under gradient descent (:numref:`sec_optimization-intro`); on
the valley the eigenvectors are the two coordinates. Take one such
direction, with curvature $\lambda$, and write $x_t$ and $v_t$ for the
components of the iterate and of the velocity along it, so that the
gradient is $\lambda x_t$. The one step that is not routine is to get rid
of the velocity. The second line of :eqref:`eq_momentum` says that
$\eta v_t = x_t - x_{t+1}$: up to the factor $\eta$, the velocity is the
step just taken. Substituting this for $v_t$ and $v_{t-1}$ in the first
line and collecting terms gives

$$x_{t+1} = (1 + \beta - \eta\lambda)\, x_t - \beta\, x_{t-1}.$$
:eqlabel:`eq_momentum-recurrence`

Gradient descent computes the next iterate from the last one; momentum
uses the last two. Such a recurrence is solved by powers: $x_t = z^t$ is a
solution if $z$ is a root of

$$z^2 - (1 + \beta - \eta\lambda)\, z + \beta = 0,$$

and every solution combines the powers of the two roots. In the long run
the root of larger absolute value wins. It plays the part that
$1 - \eta\lambda$ played for gradient descent: it is the factor by which
the direction shrinks per step. The largest such factor over all
directions is the *contraction factor* of a setting $(\eta, \beta)$.

One property of the roots can be read off without solving anything: their
product is $\beta$. If the roots form a complex-conjugate pair, the two
have the same absolute value, and so each has absolute value
$\sqrt{\beta}$, whatever the curvature. This is the surprising fact about
momentum, and the source of the $\sqrt{\beta^\star}$ in the law. Under
gradient descent every direction had a factor of its own,
$1 - \eta\lambda$, and the flat directions lagged behind. Under momentum
every direction with complex roots shrinks by the same factor per step,
flat or steep. The curvature has dropped out of the rate. It survives in
one place only: in how large $\beta$ must be before the roots of the
flattest direction turn complex.

### Critical Damping

So when are the roots complex? The recurrence describes the heavy ball
of the opening, watched once per step as it rolls in a bowl of curvature
$\lambda$ against the friction $1 - \beta$. With much friction, that is
with little momentum, the ball is *overdamped*, and both roots are real.
In a flat direction one of them is close to 1, and the ball creeps toward
the bottom as the iterate does under gradient descent. With little
friction the ball is *underdamped*: the roots are complex, both of size
$\sqrt{\beta}$, and the ball rolls through the bottom, up the other side
and back, ringing around the minimum.

The ball comes to rest fastest between the two, at *critical damping*,
where the roots coincide. That is where the discriminant
$(1 + \beta - \eta\lambda)^2 - 4\beta$ of the quadratic vanishes. For a
direction of curvature $\lambda$ at a fixed $\eta$ this happens at the
critical value

$$\beta_\textrm{c} = \left(1 - \sqrt{\eta\lambda}\right)^2,$$

and the roots are complex for every $\beta$ above it. In a steep direction
with $\eta\lambda > 1$ the ball is a poorer guide: the real roots of an
overdamped iteration are negative there, and the iterate jumps from wall
to wall. No ball does that, but ours is watched only once per step.

We can now see why :eqref:`eq_momentum-optimal` gives the best settings.
At $\eta^\star$ the critical values of the flattest and of the steepest
direction coincide, at $\beta^\star$. These two directions are critically
damped at once, and every direction between them is underdamped and
shrinks by the same $\sqrt{\beta^\star}$, so none lags behind. Nothing is
gained by moving away. More momentum makes every direction ring, with a
larger $\sqrt{\beta}$. Less momentum leaves the flattest direction
overdamped, and it creeps. A different learning rate pulls the two
critical values apart, so that no $\beta$ serves both ends.

### Four Settings on the Valley

On the valley, $\kappa = 20$ gives $\eta^\star \approx 0.67$ and
$\beta^\star \approx 0.40$, with a predicted factor of
$\sqrt{\beta^\star} \approx 0.635$ against the $0.905$ of gradient
descent. Since $\log 0.635 / \log 0.905 \approx 4.5$, which is
$\sqrt{20}$, tuned momentum should need about $4.5$ times fewer steps. We
check by counting. For each of four settings the cell computes the
contraction factor from the roots and counts the steps after which both
coordinates stay within 1%, and within $10^{-6}$, of their starting
distance from the minimum. The settings are gradient descent at its best
rate, our guess $(0.6, 0.5)$, the tuned pair, and $(0.3, 0.8)$, which
has more momentum at a learning rate that would be safe for gradient
descent alone.

```{.python .input #momentum-acceleration-and-damping-1}
%%tab pytorch, jax
def steps_to(tol, eta, beta, steps=400):
    """Steps after which |x1|, |x2| stay below tol times their start."""
    x1, x2, v1, v2, last = -5.0, -2.0, 0.0, 0.0, 0
    for t in range(1, steps + 1):
        x1, x2, v1, v2 = momentum_2d(x1, x2, v1, v2, valley_grad, eta, beta)
        if abs(x1) > 5 * tol or abs(x2) > 2 * tol:
            last = t
    return last + 1

def contraction(eta, beta):
    return max(abs(np.roots([1, -(1 + beta - eta * lam), beta])).max()
               for lam in (h1, h2))

eta_opt = 4 / (np.sqrt(h2) + np.sqrt(h1)) ** 2
beta_opt = ((np.sqrt(h2 / h1) - 1) / (np.sqrt(h2 / h1) + 1)) ** 2
print('  eta   beta   factor   steps to 1%   steps to 1e-6')
for eta, beta in [(2 / (h1 + h2), 0), (0.6, 0.5), (eta_opt, beta_opt),
                  (0.3, 0.8)]:
    print(f'{eta:5.3f}  {beta:5.3f}  {contraction(eta, beta):7.3f}  '
          f'{steps_to(1e-2, eta, beta):12d}  {steps_to(1e-6, eta, beta):14d}')
```

The first row is gradient descent, with its factor $0.905$ and its 47
steps. Tuned momentum has the predicted factor $0.635$ and reaches
$10^{-6}$ in 40 steps where gradient descent needs 139. That is $3.5$
times fewer, not $4.5$, and the shortfall has a cause. At the tuned
setting both directions of the valley are critically damped. Each has a
double root $z$, and a double root solves the recurrence with $t z^t$ as
well as with $z^t$, so the error falls like $t \cdot 0.635^t$. A pure
$0.635^t$ would be at $10^{-6}$ after 31 steps; the factor $t$ accounts
for the other nine. Our guess carries no such factor, since its roots are
complex in both directions, of size $\sqrt{0.5} \approx 0.707$, and it
wins the short race to 1% by 16 steps to 18. Nobody tunes $\beta$ to three
digits, and the table shows that nobody needs to: the guess and the tuned
pair are within two steps of each other in both columns.

The last row is the one to remember. It has the most momentum and the
most cautious learning rate of the four, and its factor $0.894$ is barely
better than that of gradient descent. Its fault is too much momentum. At
$\eta = 0.3$ the two critical values of the valley are $0.57$ and $0.01$,
so $\beta = 0.8$ is underdamped in both directions, and both ring with
the factor $\sqrt{0.8} \approx 0.894$. We plot the run.

```{.python .input #momentum-acceleration-and-damping}
%%tab pytorch, jax
d2l.show_trace_2d(f_valley, d2l.train_2d(momentum_2d, valley_grad, eta=0.3,
                                         beta=0.8))
```

The iterate swings across the valley for many steps, and along the floor
it runs past the minimum to beyond $x_1 = 1$ before it turns back. This is
a different failure from the divergence of gradient descent at
$\eta = 0.6$: the iterates stay bounded and converge, only slowly.
:numref:`fig_opt_critical_damping` draws the trade-off for a single
direction. The contraction factor falls as $\beta$ grows, is smallest at
the critical value, and past it rises again as $\sqrt{\beta}$.

![Contraction factor per step of heavy-ball momentum in a single direction of curvature $\lambda$, as a function of the momentum $\beta$, for two values of $\eta\lambda$. Below the critical value $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$ the iteration is overdamped and slow; at $\beta_\textrm{c}$ it is fastest; beyond it the iterate oscillates and the factor rises as $\sqrt{\beta}$, along one curve for every $\eta\lambda$. A flatter direction (smaller $\eta\lambda$, dashed) has its critical value closer to one.](../img/mdl-opt-critical-damping.svg)
:label:`fig_opt_critical_damping`

The $\sqrt{\kappa}$ law is a statement about quadratics. It carries over
to general objectives near a minimum, where the quadratic approximation of
the loss applies. Far from one it promises nothing: there are smooth,
strongly convex functions on which heavy-ball momentum at the tuned
settings does not converge at all :cite:`Lessard.Recht.Packard.2016`. The
evidence that momentum speeds up the training of deep networks is
empirical :cite:`Sutskever.Martens.Dahl.ea.2013`.
:numref:`subsec_mdl-momentum-acceleration` proves the law and states the
matching lower bound: no method built from gradients has a better
dependence on $\kappa$ in the worst case. The appendix writes the same
method with the learning rate inside the velocity,
$\mathbf{v} \leftarrow \beta \mathbf{v} - \eta \mathbf{g}$.

## Momentum on Data

The valley gave momentum an exact gradient. Training gives it the gradient
of a minibatch, and a run can then be held back by either of two things
(:numref:`sec_sgd`): by curvature while it is far from its noise ball, and
by noise once it gets there. The analysis makes a different prediction for
each. Where curvature limits progress, momentum should do what it did on
the valley and reach a target loss in several times fewer steps, at a
learning rate no smaller than the best one without momentum. Where noise
limits progress, the effective step decides: the best learning rate at
$\beta = 0.9$ should be about a tenth of the best one without momentum,
and at these two rates the runs should advance alike.

We test both predictions on the training set of Fashion-MNIST
(:numref:`sec_fashion_mnist`). The first needs a problem without noise and
with uneven curvature. Softmax regression (:numref:`sec_softmax`), a
single linear layer from the 784 pixels to the ten logits, is such a
problem when it is trained on all 60,000 images at every step. Its loss is
convex and its gradient exact. Its curvature is uneven because pixel
intensities are never negative: every image has a large component along
the mean image, and the loss curves far more steeply along that one
direction than along most others.

For the second prediction we take `FashionCNN`, the convolutional network
of :numref:`subsec_fashioncnn`, on minibatches of 32 images, small enough
for the noise to matter. (`TinyLM`, the other model of that section, meets
SGD with momentum in :numref:`sec_adam`, side by side with Adam.) The
training set sits on the device, and `Batches` cuts a fresh random order
of it into minibatches on every pass.

```{.python .input #momentum-momentum-on-a-network-1}
%%tab pytorch
fashion = d2l.FashionMNIST()
X = fashion.train.data[:, None].to(d2l.try_gpu()) / 255  # NCHW
y = fashion.train.targets.to(d2l.try_gpu())

class Batches:
    """The examples X, y on the GPU, in a new random order on every pass."""
    def __init__(self, X, y, batch_size):
        self.X, self.y, self.batch_size = X, y, batch_size

    def train_dataloader(self):
        order = torch.randperm(len(self.y), device=self.y.device)
        return ((self.X[i], self.y[i])
                for i in order.view(-1, self.batch_size))
```

```{.python .input #momentum-momentum-on-a-network-1}
%%tab jax
fashion = d2l.FashionMNIST()
X = jnp.asarray(fashion.train[0][..., None], jnp.float32) / 255  # NHWC
y = jnp.asarray(fashion.train[1], jnp.int32)

class Batches:
    """The examples X, y on the device, in a new random order on every pass."""
    def __init__(self, X, y, batch_size):
        self.X, self.y, self.batch_size = X, y, batch_size
        self.key = d2l.get_key()  # Advanced on every pass

    def train_dataloader(self):
        self.key, key = jax.random.split(self.key)
        order = jax.random.permutation(key, len(self.y))
        return ((self.X[i], self.y[i])
                for i in order.reshape(-1, self.batch_size))
```

Softmax regression starts from zero weights, so all of its runs start at
the same point. `make_sgd(beta)` returns SGD with momentum `beta` in the
form that `d2l.lr_sweep` expects, and `new_seed` draws the seed of a
sweep.

```{.python .input #momentum-momentum-on-data-1}
%%tab pytorch
def softmax_regression():
    """A linear layer from the pixels to the logits, starting at zero."""
    net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    nn.init.zeros_(net[1].weight)
    nn.init.zeros_(net[1].bias)
    return net

make_sgd = lambda beta: lambda model, lr: torch.optim.SGD(
    model.parameters(), lr, momentum=beta)
new_seed = lambda: random.getrandbits(31)
```

```{.python .input #momentum-momentum-on-data-1}
%%tab jax
def softmax_regression(rngs):
    """A linear layer from the pixels to the logits, starting at zero."""
    return nnx.Sequential(lambda X: X.reshape(len(X), -1),
                          nnx.Linear(28 * 28, 10, rngs=rngs,
                                     kernel_init=nnx.initializers.zeros))

make_sgd = lambda beta: lambda lr: optax.sgd(lr, momentum=beta)
new_seed = d2l.get_seed
```

:begin_tab:`pytorch`
With `momentum=beta` and its default `dampening=0`, `torch.optim.SGD`
performs the update :eqref:`eq_momentum` as written.
:end_tab:

:begin_tab:`jax`
With `momentum=beta`, `optax.sgd` chains `optax.trace`, which keeps the
velocity of :eqref:`eq_momentum`, with a scaling by $-\eta$.
:end_tab:

The comparison has two arms, in the language of
:numref:`sec_test_models`: SGD with $\beta = 0$ and with $\beta = 0.9$.
For each model and each arm, `d2l.lr_sweep` trains one model per learning
rate on a grid with factors of two, all from one initialization. We keep
the rate with the lowest final loss, the mean of the training loss over
the last 100 steps. Softmax regression trains for 1,000 steps and the CNN
for 3,000. For the best rate of each arm the cell prints the final loss
and the first step at which the training loss, averaged over blocks of 25
steps, reaches a target: $0.5$ for softmax regression and $0.3$ for the
CNN.

```{.python .input #momentum-momentum-on-a-network-2}
%%tab pytorch, jax
# Model, batch size, steps, target loss
setups = {'softmax regression': (softmax_regression, 60000, 1000, 0.5),
          'FashionCNN': (d2l.FashionCNN, 32, 3000, 0.3)}
grids = {('softmax regression', 0.0): [0.0625, 0.125, 0.25, 0.5, 1],
         ('softmax regression', 0.9): [0.1, 0.2, 0.4, 0.8],
         ('FashionCNN', 0.0): [0.05, 0.1, 0.2, 0.4],
         ('FashionCNN', 0.9): [0.005, 0.01, 0.02, 0.04]}
best = {}
for (name, beta), lrs in grids.items():
    make_model, b, num_steps, target = setups[name]
    curves, _ = d2l.lr_sweep(make_model, make_sgd(beta), Batches(X, y, b),
                             lrs, new_seed(), num_steps, verbose=False)
    best[name, beta] = min(lrs, key=lambda lr: d2l.final_loss(curves[lr]))
    c = curves[best[name, beta]]
    print(f'{name:<18}  beta={beta}  best lr {best[name, beta]:<5g}  steps to '
          f'{target}: {d2l.steps_to_reach(c, target)}  final loss '
          f'{d2l.final_loss(c):.3f}')
```

Two runs of one setting differ (:numref:`sec_test_models`), so we repeat
each arm three times at its best rate, from new initializations and data
orders.

```{.python .input #momentum-momentum-on-a-network-3}
%%tab pytorch, jax
for (name, beta), lr in best.items():
    make_model, b, num_steps, target = setups[name]
    runs = [d2l.lr_sweep(make_model, make_sgd(beta), Batches(X, y, b), [lr],
                         new_seed(), num_steps, verbose=False)[0][lr]
            for _ in range(3)]
    print(f'{name:<18}  beta={beta}  lr {lr:<5g}  steps:',
          *[d2l.steps_to_reach(c, target) for c in runs], '  final loss:',
          *[f'{d2l.final_loss(c):.3f}' for c in runs])
```

Softmax regression behaves as the valley did. Without momentum, which on
the full training set means gradient descent, it does best at
$\eta = 0.125$ and needs 625 steps to bring the loss to $0.5$. With
momentum the best rate is $0.4$, and the loss is at $0.5$ after 100 steps,
a sixth of the count. The effective step is then $0.4 / 0.1 = 4$, about
thirty times the best rate without momentum, or between 16 and 64 times
if we allow for grids with factors of two. The final loss is $0.365$ with
momentum and $0.472$ without.

The three repeats print the same numbers to the last digit, because
nothing in these runs is random: they start at zero and use all the data
at every step. There is no run-to-run spread for the saving to hide in.
Note what the square-root law did and did not predict. It gave the
direction of the effect and not its size: this loss is not a quadratic,
and $\beta$ was set to $0.9$, not tuned.

The CNN behaves as the noise argument predicted. Its best learning rate is
$0.2$ without momentum and $0.02$ with it. The two grids are tenfold
copies of each other, so this is the same grid point in effective step.
And at these rates neither arm is faster than the other by more than the
runs of one arm differ among themselves.

Six runs are easy to over-read, so let us look closely, and count the run
of the first table as a fourth for each arm. The step at which the loss
reaches $0.3$ then spans hundreds of steps within an arm, and the averages
of the two arms lie closer together than that. The final losses all lie
between about $0.26$ and $0.30$. Within an arm they span up to about
$0.02$, as in :numref:`sec_test_models`, and the averages of the two arms
differ by about $0.01$ or less. That section asks for three times the
spread before a difference counts, so this is a tie. Should all three
repeats of one arm come in ahead of all three of the other, do not be
impressed. Of the twenty ways to rank three runs against three, two put
one arm wholly in front, so chance alone manages it one time in ten.
Momentum left the speed of this run as it was and moved the learning rate
at which the run should be made.

## Nesterov Momentum

The valley left one defect of momentum unrepaired. At $\eta = 0.3$ and
$\beta = 0.8$ the iterate rang around the minimum, and the ringing has a
cause we can name. The velocity carries the iterate past the minimum, and
the gradient computed at the current point only notices the overshoot one
step after it has happened. :citet:`Nesterov.1983` proposed to evaluate
the gradient at the point the velocity is about to carry the iterate to:

$$
\begin{aligned}
\mathbf{v}_t &= \beta\, \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta \beta\, \mathbf{v}_{t-1}), \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta\, \mathbf{v}_t.
\end{aligned}
$$
:eqlabel:`eq_nesterov`

The point $\mathbf{x}_t - \eta \beta\, \mathbf{v}_{t-1}$ is where the old
velocity alone would take the iterate. If that point lies past the floor
of the valley, the gradient there already points back, and the velocity is
corrected before the overshoot rather than after it.
:numref:`fig_opt_nesterov_lookahead` shows one step of each method from the
same point and velocity.

![One step of heavy-ball and of Nesterov momentum on the valley, from the same iterate and the same velocity. Both first move by the momentum step to the look-ahead point. Heavy ball then adds the gradient step computed at the iterate it started from and overshoots across the valley; Nesterov adds the gradient step computed at the look-ahead point, which points back, and lands near the floor.](../img/mdl-opt-nesterov-lookahead.svg)
:label:`fig_opt_nesterov_lookahead`

In code the change is the point at which the gradient is taken. We run it
at the setting of the underdamped heavy-ball run, $\eta = 0.3$ and
$\beta = 0.8$.

```{.python .input #momentum-nesterov-momentum-1}
%%tab pytorch, jax
def nesterov_2d(x1, x2, v1, v2, f_grad, eta, beta):
    g1, g2 = f_grad(x1 - eta * beta * v1,  # Gradient at the look-ahead
                    x2 - eta * beta * v2)  # point, not at (x1, x2)
    v1, v2 = beta * v1 + g1, beta * v2 + g2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

d2l.show_trace_2d(f_valley, d2l.train_2d(nesterov_2d, valley_grad, eta=0.3,
                                         beta=0.8))
```

The zigzag across the valley dies out after the first few steps. After 20
steps the printed $x_2$ is smaller than $10^{-4}$ in size, where the
heavy-ball run still stood at $-0.25$. Along the floor the iterate
overshoots the minimum to about $0.7$, where heavy ball went beyond 1. The
look-ahead acts as extra damping, strongest in the steep direction, where
the overshoot was largest. (The two printed values of $x_1$ are no guide:
step 20 catches both runs on their way back from the overshoot.)

For practice it is enough to know that the look-ahead damps the ringing
of momentum and costs nothing more per step. Libraries implement it in a
form that stores the look-ahead point as the parameters, so that the
gradient is taken at the parameters as usual
(Exercise :exref:`ex_momentum-two-forms-nesterov-momentum`). The SGD
optimizer used above takes it as the argument `nesterov=True`, which is
off by default.

The method owes its fame to two guarantees that heavy-ball momentum lacks.
On smooth convex functions, with a momentum that grows toward 1 over the
steps, its error after $t$ steps falls as $O(1/t^2)$, against $O(1/t)$ for
gradient descent, and no method built from gradients does better in the
worst case. And its $\sqrt{\kappa}$ rate has a proof that is not confined
to quadratics :cite:`Nesterov.2018`. For us both are background;
:numref:`subsec_mdl-momentum-acceleration` states them.

## The Choice of $\beta$ and $\eta$

The practical advice is short. We recommend leaving $\beta$ at $0.9$ and
thinking in terms of the effective step $\eta/(1-\beta)$, the one
quantity that keeps its meaning when $\beta$ changes. When you add
momentum to a run that worked without it, start at the same effective
step, a tenth of the old learning rate, and sweep upward from there. If
noise limits the run, the best rate will stay where you started, as it did
for the CNN. If curvature limits it, the best rate will move up, as it did
for softmax regression. Expect to save steps where
the gradient is accurate and the curvature uneven, as in the full-batch
run, and expect no saving where small minibatches set the pace. In that
case momentum does no harm either, and since one rarely knows in advance
which case a run is in, it stays switched on.

We can now say what `momentum=0.9` did in recipe A of
:numref:`sec_training_recipes`, which pairs it with a base learning rate
of $0.05$ and minibatches of 128 images. Two things are certain. It gave
the gradients a memory of about ten steps, and it turned the learning rate
into an effective step of $0.5$, ten times what the number $0.05$ suggests
to someone used to plain SGD. Whether it also saved steps we cannot say
from here. Batches of 128 lie between our two experiments, and where a
network changes from the one case to the other has to be measured.
Exercise :exref:`ex_momentum-momentum-under-minibatch-noise` does so for
`FashionCNN`.

Momentum averages its way around uneven curvature without ever measuring
it. Every coordinate still receives the same learning rate, and the
steepest direction still decides how large that rate may be. In
:numref:`sec_gd` each coordinate could have a rate of its own because we
knew the curvatures. A network supplies gradients and nothing else. Can
the sizes of past gradients stand in for the curvature? That is the
question of :numref:`sec_adam`.

## Summary

Momentum steps along the velocity, a decaying sum of past gradients, at
the cost of one buffer per parameter and one hyperparameter $\beta$.
Gradients that persist from step to step add up to $1/(1-\beta)$ times a
single gradient and gradients that alternate shrink to $1/(1+\beta)$ of
one, so momentum takes long steps along the floor of a valley and short
ones across it. On a quadratic, tuned $\eta$ and $\beta$ bring the number
of steps from order $\kappa$ to order $\sqrt{\kappa}$: on the valley, with
$\kappa = 20$, from 139 to 40 for a reduction to $10^{-6}$. Too much
momentum makes the iterate ring around the minimum, which Nesterov's
look-ahead damps. On data the saving appeared where curvature limits
progress, a sixth of the steps for full-batch softmax regression, and not
where minibatch noise does: there momentum only rescales the learning rate
by $1 - \beta$. Momentum still gives every coordinate the same learning
rate.

## Exercises

1. [code] **Momentum and learning rate together.** On the valley
   `f_valley` with `momentum_2d`, run a grid of
   $\beta \in \{0, 0.25, 0.5, 0.8\}$ and
   $\eta \in \{0.1, 0.3, 0.6, 1.0\}$ for up to 1000 steps each.
    1. Record for every cell the number of steps until
       $\|\mathbf{x}_t\| \leq 10^{-3}$, or the fact that the iterates
       diverged, and display the grid as a table or heatmap.
    1. Compare the fastest $\beta$ at each stable $\eta$ with the
       $\beta^\star$ of :eqref:`eq_momentum-optimal` for $\kappa = 20$ and
       with the critical values $\beta_\textrm{c}$ of the two directions,
       and explain each learning rate at which they differ.
   :exlabel:`ex_momentum-momentum-learning-rate-together`

1. **Velocity noise.** Let
   $\mathbf{g}_t = \mathbf{g} + \boldsymbol{\xi}_t$, where the noise
   $\boldsymbol{\xi}_t$ is independent across steps, has zero mean, and has
   variance $\sigma^2$ per coordinate. Show that the stationary mean of
   $\mathbf{v}_t$ in :eqref:`eq_momentum` is $\mathbf{g}/(1-\beta)$ and its
   stationary variance is $\sigma^2/(1-\beta^2)$. Conclude that the
   noise-to-signal ratio of the velocity is $\sqrt{(1-\beta)/(1+\beta)}$
   times that of a single gradient.
   :exlabel:`ex_momentum-velocity-noise`

1. **Two forms of Nesterov momentum.** Library implementations of Nesterov
   momentum store a parameter vector $\mathbf{y}_t$ and evaluate the
   gradient there, $\mathbf{g}_t = \nabla f(\mathbf{y}_t)$. Starting
   from $\mathbf{y}_1 = \mathbf{x}_1$ and $\mathbf{v}_0 = \mathbf{0}$, they
   update
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad
   \mathbf{y}_{t+1} = \mathbf{y}_t - \eta\,(\mathbf{g}_t + \beta \mathbf{v}_t),$$
   so that the gradient is evaluated at the stored parameters, as for
   every other optimizer.
    1. Express $\mathbf{y}_t$ through the $\mathbf{x}_t$ and $\mathbf{v}_t$
       of :eqref:`eq_nesterov` so that both recurrences produce the same
       velocities, and prove the relation by induction over $t$.
    1. A checkpoint saves the library's parameters $\mathbf{y}_t$. Which
       point of :eqref:`eq_nesterov` does it store, and what else must be
       saved to recover $\mathbf{x}_t$?
   :exlabel:`ex_momentum-two-forms-nesterov-momentum`

1. **Stability of momentum.** ● The text found the ceiling
   $\eta\lambda < 2(1+\beta)$ from an iterate that flips its sign at every
   step.
    1. Derive the ceiling from :eqref:`eq_momentum-recurrence`: show that
       for $0 \leq \beta < 1$ both roots of
       $z^2 - (1 + \beta - \eta\lambda) z + \beta = 0$ lie inside the unit
       circle if and only if $0 < \eta\lambda < 2(1+\beta)$.
    1. Check the result against the largest stable $\eta$ for each
       $\beta$ in the grid of
       problem :exref:`ex_momentum-momentum-learning-rate-together`.
    1. In terms of the effective step the ceiling reads
       $\eta_\textrm{eff}\, \lambda < 2(1+\beta)/(1-\beta)$, which grows
       without bound as $\beta \to 1$. Give two reasons why a training
       run cannot use this freedom: one from the contraction factor
       $\sqrt{\beta}$ of an underdamped direction, one from minibatch
       noise.
   :exlabel:`ex_momentum-stability`

1. **The averaged form.** Adam in :numref:`sec_adam`, and many papers,
   write momentum with an average in place of the sum,
   $$\mathbf{m}_t = \beta \mathbf{m}_{t-1} + (1-\beta)\, \mathbf{g}_t, \qquad
   \mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\, \mathbf{m}_t,$$
   with $\mathbf{m}_0 = \mathbf{0}$.
    1. Show that $\mathbf{m}_t = (1-\beta)\, \mathbf{v}_t$ for the velocity
       of :eqref:`eq_momentum`, so that the two forms take the same steps
       when $\eta = (1-\beta)\, \alpha$. Which of $\eta$ and $\alpha$ is
       the effective step? Which form lets you change $\beta$ without
       retuning the learning rate?
    1. Some libraries, PyTorch among them, offer both forms through a
       *dampening* $d$, with the update
       $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-d)\, \mathbf{g}_t$, and
       set $\mathbf{v}_1 = \mathbf{g}_1$ on the first step whatever $d$
       is. For $d = \beta$, compute the sum of the weights that
       $\mathbf{v}_t$ gives to $\mathbf{g}_1, \ldots, \mathbf{g}_t$, and
       compare it with the sum of the weights in $\mathbf{m}_t$. Which of
       the two takes too small a step at the start of training, and by
       what factor at step $t$?
   :exlabel:`ex_momentum-averaged-form`

1. [code] **Many eigenvalues.** ● Consider
   $f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{10} \lambda_i x_i^2$ with
   $\lambda_i = 2^{-i}$, started from $x_i = 1$, and let
   $\kappa = \lambda_{\max} / \lambda_{\min}$.
    1. Run gradient descent at $\eta = 2/(\lambda_{\max} + \lambda_{\min})$,
       the learning rate that minimizes its worst-case contraction factor
       $\max_i |1 - \eta\lambda_i|$. Run heavy-ball momentum at the
       $\eta^\star$ and $\beta^\star$ of :eqref:`eq_momentum-optimal`.
       Plot $|x_{i,t}|$ for every coordinate on a logarithmic scale.
    1. Which coordinates limit the convergence of each method? Compare the
       ratio of the iteration counts until $\|\mathbf{x}_t\| \leq 10^{-6}$
       with $\sqrt{\kappa}$.

1. [code] **Momentum under minibatch noise.** ● Keep the noise model of
   problem :exref:`ex_momentum-velocity-noise`, but let the gradient depend
   on the iterate: on $f(x) = \frac{\lambda}{2} x^2$, take
   $g_t = \lambda x_t + \xi_t$.
    1. Compare plain SGD at learning rate $\alpha$ with momentum
       $\beta = 0.9$ at $\eta = \alpha (1-\beta)$, which keeps the
       effective step $\eta/(1-\beta)$ equal to $\alpha$. Compute the
       ratio of the two stationary variances of $x_t$ for
       $\alpha\lambda \in \{0.05, 0.2\}$, by simulation or by solving for
       the stationary covariance of iterate and velocity. Does the
       reduction of the velocity's noise-to-signal ratio from
       problem :exref:`ex_momentum-velocity-noise` carry over to the
       iterate? Explain the result.
    1. With `Batches`, `make_sgd` and `d2l.lr_sweep` as in this section,
       train `d2l.FashionCNN` for 1,000 steps at the batch sizes
       $b \in \{32, 500, 6000\}$ (a batch size must divide 60,000), with
       $\beta \in \{0, 0.9\}$ and $\eta$ chosen so that
       $\eta/(1-\beta) = 0.2$. Report the final loss of each run and the
       standard deviation of its last 100 losses. At which batch sizes do
       the two arms agree, as the first item predicts?
    1. At $b = 6000$ the gradient is nearly exact, and the analysis
       predicts that momentum now saves steps. Tune both arms on a grid
       with factors of two and repeat each best rate three times. Does
       momentum save steps, by more than the spread between the repeats?
       Does setting `nesterov=True` change the answer?
   :exlabel:`ex_momentum-momentum-under-minibatch-noise`

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1070)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1071)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.5]{.kicker}

Momentum<br>**the velocity · the effective step · acceleration and damping · momentum on data**
:::
:::

::: {.slide title="Gradient Descent Needs 47 Steps on a Valley with κ = 20"}
[The problem]{.kicker}

The valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$, from $(-5, -2)$, gradient
descent at its best learning rate $\eta = 2/(0.2 + 4) \approx 0.476$:

| target | steps |
|:--|--:|
| both coordinates below 1% of their starting values | 47 |
| both coordinates below $10^{-6}$ times their starting values | 139 |

::: {.d2l-note .warn}
The steep wall (curvature 4) holds $\eta$ below a ceiling of 0.5; at that
rate the flat floor (curvature 0.2) shrinks by less than 10% per step.
Steps grow like the condition number $\kappa = 4/0.2 = 20$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the momentum update and unroll it into a sum of past gradients;
- convert a momentum $\beta$ into a memory of $1/(1-\beta)$ steps and an
  effective step $\eta/(1-\beta)$, and say where the speed-up comes from;
- compute the best $\eta$ and $\beta$ for a quadratic and tell overdamped
  from underdamped;
- predict where momentum saves steps on data and where it only rescales
  the learning rate.

**Idea:** step along a decaying sum of past gradients.

**Running example:** the valley, $\kappa = 20$; then Fashion-MNIST at full
batch and at 32 images per batch.
:::

::: {.slide title="Momentum Answers Curvature without Measuring It"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · **Momentum** · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** gradient descent on the valley and its ceiling
  $\eta\lambda < 2$ ($\lambda$: curvature); the noise of minibatch SGD;
  the image classifier `FashionCNN`.
- **Skips:** proofs of the rates; momentum inside Adam.
:::

::: {.slide title="At η = 0.6 Gradient Descent Diverges across the Valley"}
[The valley]{.kicker}

One step multiplies each coordinate by $1 - \eta\lambda$:

| coordinate | curvature $\lambda$ | factor at $\eta = 0.6$ |
|:--|--:|--:|
| $x_1$ (floor) | 0.2 | $0.88$ |
| $x_2$ (wall) | 4 | $-1.4$ |

After 20 steps:

@!momentum-an-ill-conditioned-problem-2

::: {.d2l-note}
Along $x_2$ the gradient changes sign at every step. A sum of successive
gradients would cancel most of it.
:::
:::

::: {.slide title="The Velocity Is a Decaying Sum of Past Gradients"}
[The momentum method]{.kicker}

$$\mathbf{v}_t = \beta\, \mathbf{v}_{t-1} + \mathbf{g}_t,\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{v}_t,\qquad
\mathbf{v}_0 = \mathbf{0}$$

- $\mathbf{g}_t$: gradient at the parameters $\mathbf{x}_t$; $\eta$:
  learning rate; $\beta \in [0, 1)$: the momentum.
- Unrolled: $\mathbf{v}_t = \mathbf{g}_t + \beta\,\mathbf{g}_{t-1} +
  \beta^2\,\mathbf{g}_{t-2} + \cdots$
- $\beta = 0$ is gradient descent; $1 - \beta$ acts as friction on a
  heavy ball.

[Polyak, 1964]{.small}
:::

::: {.slide title="With β = 0.5, the Divergent Learning Rate Converges"}
[The momentum method]{.kicker}

The valley, $\eta = 0.6$, $\beta = 0.5$, 20 steps:

@!momentum-the-momentum-method-1

::: {.d2l-note .warn}
**Puzzle.** Gradients that add up make longer steps than gradient descent
takes. Why is $\eta = 0.6$ stable now?
:::
:::

::: {.slide title="Persistent Gradients Add Up; Alternating Ones Cancel"}
[The effective step]{.kicker}

| successive gradients | velocity, in gradients | step | $\eta = 0.6$, $\beta = 0.5$ |
|:--|:--|:--|--:|
| agree (floor) | $1 + \beta + \beta^2 + \cdots = \frac{1}{1-\beta}$ | $\frac{\eta}{1-\beta}$ | 1.2 |
| alternate (wall) | $1 - \beta + \beta^2 - \cdots = \frac{1}{1+\beta}$ | $\frac{\eta}{1+\beta}$ | 0.4 |

::: {.d2l-note .rule}
**Effective step** $\eta_\textrm{eff} = \eta/(1-\beta)$. Across the valley
the step is 0.4, below the 0.5 that gradient descent can afford: the
puzzle is solved.
:::
:::

::: {.slide title="Momentum May Use a Step That Gradient Descent Cannot"}
[The effective step]{.kicker}

Edge of divergence: a step flips the sign of a coordinate and keeps its
size.

| | step of a sign-flipping iterate | edge | valley, $\lambda_{\max} = 4$ |
|:--|:--|:--|:--|
| gradient descent | $\eta$ | $\eta\lambda = 2$ | $\eta < 0.5$ |
| momentum $\beta$ | $\frac{\eta}{1+\beta}$ | $\eta\lambda = 2(1+\beta)$ | $\eta < 0.75$ at $\beta = 0.5$ |

::: {.d2l-note .rule}
**The speed-up.** Along the floor momentum moves like gradient descent at
$\eta_\textrm{eff}$, and $\eta_\textrm{eff}$ may reach
$\frac{2(1+\beta)}{(1-\beta)\lambda_{\max}}$: 1.5 instead of 0.5.
:::
:::

::: {.slide title="β = 0.9 Remembers About 10 Steps"}
[The timescale of β]{.kicker}

Weights $1, \beta, \beta^2, \ldots$ sum to $\frac{1}{1-\beta}$, the
*window*. Half-life: steps until a gradient's weight has halved. Last
column: share of the total weight on the gradients inside the window.

@!momentum-effective-sample-weight

::: {.d2l-note}
What counts is $1 - \beta$: from 0.9 to 0.99 the memory grows tenfold.
:::
:::

::: {.slide title="Momentum Delays Minibatch Noise; It Does Not Remove It"}
[Minibatch noise]{.kicker}

A noise term $\boldsymbol{\xi}_t$ in the gradient at step $t$ stays in the
velocity with weights $1, \beta, \beta^2, \ldots$ and moves the parameters
by

$$\eta\,(1 + \beta + \beta^2 + \cdots)\,\boldsymbol{\xi}_t
= \eta_\textrm{eff}\,\boldsymbol{\xi}_t$$

in total: one step of plain SGD at learning rate $\eta_\textrm{eff}$,
spread over about $\frac{1}{1-\beta}$ steps.

::: {.d2l-note}
*Noise ball:* the cloud around the minimum in which SGD at a constant
$\eta$ keeps fluctuating. At equal $\eta_\textrm{eff}$ it has nearly the
same size with and without momentum.
:::
:::

::: {.slide title="Check: Raise β from 0.9 to 0.99"}
[Check]{.kicker}

A run trains well at $\eta = 0.01$, $\beta = 0.9$. You switch to
$\beta = 0.99$. What learning rate keeps the effective step, and how far
back does the velocity now look?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta_\textrm{eff} = 0.01/0.1 = 0.1$, so
$\eta = 0.1 \cdot 0.01 = 0.001$, ten times smaller. The window grows from
about 10 to about 100 steps (half-life 6.6 → 69).
:::
:::
:::

::: {.slide title="Tuned Momentum Needs √κ Steps Instead of κ"}
[Acceleration]{.kicker}

Quadratic with curvatures in $[\lambda_{\min}, \lambda_{\max}]$,
$\kappa = \lambda_{\max}/\lambda_{\min}$:

$$\eta^\star = \frac{4}{(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2},
\qquad \beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$$

| | factor per step | steps per factor of 100 |
|:--|:--:|:--:|
| gradient descent, best $\eta$ | $\frac{\kappa - 1}{\kappa + 1}$ | $2.3\,\kappa$ |
| momentum at $\eta^\star$, $\beta^\star$ | $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ | $2.3\,\sqrt{\kappa}$ |

::: {.d2l-note}
Read backwards: $\beta = 0.9$ is the best momentum for
$\kappa \approx 1{,}400$.
:::

[Polyak, 1964; exact for quadratics]{.small}
:::

::: {.slide title="With Complex Roots, Every Direction Shrinks by √β"}
[Acceleration]{.kicker}

One direction, coordinate $x_t$, curvature $\lambda$, gradient
$\lambda x_t$:

I. $\;\eta v_t = x_t - x_{t+1}$ (the velocity is the last step)

II. $\;x_{t+1} = (1 + \beta - \eta\lambda)\,x_t - \beta\,x_{t-1}$

III. Try $x_t = z^t$: $\;z^2 - (1 + \beta - \eta\lambda)\,z + \beta = 0$,
so the two roots have product $\beta$

$$\boxed{|z| = \sqrt{\beta} \ \textrm{ for complex roots, whatever } \lambda}$$

Flat or steep, such a direction shrinks by $\sqrt{\beta}$ per step: the
curvature has dropped out. The largest $|z|$ over all directions is the
*contraction factor*.
:::

::: {.slide title="Critical Damping Is the Fastest Setting"}
[Damping]{.kicker}

![](../img/mdl-opt-critical-damping.svg){width=50%}

- Too little momentum: *overdamped*, real roots, the iterate creeps.
- Too much: *underdamped*, complex roots, the iterate rings.
- Between them the roots coincide, at
  $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$.
- At $\eta^\star$ the flattest and the steepest direction are both
  critical at $\beta^\star$.
:::

::: {.slide title="On the Valley, Tuned Momentum Needs 40 Steps Instead of 139"}
[Acceleration]{.kicker}

$\kappa = 20$: $\eta^\star \approx 0.67$, $\beta^\star \approx 0.40$,
predicted factor $\sqrt{\beta^\star} \approx 0.635$ against 0.905. Rows:
gradient descent at its best $\eta$; a guess; the tuned pair; too much
momentum.

@!momentum-acceleration-and-damping-1

::: {.d2l-note}
139/40 = 3.5, below $\sqrt{20} \approx 4.5$: at critical damping the two
roots coincide, and the error falls like $t \cdot 0.635^t$.
:::
:::

::: {.slide title="Too Much Momentum Rings Around the Minimum"}
[Damping]{.kicker}

$\eta = 0.3$ (safe for gradient descent), $\beta = 0.8$, above both
critical values (0.57 and 0.01): factor $\sqrt{0.8} \approx 0.894$, barely
better than 0.905.

@!momentum-acceleration-and-damping
:::

::: {.slide title="Check: A Condition Number of 10,000"}
[Check]{.kicker}

For $\kappa = 10^4$, what is $\beta^\star$, and how many steps do gradient
descent and tuned momentum need per factor of 100?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\sqrt{\kappa} = 100$, so $\beta^\star = (99/101)^2 \approx 0.96$.
Steps: $2.3 \cdot 10^4 = 23{,}000$ against $2.3 \cdot 100 = 230$.
:::
:::
:::

::: {.slide title="The Valley Makes Two Predictions for Data"}
[Momentum on data]{.kicker}

| what limits the run | prediction at $\beta = 0.9$ |
|:--|:--|
| curvature (exact gradient) | several times fewer steps; best $\eta$ no smaller than without momentum |
| noise (small minibatches) | no saving; best $\eta$ ten times smaller, same $\eta_\textrm{eff}$ |

Test on Fashion-MNIST (60,000 training images), two *arms* each, SGD with
$\beta = 0$ and with $\beta = 0.9$:

- softmax regression, one linear layer, all images at every step;
- `FashionCNN`, a small convolutional network, 32 images per step.
:::

::: {.slide title="At Full Batch, Momentum Reaches the Target in a Sixth of the Steps"}
[Momentum on data]{.kicker}

Best learning rate of a factor-2 grid for each arm; steps until the
training loss reaches the target; final loss: mean of the last 100 steps.

@!momentum-momentum-on-a-network-2

::: {.d2l-note}
Softmax regression: 100 steps instead of 625, at an effective step
$0.4/0.1 = 4$, about 30 times the best plain rate. The `FashionCNN` rows
are single runs; the next slide repeats them.
:::
:::

::: {.slide title="At 32 Images per Batch, Momentum Only Rescales the Learning Rate"}
[Momentum on data]{.kicker}

Three repeats at each best rate (softmax regression has no randomness, so
its repeats are identical):

@!momentum-momentum-on-a-network-3

::: {.d2l-note}
`FashionCNN`: best rates 0.2 and 0.02, the same effective step. With the
runs of the previous slide, the arms differ on average by less than the
runs of one arm do: a tie. Three runs wholly ahead of three: chance does
that one time in ten.
:::
:::

::: {.slide title="Nesterov Takes the Gradient Where the Velocity Is Headed"}
[Nesterov momentum]{.kicker}

$$\mathbf{v}_t = \beta\, \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_t - \eta \beta\, \mathbf{v}_{t-1}),\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{v}_t$$

![](../img/mdl-opt-nesterov-lookahead.svg){width=50%}

The velocity is corrected before the overshoot, not one step after it.

[Nesterov, 1983]{.small}
:::

::: {.slide title="The Look-Ahead Damps the Ringing"}
[Nesterov momentum]{.kicker}

::: {.cols .vc}
::: {.col}
@!momentum-nesterov-momentum-1
:::

::: {.col}
Same $\eta = 0.3$, $\beta = 0.8$ as the underdamped heavy-ball run.

- After 20 steps $|x_2| < 10^{-4}$ (heavy ball: $-0.25$).
- Overshoot along $x_1$: about 0.7 (heavy ball: more than 1).
- Libraries store the look-ahead point: `nesterov=True`, off by default.
:::
:::
:::

::: {.slide title="Choose β = 0.9 and Tune the Effective Step"}
[In practice]{.kicker}

- Keep $\beta = 0.9$; tune $\eta_\textrm{eff} = \eta/(1-\beta)$.
- Adding momentum to a working run: start at a tenth of the old learning
  rate, then sweep upward.
- Changing $\beta$: move $\eta$ so that $\eta_\textrm{eff}$ stays fixed.

::: {.d2l-note .rule}
Expect fewer steps where the gradient is accurate and the curvature
uneven. Expect no saving, and no harm, where small minibatches set the
pace.
:::
:::

::: {.slide title="Check: Adding Momentum to a Run"}
[Check]{.kicker}

Plain SGD trains a network at $\eta = 0.1$ on minibatches of 32. You add
$\beta = 0.9$. Which learning rate do you try first, and do you expect
fewer steps? What did full-batch softmax regression do instead?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta = 0.01$, the same effective step. Noise limits the run,
so expect no saving (`FashionCNN`: best rates 0.2 and 0.02, a tie). With
an exact gradient the best effective step rose about 30-fold and the steps
fell sixfold.
:::
:::
:::

::: {.slide title="Momentum Turns κ into √κ Where Curvature Limits"}
[Wrap-up]{.kicker}

- $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$,
  $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_t$: persistent
  gradients add up to $\frac{1}{1-\beta}$, alternating ones shrink to
  $\frac{1}{1+\beta}$.
- $\beta$: a memory of $\frac{1}{1-\beta}$ steps; effective step
  $\frac{\eta}{1-\beta}$; stable for $\eta\lambda < 2(1+\beta)$.
- Valley, tuned: factor 0.635 instead of 0.905, 40 steps instead of 139.
- Data: full-batch softmax regression, 100 steps instead of 625;
  `FashionCNN` at 32 per batch, a tie at a tenth of the learning rate.

**Next:** Adam, a learning rate for each coordinate.
:::
