# Momentum
:label:`sec_momentum`

Several of the convolutional networks in this book were trained by SGD with
`momentum=0.9` (:numref:`sec_training_recipes`). Adam, in the next
section, has momentum built in, and so does almost every other optimizer in
deep learning. We have used the setting without asking what it does.
Momentum addresses the weakness of gradient descent that
:numref:`sec_optimization-intro` and :numref:`sec_gd` measured on a narrow
valley: a learning rate small enough for the steep direction makes little
progress along the flat one. This section asks how much faster an
optimizer can go if it remembers its past gradients.

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ shows how much there is
to gain. At its best learning rate, gradient descent needs 47 steps to
shrink both coordinates to 1% of their starting values. The steep
direction, with curvature 4, keeps the learning rate below $2/4 = 0.5$. At
such a rate the flat direction, with curvature 0.2, shrinks by less than a
tenth per step. The number of steps grows in proportion to the condition
number, here $\kappa = 20$, and the condition numbers of deep networks run
into the hundreds and beyond.

Gradient descent is this slow because it looks only at the current
gradient. The gradients of the last few steps say more. Along the floor of
the valley they point the same way, step after step; across the valley
they alternate in sign, because each step overshoots the opposite wall.
*Momentum* keeps a running average of past gradients and steps along the
average instead. The average grows along the floor and cancels across the
walls. With well-chosen settings it needs 40 steps on the valley where
gradient descent needs 139 to reach a millionth of the starting distance,
and in general a number of steps proportional to $\sqrt{\kappa}$ rather
than $\kappa$. Momentum needs one extra buffer per parameter and one more
hyperparameter.

We derive the method on the valley, work out how to set its
hyperparameter, and implement it for the minibatch training of
:numref:`sec_sgd`.

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

## An Ill-Conditioned Valley

The valley `d2l.f_valley` of :numref:`sec_gd` curves gently along $x_1$,
with second derivative $h_1 = 0.2$, and steeply along $x_2$, with
$h_2 = 4$. Its gradient `d2l.valley_grad` is $(h_1 x_1, h_2 x_2)$. As in
:numref:`sec_gd`, an update rule for `d2l.train_2d` receives the two
coordinates, two state variables, the gradient function, and the
hyperparameters, and returns the new coordinates and state; `train_2d` runs
it from $(-5, -2)$ and `d2l.show_trace_2d` draws the path over the contour
lines. Gradient descent, `d2l.gd_2d`, ignores the state. We run it at
$\eta = 0.4$:

<!-- WRITER: cell momentum-an-ill-conditioned-problem-1 (gradient descent at eta = 0.4, a repeat of the run in sec_gd) is cut (MOM-2); its two alias lines moved into the divergent run below. The sentence "We run it at $\eta = 0.4$:" and the paragraph after it now describe no plot: reduce to a clause, as the structure asks (keep the factors 0.92 and -0.6 if wanted, they are arithmetic). -->

In each coordinate a step of gradient descent multiplies the distance to
the minimum by $1 - \eta h$. At $\eta = 0.4$ the factor is $0.92$ along
$x_1$, so after 20 steps $x_1$ has only shrunk from $-5$ to about $-0.94$.
Along $x_2$ the factor is $1 - 1.6 = -0.6$: the iterate overshoots the
floor and lands on the opposite wall, closer to the minimum each time. A
larger learning rate would speed up $x_1$, but it would push the factor of
$x_2$ below $-1$. At $\eta = 0.6$ it is $1 - 2.4 = -1.4$, and every step
throws $x_2$ farther up the opposite wall than the last:

```{.python .input #momentum-an-ill-conditioned-problem-2}
%%tab pytorch, jax
h1, h2 = d2l.VALLEY_CURVATURES  # Curvatures of the valley along x1 and x2
f_valley, valley_grad = d2l.f_valley, d2l.valley_grad
d2l.train_2d(d2l.gd_2d, valley_grad, eta=0.6);
```

After 20 steps $x_2$ has grown from $-2$ to about $-1673$, while $x_1$
reached $-0.39$, more than twice as close as at $\eta = 0.4$. The history
of gradients separates the two directions. Along $x_1$ successive
gradients agree: they are small, but they all point the same way. Along
$x_2$ they alternate in sign, and each step undoes most of the last. An
average over past gradients would amplify the first and cancel the second.

## The Momentum Method

### Leaky Averages

A minibatch averages gradients over *examples* (:numref:`sec_sgd`).
Momentum also averages them over *time*, with an average that leaks: each
past gradient is discounted by a factor $\beta$ per step. The update
replaces the gradient by a *velocity* $\mathbf{v}_t$,

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t,
\end{aligned}
$$
:eqlabel:`eq_momentum`

where $\beta \in [0, 1)$, the velocity starts at $\mathbf{v}_0 = \mathbf{0}$,
and $\mathbf{g}_t$ is the gradient at $\mathbf{x}_{t-1}$: exact, from one
example, or from a minibatch. For $\beta = 0$ the update is gradient
descent. Unrolling the first line shows what the velocity holds,

$$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau}.$$

The velocity is an exponentially weighted sum of all past gradients, in
which the gradient of $\tau$ steps ago counts with weight $\beta^\tau$. The
name comes from mechanics. A heavy ball rolling down the objective
accumulates the forces acting on it instead of following the slope of the
moment, and $1 - \beta$ plays the role of friction. The method is
*heavy-ball momentum* :cite:`Polyak.1964`. :citet:`Sutskever.Martens.Dahl.ea.2013`
showed how much it matters for training deep networks.

### Momentum on the Valley

The update rule `momentum_2d` keeps the velocity in the two state variables
of `train_2d`. At learning rate $0.6$, where gradient descent diverged, we
add momentum $\beta = 0.5$:

```{.python .input #momentum-the-momentum-method-1}
%%tab pytorch, jax
def momentum_2d(x1, x2, v1, v2, f_grad, eta, beta):
    g1, g2 = f_grad(x1, x2)
    v1, v2 = beta * v1 + g1, beta * v2 + g2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

d2l.show_trace_2d(f_valley, d2l.train_2d(momentum_2d, valley_grad, eta=0.6,
                                         beta=0.5))
```

The first steps still swing across the valley, but the swings die out
within about ten steps, and the iterate then runs down the floor. After 20
steps both coordinates are within $0.01$ of the minimum. The velocity did
what the history of gradients suggested: the alternating components along
$x_2$ partly cancel, which tames the overshoot, while the persistent
components along $x_1$ add up.

Nothing in :eqref:`eq_momentum` requires the gradient to be exact. With
minibatch gradients the same average also smooths the sampling noise over
steps, without any extra gradient evaluations, so the velocity is less
noisy than a single minibatch gradient. Whether the iterates become less
noisy too depends on how $\eta$ changes along with $\beta$, the question
of Exercise :exref:`ex_momentum-momentum-under-minibatch-noise`.

### The Timescale of $\beta$
:label:`subsec_momentum-timescale`

How much history does the velocity hold? The weights
$1, \beta, \beta^2, \ldots$ sum to $1/(1-\beta)$, so momentum $\beta$
averages over roughly the last $1/(1-\beta)$ gradients. We compute this
window for three common values, together with the half-life
$\log \frac{1}{2} / \log \beta$, the number of steps after which a
gradient's weight has halved, and the share of the total weight that falls
on the gradients inside the window.

```{.python .input #momentum-effective-sample-weight}
%%tab pytorch, jax
print('beta   window   half-life   weight inside the window')
for beta in [0.5, 0.9, 0.99]:
    window = 1 / (1 - beta)
    half_life = np.log(0.5) / np.log(beta)
    inside = 1 - beta ** window
    print(f'{beta:4}   {window:6.0f}   {half_life:9.1f}   {inside:24.2f}')
```

The value $\beta = 0.9$, the `momentum=0.9` that we have passed to
optimizers since :numref:`sec_training_recipes`, looks back about 10 steps,
and $\beta = 0.99$ about 100. The gradients inside the window carry
between 63% and 75% of the total weight. As $\beta$
approaches 1 the share approaches $1 - e^{-1} \approx 0.63$, because
$\beta^{1/(1-\beta)} \to e^{-1}$.

The same sum fixes the length of a step. When successive gradients agree,
the velocity builds up to $1/(1-\beta)$ times a typical gradient, so in
persistent directions momentum takes steps of effective size

$$\eta_\textrm{eff} = \frac{\eta}{1-\beta}.$$

Raising $\beta$ from $0.9$ to $0.99$ at a fixed $\eta$ therefore makes the
steps ten times longer, not only smoother. When $\beta$ goes up, $\eta$ has
to come down, a rule we test on real data below.

### Acceleration and Damping
:label:`subsec_momentum_acceleration`

On an ill-conditioned quadratic, momentum is not only more stable than
gradient descent but provably faster. Let the Hessian have eigenvalues
between $\lambda_{\min}$ and $\lambda_{\max}$, with condition number
$\kappa = \lambda_{\max} / \lambda_{\min}$. At its best learning rate,
gradient descent shrinks the error by the factor
$(\kappa - 1)/(\kappa + 1)$ per step (:numref:`sec_optimization-intro`),
and so needs about $\frac{\kappa}{2} \log \frac{1}{\epsilon}$ steps to
reduce it by a factor $\epsilon$. Heavy-ball momentum with the settings

$$
\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2},
\qquad
\beta^\star = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2}
$$
:eqlabel:`eq_momentum-optimal`

shrinks it by the factor $\sqrt{\beta^\star} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$
per step, and so needs about $\frac{\sqrt{\kappa}}{2} \log \frac{1}{\epsilon}$
steps :cite:`Polyak.1964`. The square root of the condition number takes
the place of the condition number. For $\kappa = 100$ that saves a factor
of 10 with $\beta^\star \approx 0.67$; for $\kappa = 10^4$ a factor of 100
with $\beta^\star \approx 0.96$. The worse the conditioning, the closer the
best $\beta$ moves to 1. Since $\beta^\star \approx 1 - 4/\sqrt{\kappa}$
for large $\kappa$, the timescale reading of
:numref:`subsec_momentum-timescale` says that the best average spans about
$\sqrt{\kappa}/4$ steps: 25 steps for $\kappa = 10^4$.

On the valley, $\kappa = 20$ gives $\eta^\star \approx 0.67$ and
$\beta^\star \approx 0.40$. We check the prediction by counting steps. For
each setting the code below runs `momentum_2d` and counts the steps after
which both coordinates stay within 1%, and within $10^{-6}$, of their
starting distance from the minimum. It also computes the contraction
factor, the largest factor by which a mode of the valley shrinks per step.
In a mode of curvature $\lambda$ the
coordinate obeys $x_{t+1} = (1 + \beta - \eta\lambda)\, x_t - \beta x_{t-1}$,
a recurrence whose solutions are powers $z^t$ of the roots $z$ of
$z^2 - (1 + \beta - \eta\lambda) z + \beta = 0$; the factor is the largest
root in absolute value over the two curvatures.

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

The first row is gradient descent at its best rate $2/(h_1 + h_2)$, with
factor $0.905$. Tuned momentum shrinks the error by $0.635$ per step and
needs 40 steps to reach $10^{-6}$ where gradient descent needs 139. Per
step, the logarithm of its factor is $\log 0.635 / \log 0.905 \approx 4.5$
times larger, which is $\sqrt{\kappa}$; the step counts differ by a
smaller ratio, 3.5, because the factor describes the error only after the
first few steps. The setting $(0.6, 0.5)$ of the plot above is nearly as
fast, and to reach 1% it is even faster.

The last row raises $\beta$ to $0.8$ and halves the learning rate to a
value at which gradient descent alone would be stable. Its factor $0.894$
is barely better than that of gradient descent.

A model of the iteration explains why. In each eigendirection of the Hessian, the recurrence above
is a mass on a spring with friction $1 - \beta$. With too little momentum
the system is *overdamped*: both roots are real and one of them is close to
1, so the iterate creeps down the valley as gradient descent does. With too
much momentum it is *underdamped*: the roots are complex, both have size
$\sqrt{\beta}$, and the iterate overshoots and rings around the minimum.
The fastest setting sits between the two, at *critical damping*, where the
two roots coincide. For a mode of curvature $\lambda$ at a fixed $\eta$
this happens at the critical value
$\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$. At $\eta^\star$ the
critical values of the flattest and of the steepest mode both equal
$\beta^\star$, which is why :eqref:`eq_momentum-optimal` is the best
setting for all modes at once. At $\eta = 0.3$ the two critical values of
the valley are $0.57$ and $0.01$, so $\beta = 0.8$ is underdamped in
both. We plot the run.

```{.python .input #momentum-acceleration-and-damping}
%%tab pytorch, jax
d2l.show_trace_2d(f_valley, d2l.train_2d(momentum_2d, valley_grad, eta=0.3,
                                         beta=0.8))
```

The iterate swings across the valley for many steps, and along the floor
it overshoots the minimum and swings back before it settles. This is a
different failure from the divergence of gradient descent at $\eta = 0.6$:
the iterates stay bounded and converge, only slowly. :numref:`fig_opt_critical_damping`
draws the trade-off for a single mode. The contraction factor falls as
$\beta$ grows, is smallest at the critical value $\beta_\textrm{c}$, and
past it rises again as $\sqrt{\beta}$.

![Convergence rate per step, the contraction factor of the error, of heavy-ball momentum on a single quadratic mode, as a function of the momentum $\beta$. Below the critical value $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$ the iteration is overdamped and slow; at $\beta_\textrm{c}$ it is fastest; beyond it the rate degrades gently as $\sqrt{\beta}$. Worse conditioning (smaller $\eta\lambda$, dashed) pushes $\beta_\textrm{c}$ toward one, which is why large momentum values are the common default.](../img/mdl-opt-critical-damping.svg)
:label:`fig_opt_critical_damping`

<!-- WRITER: MOM-8: the figure's y axis now reads "contraction factor per step" (was "convergence rate per step"), its tick labels match the in-figure labels, and the underdamped label reads "oscillates, factor sqrt(beta)". Start the caption with "Contraction factor per step" and drop "Convergence rate per step" / "the rate degrades" so caption and axis use one term. -->

The $\sqrt{\kappa}$ law is a statement about quadratics. On general
objectives it holds near a minimum, where the quadratic approximation of
the loss applies, and the evidence that momentum speeds up the training of
deep networks is empirical :cite:`Sutskever.Martens.Dahl.ea.2013`.
:numref:`subsec_mdl-momentum-acceleration` proves the law and states
the matching lower bound: no method built from gradients has a better
dependence on $\kappa$ in the worst case.

## Momentum on Data

<!-- WRITER: the airfoil cells (momentum-implementation-from-scratch-1/-2/-3, momentum-concise-implementation, and momentum-nesterov-momentum-2 below) are cut (MOM-9: five runs between 0.244 and 0.287, no gain shown), with the from-scratch update sgd_momentum, its tab notes and the train_with_rule / train_with_optimizer helpers (deleted from the library in sec_sgd). This section replaces "Implementation" (structure 5). The large-batch arm is the structure's fallback, softmax regression on Fashion-MNIST at full batch: FashionCNN at b = 6,000 did not pass the acceptance test (see the WRITER note after the repeats cell). Predictions to state before the sweep cell: at full batch, where curvature limits the step (raw pixels are not centered, so the problem is badly conditioned), momentum is stable at an effective step eta/(1-beta) far beyond the plain ceiling and reaches the target in far fewer steps; on the CNN at b = 32, where noise limits it, the best rates differ by 1/(1-beta) = 10 and the two tie at the same effective step. Momentum in the frameworks is one argument: torch.optim.SGD(..., momentum=beta) implements eq_momentum as written (dampening=0); optax.sgd(lr, momentum=beta) chains optax.trace, the velocity of eq_momentum, with a scaling by -eta. Those two facts were tab notes of the cut concise cell; keep them as one tab note each at the cell that defines make_sgd. -->

We train two models on the Fashion-MNIST training set
(:numref:`sec_fashion_mnist`), each by SGD with and without momentum. The
first is softmax regression, a single linear layer from the pixels to the
ten logits: its loss is convex, and we train it on all 60,000 images at
every step, so no sampling noise limits it. The second is `FashionCNN` of
:numref:`subsec_fashioncnn`, trained on minibatches of 32 images, where
sampling noise does. The training set sits on the device, and `Batches`
cuts a fresh random order into minibatches on every pass;
`d2l.train_steps` asks a data object only for `train_dataloader()`.

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

Softmax regression starts from zero weights. `make_sgd(beta)` returns the
optimizer with momentum `beta` in the form that `d2l.lr_sweep` expects.

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

For each model and each $\beta \in \{0, 0.9\}$ we sweep the learning
rate in factors of two with `d2l.lr_sweep` and keep the rate with the
lowest final loss. Softmax regression trains for 1,000 steps, the CNN for
3,000. For the best rate of each arm we print the first step at which the
smoothed training loss reaches a target, 0.5 for softmax regression and 0.3
for the CNN (`d2l.steps_to_reach`), and the final loss.

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

Three runs at each best rate, from three new initializations and data
orders, give the spread of these numbers.

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

<!-- WRITER: measured, development runs on GPU (not captured): PyTorch and JAX, one sweep and three repeats per arm per notebook run. Softmax regression, full batch, 1,000 steps: plain best lr 0.125, 625 steps to loss 0.5, final loss 0.472; momentum 0.9 best lr 0.4 (effective step 4, 32 times the plain best), 100 steps to 0.5, final loss 0.365. Identical in both tabs and in every repeat (zero start, full batch: the only randomness left is the order of a sum), so the run-to-run range is zero and the sixfold gain in steps is firm. FashionCNN, b = 32, 3,000 steps (four notebook runs, two per tab): plain best lr 0.2 in every run; momentum best lr 0.02 in three runs and 0.04 in one, so the best rates differ by a factor of 10 (once 5, one grid step). Steps to loss 0.3: plain 1,225 to 2,000, momentum 1,075 to 1,750; final loss: plain 0.264 to 0.300, momentum 0.255 to 0.299. The ranges overlap: at the small batch the two tie, at a tenfold smaller rate for momentum. Do not state a winner on the CNN. Not printed by the notebook (do not quote numbers): FashionCNN at b = 6,000 (1,000 steps, five seeds per tab) gained about twofold in steps to 0.3 (plain 0.4: 575 to 825; momentum 0.08: 300 to 375), but momentum's best grid point 0.16 sits at its stability edge (steps 225 to 550, one of five PyTorch runs stuck at ln 10), so three repeats did not give a gain three times the spread; this is why the fallback arm is softmax regression. -->

## Nesterov Momentum

The ringing at $\beta = 0.8$ has a cause we can name. The velocity carries
the iterate past the minimum, and the gradient computed at the current
point only notices the overshoot one step after it has happened.
:citet:`Nesterov.1983` proposed to evaluate the gradient at the point the
velocity is about to carry the iterate to:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_{t-1} - \eta \beta\, \mathbf{v}_{t-1}), \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t.
\end{aligned}
$$
:eqlabel:`eq_nesterov`

The point $\mathbf{x}_{t-1} - \eta \beta\, \mathbf{v}_{t-1}$ is where the
old velocity alone would take the iterate. If that point lies past the
floor of the valley, the gradient there already points back, and the
velocity is corrected before the overshoot rather than after it.
:numref:`fig_opt_nesterov_lookahead` shows one step of each method from the
same point and velocity.

![One step of heavy-ball and of Nesterov momentum on the valley, from the same iterate and the same velocity. Both first move by the momentum step to the look-ahead point. Heavy ball then adds the gradient step computed at the old iterate and overshoots across the valley; Nesterov adds the gradient step computed at the look-ahead point, which points back, and lands near the floor.](../img/mdl-opt-nesterov-lookahead.svg)
:label:`fig_opt_nesterov_lookahead`

In code the change is the point at which the gradient is taken:

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

At the same $\eta = 0.3$ and $\beta = 0.8$ as the underdamped heavy-ball
run, the zigzag across the valley dies out after the first steps, and the
overshoot along the floor is smaller. The look-ahead acts as extra damping,
strongest in the steep direction, where the overshoot is largest. Nesterov's
method also has guarantees that heavy-ball momentum lacks. On smooth convex
functions, with a momentum that grows toward 1 over the steps, its error
after $k$ steps falls as $O(1/k^2)$, against $O(1/k)$
for gradient descent, and no method built from gradients does better in
the worst case. On strongly convex functions a constant momentum
$\beta = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ achieves the
$\sqrt{\kappa}$ rate, with a proof that is not confined to quadratics
:cite:`Nesterov.2018`. :numref:`subsec_mdl-momentum-acceleration`
states the result, proves it on quadratics, and cites the general proof.

<!-- WRITER: cell momentum-nesterov-momentum-2 (Nesterov on the airfoil data) is cut with the other airfoil runs; the two paragraphs that stood here (the nesterov=True flag, and "On the airfoil data the loss curve is nearly the same ... matters when curvature rather than noise limits progress") lose their cell. Keep the flag in a sentence (torch.optim.SGD(..., nesterov=True), optax.sgd(..., nesterov=True); off by default; the library form needs no extra gradient evaluation, Exercise ex_momentum-two-forms-nesterov-momentum); the data claim has no run behind it now: cut it, or state it only as what the large-batch arm of "Momentum on a Network" shows for heavy ball. Also (structure 5, reviewer): the printed step 20 is not the place to compare overshoot (step 20: Nesterov x1 = 0.218 vs heavy ball 0.196; largest overshoot of x1 0.67 vs 1.13). -->

Momentum fixes the slow progress along the flat directions of a valley,
but it does so with one learning rate for all coordinates, and that rate
is still capped by the steepest direction. The next section gives each
coordinate a step size of its own.

## Summary

Momentum replaces the gradient with a leaky average of past gradients, the
velocity, using one buffer per parameter and one hyperparameter
$\beta$. On an ill-conditioned valley the components of the gradient that
persist from step to step accumulate up to $1/(1-\beta)$-fold, while those
that alternate cancel. With tuned $\eta$ and $\beta$ the number of steps
falls from order $\kappa$ to order $\sqrt{\kappa}$: on the valley with
$\kappa = 20$, 40 steps instead of 139 to reach $10^{-6}$. The parameter
$\beta$ is a timescale, an average over about $1/(1-\beta)$ gradients, and
a damping knob: past critical damping the iterate rings around the
minimum. The effective step is $\eta/(1-\beta)$, so raising
$\beta$ requires lowering $\eta$. Nesterov's look-ahead damps the ringing
and carries guarantees beyond quadratics without extra work per step. What
momentum cannot do is give different coordinates different step sizes.

## Exercises

1. [code] **Momentum and learning rate together.** On the valley
   `f_valley` with `momentum_2d`, run a grid of
   $\beta \in \{0, 0.25, 0.5, 0.8\}$ and
   $\eta \in \{0.1, 0.3, 0.6, 1.0\}$ for up to 1000 steps each.
    1. Record for every cell the number of steps until
       $\|\mathbf{x}_t\| \leq 10^{-3}$, or the fact that the iterates
       diverged, and display the grid as a table or heatmap.
    1. Compare the fastest $\beta$ at each stable $\eta$ with the
       $\beta^\star$ of :eqref:`eq_momentum-optimal` for $\kappa = 20$,
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
   gradient there, $\mathbf{g}_t = \nabla f(\mathbf{y}_{t-1})$. Starting
   from $\mathbf{y}_0 = \mathbf{x}_0$ and $\mathbf{v}_0 = \mathbf{0}$, they
   update
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad
   \mathbf{y}_t = \mathbf{y}_{t-1} - \eta\,(\mathbf{g}_t + \beta \mathbf{v}_t),$$
   so that no extra gradient evaluation is needed.
    1. Express $\mathbf{y}_t$ through the $\mathbf{x}_t$ and $\mathbf{v}_t$
       of :eqref:`eq_nesterov` so that both recurrences produce the same
       velocities, and prove the relation by induction over $t$.
    1. A checkpoint saves the library's parameters $\mathbf{y}_t$. Which
       point of :eqref:`eq_nesterov` does it store, and what else must be
       saved to recover $\mathbf{x}_t$?
   :exlabel:`ex_momentum-two-forms-nesterov-momentum`

1. **Stability of momentum.** ● The effective-step reading of
   :numref:`subsec_momentum-timescale` suggests that raising $\beta$ forces
   $\eta$ down. Does the largest stable $\eta$ in the grid of
   problem :exref:`ex_momentum-momentum-learning-rate-together` behave that
   way? Derive the stability condition of :eqref:`eq_momentum` for a single
   eigendirection with curvature $\lambda$ and reconcile the two.

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
   $g_t = \lambda x_{t-1} + \xi_t$.
    1. Compare plain SGD at learning rate $\alpha$ with momentum
       $\beta = 0.9$ at $\eta = \alpha (1-\beta)$, which keeps the
       effective step $\eta/(1-\beta)$ equal to $\alpha$. Compute the
       ratio of the two stationary variances of $x_t$ for
       $\alpha\lambda \in \{0.05, 0.2\}$, by simulation or by solving for
       the stationary covariance of $(x_t, v_t)$. Does the reduction of the
       velocity's noise-to-signal ratio from problem
       :exref:`ex_momentum-velocity-noise` carry over to the iterate?
       Explain the result.
    1. On the airfoil data, rebuild `data_iter` and `feature_dim` with
       `d2l.get_airfoil(batch_size=b)` for $b \in \{1, 10, 100\}$. Run
       `d2l.train_with_rule` with `sgd_momentum` and `num_epochs=10` at
       $\beta \in \{0, 0.9\}$, choosing $\eta$ so that
       $\eta/(1-\beta) = 0.05$. Report the final loss of each run and the
       spread of the recorded losses over its last epoch, and compare the
       effect of momentum with the prediction of the first item.
   :exlabel:`ex_momentum-momentum-under-minibatch-noise`

<!-- WRITER: item 2 of the exercise above uses d2l.get_airfoil with d2l.train_with_rule and sgd_momentum, which no longer exist (train_with_rule is deleted from the library; sgd_momentum was cut with the airfoil cells). Rewrite it on the network of "Momentum on a Network": Batches(X, y, b) for b in {32, 256, 6000}, beta in {0, 0.9} at a matched eta/(1-beta), d2l.lr_sweep, final loss and spread of the last 100 losses. Worklist: also add the averaged form m_t = beta m_{t-1} + (1-beta) g_t and PyTorch's dampening. -->

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1070)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1071)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.4]{.kicker}

Momentum<br>**leaky averages · the timescale of β · acceleration · Nesterov's look-ahead**
:::
:::

::: {.slide title="Gradient Descent Needs 139 Steps on a Valley with κ = 20"}
[The problem]{.kicker}

The valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$, from $(-5, -2)$, at the
best learning rate $\eta = 2/(0.2 + 4) \approx 0.476$:

| target | steps |
|:--|--:|
| both coordinates below 1% of their start | 47 |
| both coordinates below $10^{-6}$ of their start | 139 |

::: {.d2l-note .warn}
The steep wall caps $\eta$ below 0.5; at that rate the flat floor shrinks
by less than 10% per step. The step count grows like $\kappa$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the heavy-ball update and unroll it into an average of gradients;
- read $\beta$ as a timescale and convert it to an effective step;
- compute the optimal $\eta^\star$, $\beta^\star$ and the contraction
  factor for a quadratic;
- tell overdamped from underdamped momentum;
- explain Nesterov's look-ahead.

**Running example:** the valley, curvatures $0.2$ and $4$, $\kappa = 20$;
then the airfoil regression data.
**The idea:** average gradients over time.
:::

::: {.slide title="Momentum Is the Fourth Stop of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| Gradient Descent | step size against curvature |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| **Momentum** | **averaging over time** |
| Adam and AdamW | a step size per coordinate, decoupled decay |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** gradient descent on the valley; minibatch SGD on the airfoil
  data.
- **Skips:** proofs of the rates (math appendix); momentum inside Adam
  (next deck).
:::
:::
:::

::: {.slide title="Across the Valley Gradients Alternate; Along It They Agree"}
[The valley]{.kicker}

Gradient descent, $\eta = 0.4$, 20 steps: factor $1 - \eta h$ is $0.92$
along $x_1$ and $-0.6$ along $x_2$.

<!-- WRITER: cell momentum-an-ill-conditioned-problem-1 is cut (MOM-2); this slide loses its plot: fold its two factors into the next slide or drop it. -->
:::

::: {.slide title="At η = 0.6 Gradient Descent Diverges Along the Steep Wall"}
[The valley]{.kicker}

@!momentum-an-ill-conditioned-problem-2

- Along $x_2$: $1 - 0.6 \cdot 4 = -1.4$; each step throws the iterate
  farther up the opposite wall.
- Along $x_1$: $1 - 0.6 \cdot 0.2 = 0.88$; twice as close as at
  $\eta = 0.4$.

::: {.d2l-note}
An average of past gradients would keep the agreeing $x_1$ components and
cancel the alternating $x_2$ ones.
:::
:::

::: {.slide title="The Velocity Is a Leaky Sum of Past Gradients"}
[The momentum method]{.kicker}

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t,\qquad
\mathbf{x}_t = \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t,\qquad
\mathbf{v}_0 = \mathbf{0}$$

Unrolled: $\mathbf{v}_t = \sum_{\tau=0}^{t-1} \beta^{\tau}\, \mathbf{g}_{t-\tau}$.

- Components that agree accumulate, up to $\frac{1}{1-\beta}$ times.
- Components that alternate cancel.
- $\beta = 0$ is gradient descent; $1 - \beta$ acts as friction on a
  heavy ball.

[Polyak, 1964]{.small}
:::

::: {.slide title="With β = 0.5, the Divergent Learning Rate Converges"}
[The momentum method]{.kicker}

$\eta = 0.6$, $\beta = 0.5$, 20 steps:

@!momentum-the-momentum-method-1
:::

::: {.slide title="β = 0.9 Averages About the Last 10 Gradients"}
[The timescale of β]{.kicker}

@!momentum-effective-sample-weight

- Window $\frac{1}{1-\beta}$; the gradients inside it carry 63–75% of
  the weight.
- Effective step in persistent directions: $\eta_\textrm{eff} = \frac{\eta}{1-\beta}$.
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

Hessian eigenvalues in $[\lambda_{\min}, \lambda_{\max}]$,
$\kappa = \lambda_{\max}/\lambda_{\min}$:

$$\eta^\star = \frac{4}{(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2},
\qquad \beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$$

| | factor per step | steps for $\epsilon$ |
|:--|:--:|:--:|
| gradient descent | $\frac{\kappa - 1}{\kappa + 1}$ | $\frac{\kappa}{2}\log\frac{1}{\epsilon}$ |
| heavy ball | $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ | $\frac{\sqrt{\kappa}}{2}\log\frac{1}{\epsilon}$ |

[Polyak, 1964; proofs in the math appendix]{.small}
:::

::: {.slide title="On the Valley, Tuned Momentum Needs 40 Steps Instead of 139"}
[Acceleration]{.kicker}

$\kappa = 20$: $\eta^\star \approx 0.67$, $\beta^\star \approx 0.40$,
factor $\sqrt{\beta^\star} \approx 0.635$.

@!momentum-acceleration-and-damping-1

Mode with curvature $\lambda$: $x_{t+1} = (1 + \beta - \eta\lambda)x_t - \beta x_{t-1}$;
the factor is the largest root of $z^2 - (1 + \beta - \eta\lambda)z + \beta$.
:::

::: {.slide title="Past Critical Damping the Iterate Rings Around the Minimum"}
[Acceleration]{.kicker}

$\eta = 0.3$ (stable for gradient descent), $\beta = 0.8$: complex roots of
size $\sqrt{0.8} \approx 0.894$ in both modes.

@!momentum-acceleration-and-damping
:::

::: {.slide title="Critical Damping Is the Fastest Setting"}
[Acceleration]{.kicker}

![](../img/mdl-opt-critical-damping.svg){width=55%}

Critical value of one mode: $\beta_\textrm{c} = (1 - \sqrt{\eta\lambda})^2$.
Below: overdamped, creeps like gradient descent. Above: underdamped, rate
$\sqrt{\beta}$. At $\eta^\star$ both extreme modes are critical at $\beta^\star$.
:::

::: {.slide title="Check: A Condition Number of 10,000"}
[Check]{.kicker}

For $\kappa = 10^4$, what is $\beta^\star$, and how many steps do gradient
descent and tuned momentum need per factor of 100?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\sqrt{\kappa} = 100$, so $\beta^\star = (99/101)^2 \approx 0.96$.
Factors $0.9998$ and $0.980$ per step: about 23,000 steps against about
230.
:::
:::
:::

<!-- WRITER: the two airfoil slides ("Momentum Needs One Buffer per Parameter", "At a Matched Effective Step, Momentum and SGD Train Alike") are cut with their cells. The two slides below carry the network result; titles and takeaways to be written from the measured numbers (full batch: momentum gains in steps; CNN at b = 32: tie at a tenfold smaller rate). -->

::: {.slide title="Momentum on Data"}
[Momentum on data]{.kicker}

SGD with $\beta = 0$ and $\beta = 0.9$, best rate of a factor-2 grid:
softmax regression at full batch (60,000 images), `FashionCNN` at 32
images per batch:

@!momentum-momentum-on-a-network-2
:::

::: {.slide title="Three Repeats at the Best Rates"}
[Momentum on data]{.kicker}

@!momentum-momentum-on-a-network-3
:::

::: {.slide title="Nesterov Takes the Gradient Where the Velocity Is Headed"}
[Nesterov momentum]{.kicker}

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{x}_{t-1} - \eta \beta\, \mathbf{v}_{t-1}),\qquad
\mathbf{x}_t = \mathbf{x}_{t-1} - \eta\, \mathbf{v}_t$$

![](../img/mdl-opt-nesterov-lookahead.svg){width=52%}

The velocity is corrected before the overshoot, not one step after it.

[Nesterov, 1983]{.small}
:::

::: {.slide title="The Look-Ahead Damps the Ringing"}
[Nesterov momentum]{.kicker}

Same $\eta = 0.3$, $\beta = 0.8$ as the underdamped heavy-ball run:

@!momentum-nesterov-momentum-1

- Libraries store the look-ahead point: no extra gradient evaluation.
- Convex, $\beta \to 1$: error $O(1/k^2)$ against
  $O(1/k)$, optimal among gradient methods.
:::

<!-- WRITER: slide "Nesterov Is One Flag, Off by Default" is cut with cell momentum-nesterov-momentum-2 (airfoil); the flag can go as a bullet on the previous slide. The goals slide and "Momentum Is the Fourth Stop" still name the airfoil data as running example / use; replace by softmax regression at full batch and FashionCNN at b = 32. -->

::: {.slide title="Momentum Turns κ into √κ"}
[Wrap-up]{.kicker}

- $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t$,
  $\mathbf{x}_t = \mathbf{x}_{t-1} - \eta \mathbf{v}_t$: agreeing gradient
  components add up, alternating ones cancel.
- $\beta$ is a window of $\frac{1}{1-\beta}$ steps; the effective step is
  $\frac{\eta}{1-\beta}$.
- Tuned on the valley: factor $0.635$ instead of $0.905$, 40 steps instead
  of 139.
- Past critical damping: ringing at rate $\sqrt{\beta}$; Nesterov's
  look-ahead damps it.

**Next:** Adam, a step size for each coordinate.
:::
