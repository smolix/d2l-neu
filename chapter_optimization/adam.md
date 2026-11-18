# Adam
:label:`sec_adam`

Of the three learning rates that we tried for plain SGD on `TinyLM`, 0.5
worked best. The library's AdamW, which we called in
:numref:`sec_test_models` before we could explain it, was given 0.003 and
ended half a nat lower. A rate more than a hundred times smaller did not
train a hundred times more slowly. Evidently the learning rate of this
optimizer does not mean what the learning rate of SGD means.

Momentum (:numref:`sec_momentum`) explained one of the two things that
AdamW does, the averaging of gradients over steps. It left the other as a
question: a network does not tell us its curvatures, so can the sizes of a
coordinate's past gradients stand in for them? Run without weight decay,
as it was in that experiment, AdamW is the method called Adam
:cite:`Kingma.Ba.2014`. Adam answers by giving every parameter a learning
rate of its own, set from the size of that parameter's past gradients. The
idea is older than its use on valleys. It began with features that occur
rarely, such as uncommon words, and it reached Adam in three steps, each a
repair of the one before.

Adam is the standard optimizer for transformers and the baseline against
which newer methods are measured. We derive it on the valley, where every
step can be checked by hand, and ask two questions there: what does its
learning rate measure, and does the iterate come to rest? We then return to
`TinyLM`, measure how much the gradients of its parameters differ in size,
and compare Adam with SGD with momentum, this time with a grid of learning
rates for each.

```{.python .input #adam}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #adam}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #adam-adam-and-adamw-1}
%%tab pytorch
import copy
import random
```

## Per-Coordinate Learning Rates

### AdaGrad

Consider learning problems whose features occur with very different
frequencies. This is the normal state of affairs in language: we will see
the word *learning* far more often than the word *preconditioning*.
Parameters tied to rare features receive informative gradients only when
those features occur. Now take one of the two remedies for gradient noise
that :numref:`sec_sgd` named, a learning rate that falls during training.
Under it the parameters of frequent features approach their optima
quickly, while those of rare features are still waiting for evidence. When
the evidence arrives, the shared rate has decayed too far to use it. One
clock cannot serve both populations.

AdaGrad :cite:`Duchi.Hazan.Singer.2011` gives every coordinate its own
clock. The step of a coordinate no longer decays with the iteration count
$t$. It decays with that coordinate's own activity, measured by the
running sum of its squared gradients:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_adagrad`

Here $\mathbf{g}_t$ is the gradient at $\mathbf{x}_t$, all operations are
elementwise, $\mathbf{s}_0 = \mathbf{0}$, and a small $\epsilon > 0$ guards
the division. A coordinate whose gradients have been large moves cautiously
from now on. A coordinate that has barely been touched keeps a large step
and can respond as soon as its rare feature appears.

The same rule addresses the valley, and with it the question that Momentum
left open. On a quadratic whose axes are the coordinate axes, the gradient
of a coordinate is its curvature times its distance from the minimum. In
the ratio of a gradient to the size of the gradients before it, the
curvature cancels. AdaGrad thus does the work of the diagonal
preconditioner of :numref:`subsec_gd-preconditioning`, and it needs
nothing but gradients to do so. What the ratio keeps is the present
distance measured against the earlier ones. The step no longer knows how
far away the minimum is, a side effect that will occupy us for the rest of
the section. (:numref:`subsec_mdl-per-coordinate` makes both readings of
AdaGrad precise.)

We run AdaGrad on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of
:numref:`sec_optimization-intro`, whose curvatures are $0.2$ and $4$, from
$(-5, -2)$ at $\eta = 0.4$, close to the ceiling of $2/4 = 0.5$ for
gradient descent. The function `d2l.train_2d` of :numref:`sec_gd` carries
the two sums from step to step and prints the last iterate. Two properties
of :eqref:`eq_adagrad` let us predict the outcome. On the first step
$\mathbf{s}_1 = \mathbf{g}_1^2$, so every coordinate moves by $\eta$
whatever its gradient, and the iterate lands on $(-4.6, -1.6)$. After
that, as long as the gradient of a coordinate does not grow, its sum is at
least $t$ times the current squared gradient, and step $t$ is at most
$\eta/\sqrt{t}$ long. Twenty steps therefore cover at most
$\eta \sum_{t=1}^{20} t^{-1/2} \approx 7.6\,\eta = 3.0$ in each coordinate,
which is not enough for the distance of $5$ along $x_1$.

```{.python .input #adam-per-coordinate-learning-rates-1}
%%tab pytorch, jax
def adagrad_2d(x1, x2, s1, s2, f_grad, eta=0.4, eps=1e-6):
    g1, g2 = f_grad(x1, x2)
    s1, s2 = s1 + g1 ** 2, s2 + g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

d2l.show_trace_2d(d2l.f_valley, d2l.train_2d(adagrad_2d, d2l.valley_grad))
```

After 20 steps $x_1$ is at $-2.38$: it has used $2.62$ of its budget of
$3.0$. The path is smooth, because the steep coordinate $x_2$ is scaled
down automatically and does not oscillate, and the run is slow for a reason
that has nothing to do with curvature. Under AdaGrad the learning rate is a
travel budget, about $7.6\,\eta$ over the first twenty steps in every
coordinate, and it has to be chosen from the distance to be covered. The
ceiling that curvature put on gradient descent is gone. At $\eta = 2$, four
times that ceiling, the budget is $15$, and the first step moves both
coordinates by $2$, which takes $x_2$ onto the floor of the valley at once.

```{.python .input #adam-per-coordinate-learning-rates-2}
%%tab pytorch, jax
d2l.show_trace_2d(d2l.f_valley, d2l.train_2d(adagrad_2d, d2l.valley_grad,
                                              eta=2))
```

The plot has a corner at $(-3, 0)$, where the first step ends, and from
there the iterate runs along the floor to $x_1 = -0.002$ at step 20. A
learning rate at which gradient descent diverges is harmless here, because
$\eta$ never multiplies a bare gradient. It multiplies the ratio of a
gradient to the size of the gradients before it.

### RMSProp

AdaGrad's sum never forgets. Under persistent gradient noise
$\mathbf{s}_t$ grows roughly linearly in $t$, so the step decays like
$t^{-1/2}$ whether or not the optimization is anywhere near done. This
decay suits the convex problems that AdaGrad was designed for, but it can
stop a network that still has far to travel late in training.

RMSProp :cite:`Tieleman.Hinton.2012` keeps the division and replaces the
sum with an exponential moving average, in which old squared gradients
fade as old gradients fade in the velocity of momentum:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_rmsprop`

We keep the letter $\mathbf{v}_t$ that the literature uses for this
average; it is not the velocity of :numref:`sec_momentum`. The average
estimates the *current* size of the squared gradient rather than its
lifetime total, over a window of about $1/(1-\beta_2)$ steps
(:numref:`subsec_momentum-timescale`), ten at the original
$\beta_2 = 0.9$. The step no longer decays by construction, so at
$\eta = 0.4$, where AdaGrad ran out of budget, RMSProp should arrive.

```{.python .input #adam-rmsprop-forgetting-on-purpose-1}
%%tab pytorch, jax
def rmsprop_2d(x1, x2, v1, v2, f_grad, eta=0.4, beta2=0.9, eps=1e-6):
    g1, g2 = f_grad(x1, x2)
    v1 = beta2 * v1 + (1 - beta2) * g1 ** 2
    v2 = beta2 * v2 + (1 - beta2) * g2 ** 2
    x1 -= eta / math.sqrt(v1 + eps) * g1
    x2 -= eta / math.sqrt(v2 + eps) * g2
    return x1, x2, v1, v2

d2l.show_trace_2d(d2l.f_valley, d2l.train_2d(rmsprop_2d, d2l.valley_grad))
```

After 20 steps $x_1$ is within $0.011$ of the minimum. The first step in
the plot is conspicuously long. Since $\mathbf{v}_0 = \mathbf{0}$, the
average after one step is $\mathbf{v}_1 = 0.1\, \mathbf{g}_1^2$, a tenth of
the squared gradient, and each coordinate moves by
$\eta/\sqrt{0.1} \approx 3.2\,\eta$ instead of $\eta$. An average that
starts at zero needs about ten steps to fill its window, a fault that Adam
will repair.

Twenty steps hide a second consequence of forgetting, and it is the more
important one. A step that is divided by the recent size of the gradient
does not get shorter when the gradient does. To see what this does near the
minimum, we run both methods for 200 steps and print the last four values
of $x_1$.

```{.python .input #adam-rmsprop-1}
%%tab pytorch, jax
def last_x1(path, k=4):
    return ' '.join(f'{x1:.3f}' for x1, _ in path[-k:])

for name, rule in [('AdaGrad', adagrad_2d), ('RMSProp', rmsprop_2d)]:
    print(f'{name}, ', end='')
    path = d2l.train_2d(rule, d2l.valley_grad, steps=200)
    print(f'  x1 over the last four steps: {last_x1(path)}')
```

AdaGrad is still creeping toward the minimum, by about $0.001$ per step.
RMSProp has gone the other way. It is farther from the minimum than it was
at step 20, it hops between $-0.2$ and $0.2$ in both coordinates, and no
number of further steps will bring it closer. The pattern can be read off
:eqref:`eq_rmsprop`. If a coordinate alternates between $-a$ and $a$, its
gradient has the same size at every step, $v_t$ equals the squared
gradient, and the update is $\eta$ times a sign. A jump of length $\eta$
takes $-a$ to $a$ if $a = \eta/2 = 0.2$. The curvature does not enter,
which is why both coordinates hop in the same band.

Even the minimum cannot hold a method that forgets. As the gradient
shrinks, its average shrinks after it, the coordinate's rate
$\eta/\sqrt{v_t}$ grows, and sooner or later it passes the ceiling of 2
divided by the curvature (:numref:`sec_gd`) and throws the iterate out
again. The run shows a property of every method that divides by the recent
size of the gradient, and the chapter will need it again: *a normalized
step does not shrink with the gradient*. Once the gradients of a coordinate
have a steady size, its step is about $\eta$ long, however close the
minimum is. At a constant learning rate such a method does not settle.

### Adam

Neither AdaGrad nor RMSProp is what one reaches for today to train a
network, and the reader may treat both as steps on the way to Adam. Adam
changes RMSProp in two places. It averages the gradient as well as its
square, which builds momentum into the method. And it divides each average
by the total weight that the average has accumulated so far, which removes
the fault of the first steps:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1 - \beta_1^t), \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1 - \beta_2^t).
\end{aligned}
$$
:eqlabel:`eq_adam-moments`

In the names of the Adam paper, $\mathbf{m}_t$ is the *first-moment
estimate*, which is the momentum average, and $\mathbf{v}_t$ the
*second-moment estimate*, an average of squared gradients. (A variance
would subtract the mean of the gradients first; $\mathbf{v}_t$ averages
the squares as they are.) The divisions that produce the hatted quantities
are the *bias correction*. The square root of $\hat{\mathbf{v}}_t$ has the
units of a gradient, and we call it the *scale* of a coordinate. The
update divides the corrected momentum by the scale:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_adam-update`

Each coordinate moves by $\eta$ times the ratio of its average gradient to
its scale. Note one difference from :numref:`sec_momentum`, where the
method *adds* each gradient to $\beta$ times its buffer. The factor
$1 - \beta_1$ turns that sum into an *average*, which amounts to the same
method with the learning rate multiplied by $1 - \beta_1$.

The defaults are $\beta_1 = 0.9$ and $\beta_2 = 0.999$: the direction is
averaged over about ten steps, the scale over about a thousand. In
practice $\beta_1$ is rarely touched. It is $\beta_2$ that recipes change,
and those for language models since GPT-3 :cite:`brown2020language` often
use $\beta_2 = 0.95$, a window of twenty steps. The scale then follows
changes in the gradients more quickly, and is itself noisier.

The bias correction is not a heuristic. Unrolled from
$\mathbf{v}_0 = \mathbf{0}$, the average
$\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^{t} \beta_2^{t-s}\, \mathbf{g}_s^2$
is a weighted sum whose weights total $1 - \beta_2^t$, not $1$. If the
squared gradients have a steady mean, $\mathbf{v}_t$ carries the fraction
$1 - \beta_2^t$ of it in expectation, and dividing by that fraction removes
the deficit at every $t$. The deficit is large: at $\beta_2 = 0.999$, after
ten steps $\mathbf{v}_t$ holds about $1\%$ of its stationary value. Without
the correction the first step would be
$(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/\sqrt{0.001} \approx 3.2$ times $\eta$,
about as large as the first step of RMSProp. With it,
$\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and
$\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the first step moves every
coordinate by $\eta$, as under AdaGrad.

Adam also adds $\epsilon$ *outside* the square root, where
:eqref:`eq_rmsprop` had it inside, so that $\epsilon$ is itself the
smallest value of the denominator. The usual default is $10^{-8}$, and
:numref:`subsec_adam-variance-estimate` returns to what such a floor does.

### The Learning Rate as a Step Length

The update lets us predict the path of Adam on the valley at $\eta = 0.4$.
The first step is $\eta$ in each coordinate. For the next few steps the
gradients of a coordinate keep their sign. They shrink, but the corrected
momentum is then an average of a few gradients of one sign and the scale
is their root mean square, and the two stay close. Each step is again
about $\eta$ long. Adam should therefore set off at $45^\circ$, $0.4$ per
step in both coordinates, without regard to the shape of the valley. It
reaches the floor $x_2 = 0$ after five or six steps, with $x_1$ still
about $3$ from the minimum, and since a momentum average cannot stop at
once, it must overshoot. The cell runs 200 steps with the defaults and
with the short window of RMSProp, $\beta_2 = 0.9$, and plots the default
run.

```{.python .input #adam-adam-on-the-valley-1}
%%tab pytorch, jax
def adam_2d(x1, x2, s1, s2, f_grad, eta=0.4, beta1=0.9, beta2=0.999,
            eps=1e-8):
    def update(x, g, state):  # state: (m, v, t), 0 before the first step
        m, v, t = state if state else (0, 0, 0)
        m, v = beta1 * m + (1 - beta1) * g, beta2 * v + (1 - beta2) * g ** 2
        m_hat, v_hat = m / (1 - beta1 ** (t + 1)), v / (1 - beta2 ** (t + 1))
        return x - eta * m_hat / (math.sqrt(v_hat) + eps), (m, v, t + 1)
    g1, g2 = f_grad(x1, x2)
    (x1, s1), (x2, s2) = update(x1, g1, s1), update(x2, g2, s2)
    return x1, x2, s1, s2

paths = {}
for beta2 in [0.999, 0.9]:
    print(f'beta2 = {beta2}, ', end='')
    paths[beta2] = d2l.train_2d(adam_2d, d2l.valley_grad, steps=200,
                                beta2=beta2)
    print(f'  x1 over the last four steps: {last_x1(paths[beta2])}')
d2l.show_trace_2d(d2l.f_valley, paths[0.999])
```

The plot shows the march and the overshoot: a straight run at $45^\circ$, a
swing up to $x_2 \approx 0.8$ and past the minimum to $x_1 \approx 1$, and
a spiral back. After 200 steps the default run is within $10^{-4}$ of the
minimum. The run with the ten-step window is not: its $x_1$ keeps changing
sign within about $0.02$ of the minimum, the hop of RMSProp again, narrower
only because gradients of alternating sign nearly cancel in the momentum
average.

Has the default run come to rest, then? It has not. A window of a thousand
steps still remembers the large gradients of the start, so the scale stays
large while the gradients shrink. That memory runs out as well, and
thousands of steps later the default run is thrown out of the minimum as
RMSProp was (Exercise :exref:`ex_adam-constant-rate`). A longer window
postpones the departure and does not prevent it.

This is not a fast descent. Gradient descent at its best rate shrinks both
coordinates by the factor $0.905$ per step
(:numref:`sec_optimization-intro`) and would be within
$5 \cdot 0.905^{200} \approx 10^{-8}$ of the minimum by step 200. What Adam
offers is that nothing in its path depended on the curvatures. To see why,
multiply the loss by 1,000. Every gradient grows by that factor, and so do
$\hat{\mathbf{m}}_t$ and $\sqrt{\hat{\mathbf{v}}_t}$. Their ratio does not
change, so Adam takes the same steps on $f$ and on $1000 f$, as long as
$\epsilon$ is negligible beside the scale. The argument works for one
coordinate at a time, since each is divided by its own scale. Now the
gradient of the valley is that of the round bowl
$\frac{1}{2}(x_1^2 + x_2^2)$ with its two coordinates multiplied by $0.2$
and $4$. On the valley Adam therefore takes the steps that it would take
on the bowl: the condition number of 20, which set the pace of gradient
descent and the best setting of momentum, has dropped out of the problem.
Like the diagonal preconditioner of :numref:`subsec_gd-preconditioning`,
the invariance is tied to the coordinate axes, and a valley at an angle to
them keeps its condition number
(Exercise :exref:`ex_adam-rotation`).

This invariance is the main result of the section, because it says what
the learning rate of Adam is. Under gradient descent $\eta$ multiplies a
gradient, and how far a step moves a weight depends on the size of that
gradient. Under Adam $\eta$ is a length in parameter space. A coordinate
whose gradients agree from step to step moves by about $\eta$ per step, a
coordinate whose gradients disagree moves by less, and the size of the
gradients does not matter. This is why one default, $\eta = 10^{-3}$,
serves models whose gradients differ by orders of magnitude. It also
resolves the puzzle of the opening: the rate $0.5$ of SGD and the rate
$0.003$ of AdamW cannot be compared, because they do not have the same
units. A learning rate for Adam is chosen like the travel budget of
AdaGrad, from how far the weights have to move.

### Adam from Scratch

The method itself takes a handful of lines: the two averages, their bias
correction, and the division by the scale. Everything around them is
storage, two buffers per parameter and a step counter. We write the update
the way the framework writes its own optimizers, so that it can train any
model.

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab pytorch
class Adam(torch.optim.Optimizer):
    """Adam from scratch: steps scaled by two debiased moving averages."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (beta1, beta2), lr, eps = group['betas'], group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:  # A parameter that received no gradient
                    continue
                state = self.state[p]
                if not state:  # First step: t = 0 and m = v = 0
                    state.update(t=0, m=torch.zeros_like(p),
                                 v=torch.zeros_like(p))
                state['t'] += 1
                t, m, v = state['t'], state['m'], state['v']
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).add_(p.grad ** 2, alpha=1 - beta2)
                m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
```

```{.python .input #adam-adam-both-moments-debiased-1}
%%tab jax
def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8):
    """Adam's direction m_hat / (sqrt(v_hat) + eps), from scratch."""
    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return dict(t=jnp.zeros((), jnp.int32), m=zeros, v=zeros)

    def update(grads, state, params=None):
        t = state['t'] + 1
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g,
                         state['m'], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2,
                         state['v'], grads)
        def direction(m, v):
            return m / (1 - b1 ** t) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        return jax.tree.map(direction, m, v), dict(t=t, m=m, v=v)
    return optax.GradientTransformation(init, update)

def adam(learning_rate):
    return optax.chain(scale_by_adam(), optax.scale(-learning_rate))
```

:begin_tab:`jax`
An Optax optimizer is a chain of gradient transformations. `scale_by_adam`
turns the gradient into Adam's direction, and `optax.scale` multiplies it
by $-\eta$. Keeping the two apart lets the next section insert weight decay
between them.
:end_tab:

Before using our version we check it against the library on a small
least-squares problem: two copies of the same linear model, the same data,
100 steps each.

```{.python .input #adam-adam-both-moments-debiased-2}
%%tab pytorch
def compare(ours, theirs, lr, num_steps=100):
    X, y = torch.randn(512, 20), torch.randn(512, 1)
    net = nn.Linear(20, 1)
    twin = copy.deepcopy(net)
    optimizers = [ours(net.parameters(), lr), theirs(twin.parameters(), lr)]
    w0 = net.weight.detach().clone()
    for t in range(num_steps):
        for model, optimizer in zip([net, twin], optimizers):
            loss = F.mse_loss(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if t == 0:
            first = (net.weight - w0).abs() / lr
            print(f'first step / lr: {first.min():.4f} to {first.max():.4f}')
    print(f'largest difference after {num_steps} steps: '
          f'{(net.weight - twin.weight).abs().max():.1e}')

compare(Adam, torch.optim.Adam, lr=0.01)
```

```{.python .input #adam-adam-both-moments-debiased-2}
%%tab jax
@nnx.jit
def least_squares_step(model, optimizer, X, y):
    grads = nnx.grad(lambda model: ((model(X) - y) ** 2).mean())(model)
    optimizer.update(model, grads)

def compare(ours, theirs, lr, num_steps=100):
    k_X, k_y, k_w = jax.random.split(d2l.get_key(), 3)
    X, y = jax.random.normal(k_X, (512, 20)), jax.random.normal(k_y, (512, 1))
    net = nnx.Linear(20, 1, rngs=nnx.Rngs(k_w))
    twin = nnx.clone(net)
    optimizers = [nnx.Optimizer(net, ours, wrt=nnx.Param),
                  nnx.Optimizer(twin, theirs, wrt=nnx.Param)]
    w0 = net.kernel[...]
    for t in range(num_steps):
        for model, optimizer in zip([net, twin], optimizers):
            least_squares_step(model, optimizer, X, y)
        if t == 0:
            first = jnp.abs(net.kernel[...] - w0) / lr
            print(f'first step / lr: {first.min():.4f} to {first.max():.4f}')
    diff = jnp.abs(net.kernel[...] - twin.kernel[...]).max()
    print(f'largest difference after {num_steps} steps: {diff:.1e}')

compare(adam(0.01), optax.adam(0.01), lr=0.01)
```

The first step moves every weight by $\eta$, as the bias correction
promises, and after 100 steps the two copies agree to within rounding
error. From now on we call the library version.

## Adam on a Language Model

The valley has two coordinates and no noise. For a network we turn to
`TinyLM` (:numref:`subsec_tinylm`), the small transformer of
:numref:`sec_test_models`: two blocks, about 0.47 million parameters,
trained to predict the next byte of Wikipedia text from the 64 bytes before
it, on minibatches of 64 such windows. Transformers are the models for
which Adam became the standard, and this one is small enough to train
dozens of times.

```{.python .input #adam-adam-from-scratch-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

### Gradient Scales in `TinyLM`

The two coordinates of the valley began with gradients of size $1$ and $8$.
How unequal are the parameters of a network? We measure the root mean
square (RMS) of the gradient entries of `TinyLM` over ten minibatches,
separately for each kind of parameter, at initialization and again after
500 steps of Adam at $\eta = 3 \times 10^{-3}$. We also record how far
Adam then moves the weights in a single step.

```{.python .input #adam-gradient-scales-in-a-network-1}
%%tab pytorch
def gradients(model, data, num_batches=10):
    """RMS over num_batches minibatches of each parameter's gradient."""
    device = d2l.try_gpu()
    sq = {name: 0 for name, _ in model.named_parameters()}
    for _, (X, Y) in zip(range(num_batches), data.train_dataloader()):
        model.zero_grad()
        loss = F.cross_entropy(model(X.to(device)).flatten(0, 1),
                               Y.to(device).flatten())
        loss.backward()
        for name, p in model.named_parameters():
            sq[name] += p.grad ** 2 / num_batches
    return {name: v.sqrt() for name, v in sq.items()}

def snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}

model = d2l.TinyLM(len(data.vocab)).to(d2l.try_gpu())
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
```

```{.python .input #adam-gradient-scales-in-a-network-1}
%%tab jax
def flat(state):  # {'blks.0.qkv.kernel': array, ...}
    return {'.'.join(map(str, path)): v[...]
            for path, v in nnx.to_flat_state(state)}

@nnx.jit
def gradient(model, X, Y):
    loss = lambda model: optax.softmax_cross_entropy_with_integer_labels(
        model(X), Y).mean()
    return nnx.grad(loss)(model)

def gradients(model, data, num_batches=10):
    """RMS over num_batches minibatches of each parameter's gradient."""
    sq = {}
    for _, (X, Y) in zip(range(num_batches), data.train_dataloader()):
        for name, g in flat(gradient(model, jnp.asarray(X),
                                     jnp.asarray(Y))).items():
            sq[name] = sq.get(name, 0) + g ** 2 / num_batches
    return {name: jnp.sqrt(v) for name, v in sq.items()}

def snapshot(model):
    return flat(nnx.state(model, nnx.Param))

model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_seed()))
optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
```

```{.python .input #adam-gradient-scales-in-a-network-2}
%%tab pytorch, jax
KINDS = ['token_emb', 'pos_emb', 'qkv', 'proj', 'mlp', 'head']

def rms_by_kind(tensors):
    """RMS of the entries of each kind of parameter, from {name: tensor}."""
    sums, counts = {}, {}
    for name, t in tensors.items():
        kind = (('LayerNorm' if 'norm' in name else 'biases')
                if len(t.shape) == 1 else
                next(k for k in KINDS if k in name))
        sums[kind] = sums.get(kind, 0) + float((t ** 2).sum())
        counts[kind] = counts.get(kind, 0) + math.prod(t.shape)
    return {kind: math.sqrt(sums[kind] / counts[kind]) for kind in sums}

grad_init = rms_by_kind(gradients(model, data))
d2l.train_steps(model, data, optimizer, num_steps=500)
grad_500 = rms_by_kind(gradients(model, data))
before = snapshot(model)
d2l.train_steps(model, data, optimizer, num_steps=1)
after = snapshot(model)
step = rms_by_kind({name: (before[name] - after[name]) / 3e-3
                    for name in before})
print(f'{"":<11}{"RMS gradient at step 0":>24}{"at step 500":>13}'
      f'{"Adam step / lr":>16}')
for kind in KINDS + ['LayerNorm', 'biases']:
    print(f'{kind:<11}{grad_init[kind]:>24.1e}{grad_500[kind]:>13.1e}'
          f'{step[kind]:>16.2f}')
ratio = lambda column: max(column.values()) / min(column.values())
print(f'{"max / min":<11}{ratio(grad_init):>24.0f}{ratio(grad_500):>13.0f}'
      f'{ratio(step):>16.1f}')
```

The rows refine the three kinds of parameters of :numref:`subsec_tinylm`:
the embeddings of tokens and of positions; the matrices `qkv` and `proj`
of each attention layer, `mlp` of the MLPs, and `head`, the output layer;
the LayerNorm scales and shifts; and the biases. The first two columns
hold the quantity that Adam's scale estimates, which under SGD is also the
step per unit of $\eta$. The last column is the RMS of the step that Adam
takes next, divided by $\eta$, and the bottom row divides the largest
entry of each column by the smallest.

At initialization the gradients of the eight kinds differ about fortyfold
in size, from about $2 \times 10^{-4}$ for the embeddings to about
$7 \times 10^{-3}$ for the biases, and after 500 steps they still differ
about fifteenfold. The embeddings are at the bottom, the token embeddings
partly for the reason that motivated AdaGrad: a row of the embedding
matrix receives a gradient only when its byte occurs in the minibatch.
Under SGD these numbers are the steps, so at the start a single learning
rate moves a bias about forty times as far as an entry of an embedding. On
the valley that was the whole difficulty: the coordinate with the large
gradients set the ceiling on $\eta$, and the other one crawled. Whether
the embeddings of `TinyLM` crawl under SGD the table cannot say, since a
weight with small gradients may also be one that needs to move little
(Exercise :exref:`ex_adam-per-group-learning-rates` puts it to the test).
Adam's steps, in the last column, lie between $0.1\,\eta$ and
$0.3\,\eta$ for every kind, a spread of about two.

Why a fifth of $\eta$, and not $\eta$? At step 1 the ratio was 1 for every
coordinate that received a gradient. A coordinate keeps the full $\eta$
only while its gradients agree from step to step and hold their size, and
in a network they do neither. Minibatch gradients disagree, and the
momentum average of gradients that keep changing sign is smaller than any
one of them. For pure noise of a steady size it has
$\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23$ times the RMS of a single
gradient, since the squared weights of the average sum to
$(1-\beta_1)/(1+\beta_1)$. The scale, for its part, has a long memory. At
$\beta_2 = 0.999$ it still averages over all 500 steps, and for most kinds
the gradients at initialization were two to four times as large as they
are now. The table cannot tell us the shares of the two effects. The first
of them will not go away, however good the weights become, so at a
constant $\eta$ the weights keep moving: the hop of the valley has its
counterpart in a network.

### Comparison with SGD
:label:`subsec_adam-where-wins`

The comparison follows the protocol of :numref:`sec_test_models`. Adam and
SGD with momentum $0.9$ train `TinyLM` for 2,000 steps at a constant
learning rate, each on its own grid of rates. We read the final training
loss, which on text that a run sees only once is also a held-out loss.
Each grid is run with three seeds, and the runs of a seed share their
initial weights.

The reading of $\eta$ as a step length makes two predictions. The first
says where the good rates lie. Adam's momentum average is
$1 - \beta_1 = 0.1$ times the velocity of :numref:`sec_momentum`, so Adam
at the rate $\eta$ moves a coordinate as SGD with momentum would at the
rate $0.1\,\eta$ divided by that coordinate's scale. With scales between
$10^{-4}$ and $10^{-2}$, the two methods take steps of similar length when
the rate of SGD is ten to a thousand times that of Adam. We expect its
good rates about two orders of magnitude above Adam's. The second
prediction is about failure. Above its ceiling gradient descent feeds on
itself: a large gradient makes a large step, which makes a larger
gradient. The step of Adam stays about $\eta$ however large the gradient
becomes, so we expect Adam to train badly at a rate that is too large, but
not to diverge.

```{.python .input #adam-comparison-on-the-language-model-1}
%%tab pytorch
make_lm = lambda: d2l.TinyLM(len(data.vocab))
make_sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                             momentum=0.9)
make_adam = lambda model, lr: torch.optim.Adam(model.parameters(), lr)
seeds = [random.getrandbits(31) for _ in range(3)]
```

```{.python .input #adam-comparison-on-the-language-model-1}
%%tab jax
make_lm = lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs)
make_sgd = lambda lr: optax.sgd(lr, momentum=0.9)
make_adam = optax.adam
seeds = [d2l.get_seed() for _ in range(3)]
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab pytorch, jax
def sweep(make_optimizer, lrs):
    """Loss curves of one run per learning rate and seed, printing the final
    losses; the runs of one seed start from the same initialization."""
    runs = [d2l.lr_sweep(make_lm, make_optimizer, data, lrs, seed,
                         verbose=False)[0] for seed in seeds]
    for lr in lrs:
        print(f'  lr {lr:<7g}', *(f'{d2l.final_loss(run[lr]):6.3f}'
                                  for run in runs))
    return {lr: [run[lr] for run in runs] for lr in lrs}

print('final loss of three runs, SGD with momentum')
sgd_lm = sweep(make_sgd, [0.1, 0.3, 0.5, 1.0])
print('Adam')
adam_lm = sweep(make_adam, [3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
```

SGD with momentum has one rate that works. At $0.1$ it ends about $0.2$
nats above its result at $0.3$. At $0.5$ most runs diverge, which the table
prints as `inf`, and at $1$ all of them do. Whenever a run survives at
$0.5$, it ends lower than the runs at $0.3$: the best learning rate of SGD
lies where it usually explodes. The table of Adam has a different shape.
Its best rate, $3 \times 10^{-3}$, is a hundred times smaller than that of
SGD, as the first prediction said, and it has a neighbor on either side
within a tenth of a nat. And at $3 \times 10^{-2}$ the loss is poor, about
$2.8$, but finite in every run. Adam stalls where SGD diverges, as the
second prediction said: a step length of $0.03$ is too coarse for this
model, but steps of a fixed length cannot grow.

The next cell condenses the two tables into one line per optimizer, and it
plots the loss of the first seed's run at each best rate.

```{.python .input #adam-the-race-on-the-language-model-3}
%%tab pytorch, jax
def best_rate(curves):
    """The lr with the lowest mean final loss, and each lr's mean."""
    mean = {lr: sum(map(d2l.final_loss, runs)) / len(runs)
            for lr, runs in curves.items()}
    return min(mean, key=mean.get), mean

print(f'{"":6}{"best lr":>8}{"final loss":>17}   {"within 0.15 nats":<18}'
      f'{"steps to 1.8":<16}largest one-step rise')
for name, curves in [('SGD', sgd_lm), ('Adam', adam_lm)]:
    best, mean = best_rate(curves)
    runs = curves[best]
    finals = [d2l.final_loss(run) for run in runs]
    usable = [f'{lr:g}' for lr in curves if mean[lr] <= mean[best] + 0.15]
    steps = ' '.join(str(d2l.steps_to_reach(run, 1.8)) for run in runs)
    rises = ' '.join(f'{max(b - a for a, b in zip(run, run[1:])):.2f}'
                     for run in runs)
    print(f'{name:<6}{best:>8g}{min(finals):>8.3f} to {max(finals):.3f}   '
          f'{" ".join(usable):<18}{steps:<16}{rises}')
best_sgd, best_adam = best_rate(sgd_lm)[0], best_rate(adam_lm)[0]
d2l.plot(list(range(0, 2000, 25)),
         [d2l.smooth(sgd_lm[best_sgd][0]),
          d2l.smooth(adam_lm[best_adam][0])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd:g}',
                 f'Adam, lr {best_adam:g}'])
```

The usable rates are those whose mean final loss lies within $0.15$ nats
of the lowest, the margin of :numref:`sec_test_models`. Adam has three of
them, spanning a factor of ten, and SGD a single one. The next column
gives the step at which each of the three runs at the best rate first
reaches a loss of $1.8$, a level that all six pass well before the end.
Adam is there after 650 to 800 steps and SGD after 1,100 to 1,500. Adam
needs between half and seven tenths of the steps, and the two sets of
numbers do not overlap. The last column is the largest rise of the loss
from one step to the next. For Adam it is below $0.2$ nats in every run.
The worst of SGD's three runs is thrown up by several tenths of a nat or
more in a single step. At its best rate SGD is still jolted now and then,
a mild form of the divergence that waits one grid point higher.

The final losses favor Adam too, in every run, by several hundredths to
about a tenth of a nat, and the two ranges in the table do not overlap.
The plot shows where this lead is made. The curves part within the first
few hundred steps, while SGD lingers near a loss of $2.5$, and from there
on they fall side by side, the gap narrowing a little toward the end. Adam
gains its lead at the start, and SGD does not make it up. Nor can SGD
simply go faster, since above its best rate it diverges.
:numref:`subsec_practice-nan` traces that divergence to a few oversized
gradients in the first steps and removes it by clipping them
(:numref:`sec_rnn-scratch`). SGD then trains at $\eta = 1$ and ends within
a few hundredths of a nat of Adam.

The lead in the final loss is thus the least of Adam's advantages on this
model. The strong ones are a tenfold range of rates that work and a
failure that shows up as a stalled loss, not as `inf`. Our advice for a
new model follows from them: begin with Adam at three rates around
$10^{-3}$, a factor of three apart. The usable window of SGD here was less
than a factor of five wide, and it ended at divergence.

### Sources of the Advantage

Adam owes its standing to models far larger than this one. On large
transformers SGD with momentum falls far behind, while on convolutional
networks of similar size the two stay close
:cite:`Zhang.Chen.Ding.ea.2024`. `TinyLM` is too small to show that gap,
so here we report what others have measured. The explanation that we find
most convincing starts from an identity. Switch off both averages in
:eqref:`eq_adam-update`, $\beta_1 = \beta_2 = 0$ and $\epsilon = 0$, and
the update becomes
$\eta\, \mathbf{g}_t / |\mathbf{g}_t| = \eta\, \mathrm{sign}(\mathbf{g}_t)$.
This is *sign descent*, which moves every coordinate by $\eta$ against the
sign of its gradient and ignores the size. Adam is sign descent with two
averages added, and :citet:`Kunstner.Chen.Lavington.ea.2023` found that
sign descent with momentum shares Adam's lead over SGD on transformers,
with full-batch gradients too, where there is no noise to average away.
What Adam gains, it gains by equalizing the steps of coordinates whose
gradients differ in size, the effect that our table measured.

Two sources of such differences grow as a language model grows. The blocks
of a transformer differ far more in curvature than those of a
convolutional network :cite:`Zhang.Chen.Ding.ea.2024`. And among words or
word pieces most tokens are rare, yet together they make up much of the
text. Gradient descent makes almost no progress on their share of the
loss, while Adam does :cite:`Kunstner.Yadav.Milligan.ea.2024`. `TinyLM`
has two blocks, and its tokens are bytes. The next cell counts how rare
the rare bytes are.

```{.python .input #adam-sources-of-the-gap-1}
%%tab pytorch, jax
text = data.train[::10]  # Every tenth byte of the training text
freq = np.bincount(text, minlength=256) / len(text)
per_batch = freq * 64 * 64  # Expected count in a minibatch of 4,096 targets
rare = (freq > 0) & (per_batch < 0.1)
print(f'{rare.sum()} byte values appear less than once in ten minibatches, '
      f'{freq[rare].sum():.2%} of the text')
print(f'rarest lowercase letter: {per_batch[97:123].min():.1f} per minibatch')
```

Even the rarest lowercase letter occurs two to three times in a minibatch
of 4,096 targets. The 99 byte values that occur less than once in ten
minibatches, most of them pieces of multi-byte characters such as accented
letters, make up $0.07\%$ of the text. `TinyLM` is a mild case on both
counts. Exercises :exref:`ex_adam-frequent-and-rare-tokens` and
:exref:`ex_adam-per-group-learning-rates` test the two explanations on
this model, the first of them also with a vocabulary of word pieces.

## Limits of the Second-Moment Estimate
:label:`subsec_adam-variance-estimate`

The scale $\sqrt{\hat{\mathbf{v}}_t}$ is an estimate built from one noisy
stream of gradients, and its errors are not symmetric: a scale that is too
small produces a step that is too large. The constant $\epsilon$ limits the
damage. It is meant to guard against division by zero, but it also acts as
a floor on the denominator of :eqref:`eq_adam-update`. The learning rate of
a coordinate, the factor $\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ that
multiplies its momentum $\hat{m}_{t,i}$, therefore never exceeds
$\eta/\epsilon$. A small $\epsilon$ trusts the scale even when it is tiny;
a large $\epsilon$ does not. On `TinyLM` the floor is idle at the default:
the scales in our table, about $10^{-4}$ and above, are nearly four orders
of magnitude larger.

In the limit of large $\epsilon$ the denominator is $\epsilon$ for every
coordinate, and Adam becomes
$\mathbf{x}_{t+1} = \mathbf{x}_t - (\eta/\epsilon)\, \hat{\mathbf{m}}_t$.
This is the rule that we met before the sweep, with $\epsilon$ in the
place of the scale: SGD with momentum $\beta_1$ at the learning rate
$(1 - \beta_1)\eta/\epsilon$, up to the bias correction of the first
steps. Values of $\epsilon$ between the default $10^{-8}$ and this limit
interpolate between the two methods, which makes $\epsilon$ a tuning knob
rather than a numerical detail.

The floor does not help against a subtler fault of a scale that forgets.
Give a coordinate the gradients $+4, -1, -1$, over and over. They sum to
$+2$ per period, so the coordinate ought to move down. But a scale with a
short memory divides every gradient by roughly its own size, and each of
the three then moves the coordinate by about $\eta$. With $\beta_1 = 0$
and $\beta_2 = 1/17$, a window of about one step, the three steps are
$-1.03\,\eta$, $+0.74\,\eta$, and $+0.98\,\eta$: a net $+0.69\,\eta$ per
period, in the wrong direction. Normalization has turned a vote weighted
by gradient size into a head count, and the two small gradients win.

:citet:`Reddi.Kale.Kumar.2019` built from this pattern a convex problem,
the losses $4x, -x, -x$ in turn on the interval $[-1, 1]$, on which Adam
converges to $x = 1$, the *worst* point of the interval. The problem is
adversarial and says little about ordinary training data, but it shows
that Adam has no general convergence guarantee. Their repair, AMSGrad,
divides by the running maximum of $\mathbf{v}_t$, which never forgets.
Neither AMSGrad nor later repairs displaced Adam in practice
(Exercise :exref:`ex_adam-two-repairs-variance-estimate` rebuilds the
example, and :numref:`subsec_mdl-adam-counterexample` works through the
construction).

Adam decides the direction of a step and the relative lengths of its
coordinates, and it leaves the length itself, $\eta$, to us. A constant
$\eta$, we have seen, does not let the weights come to rest. On the valley
the iterate hopped across the minimum, and in `TinyLM` the weights were
still moving by a fifth of $\eta$ per step, in root mean square, after 500
steps. The remedy is to lower $\eta$ as the run proceeds, and
:numref:`sec_scheduler` asks how. Before that, the per-coordinate scale
turns out to interfere with a part of the standard recipe that looked
harmless, weight decay.

## Summary

A single learning rate has to suit the coordinates with the largest
gradients, and all the others then move too slowly. Adam gives each
coordinate a rate of its own by dividing a momentum average of the
gradient by the root of an average of its square. AdaGrad contributed the
division, RMSProp the forgetting, and Adam the momentum and the bias
correction. Because the update is a ratio, it does not change when the
loss, or the gradient of a single coordinate, is rescaled. The learning
rate is therefore a step length, about $\eta$ per step for a coordinate
whose gradients agree and less for one whose gradients disagree. In
`TinyLM` the gradients of the eight kinds of parameters differ fifteen- to
fortyfold, and the steps of Adam twofold. Against SGD with momentum, Adam
trains within $0.15$ nats of its best loss over a tenfold range of
learning rates, stalls where SGD diverges, and ends several hundredths to
a tenth of a nat lower. The one thing that Adam does not choose is $\eta$,
and at a constant $\eta$ it does not settle.

## Exercises

1. [code] **Scale invariance and its limit.**
    1. Show that with $\epsilon = 0$ Adam produces the same iterates on $f$
       and on $c f$ for every $c > 0$. Extend the argument to two problems
       whose gradients are related by
       $\nabla \tilde{f}(\mathbf{x}) = \mathbf{C}\, \nabla f(\mathbf{x})$
       for a diagonal matrix $\mathbf{C}$ with positive entries, and
       conclude that Adam takes the same steps on the valley and on the
       round bowl $\frac{1}{2}(x_1^2 + x_2^2)$.
    1. Check the conclusion with `adam_2d`. Then find the step from which
       both coordinates stay below 1% of their starting values, on the
       valley and on a valley whose curvature along $x_2$ is $400$ instead
       of $4$. Compare with gradient descent, `d2l.gd_2d`, at its best
       rate $2/(\lambda_1 + \lambda_2)$ on the same two problems, where
       $\lambda_1$ and $\lambda_2$ are the two curvatures.
    1. With the default $\epsilon = 10^{-8}$, pass `adam_2d` the gradient
       of $c f$ for the valley $f$ and
       $c \in \{10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}\}$. From which $c$ on
       does the path change visibly, and how large are the gradients then,
       compared with $\epsilon$? Which method of this chapter does Adam
       turn into as $c \to 0$, and at which learning rate?

1. [code] **Rotating the coordinates.** Per-coordinate methods are tied to
   the coordinate axes. Rotating the valley `d2l.f_valley` by $45^\circ$
   gives
   $$f(\mathbf{x}) = 0.05 (x_1 + x_2)^2 + (x_1 - x_2)^2,$$
   whose Hessian has the same eigenvalues as that of the valley, $0.2$ and
   $4$.
    1. Write the gradient of the rotated valley as a function like
       `d2l.valley_grad`, and pass it as `f_grad` to `adagrad_2d`,
       `rmsprop_2d`, and `d2l.gd_2d`. On both valleys, run AdaGrad at
       $\eta = 0.4$ and $\eta = 2$, RMSProp at $\eta = 0.4$, and gradient
       descent at $\eta = 0.4$ with `d2l.train_2d` and `steps=2000`. For
       each run, record the first step at which
       $\|\mathbf{x}\| \leq 10^{-3}$, if there is one, and
       $\|\mathbf{x}\|$ at step 2,000.
    1. How much of the advantage of the two adaptive methods over gradient
       descent survives the rotation? Explain the answer from the Hessian
       of the rotated valley.
    1. Repeat the comparison for `adam_2d` at $\eta = 0.4$. At a condition
       number of 20 the rotation costs Adam almost nothing. Now raise the
       larger curvature from $4$ to $400$ in both valleys (their
       coefficients $2$ and $1$ become $200$ and $100$) and run 5,000
       steps. Which run does not change at all, and how many times more
       steps does the other one need?
   :exlabel:`ex_adam-rotation`

1. **Bias correction by initialization.** Instead of the zero
   initialization and the corrections in :eqref:`eq_adam-moments`,
   initialize $\mathbf{m}_1 = \mathbf{g}_1$ and
   $\mathbf{v}_1 = \mathbf{g}_1^2$ and use the uncorrected averages from
   then on.
    1. Show that the two schemes coincide at $t = 1$ and derive the weights
       that each assigns to $\mathbf{g}_1, \ldots, \mathbf{g}_t$ for
       $t > 1$.
    1. What is lost relative to the corrections of
       :eqref:`eq_adam-moments`, and for which range of $t$ does the
       difference matter?

1. [code] **A hop that does not end.** ● Below, a coordinate has the
   gradient $\lambda x$, with curvature $\lambda > 0$.
    1. Suppose that under Adam the coordinate alternates,
       $x_t = (-1)^t a$. Show that its momentum average settles at
       $(1 - \beta_1)/(1 + \beta_1)$ times the current gradient and its
       scale at $\lambda a$, so that
       $a = \frac{1 - \beta_1}{1 + \beta_1} \cdot \frac{\eta}{2}$,
       whatever the curvature: about $0.011$ at $\beta_1 = 0.9$ and
       $\eta = 0.4$. Print the last 20 values of $x_1$ in the run with
       $\beta_2 = 0.9$. Is its hop such an alternation, and how does its
       size compare?
    1. Run `adam_2d` with the defaults on the valley for 20,000 steps, and
       print the largest $|x_1|$ and the largest $|x_2|$ over the 50 steps
       before steps 200, 2,000, 5,000, 10,000, and 20,000. The iterate
       comes within $10^{-40}$ of the minimum and then leaves again. Which
       coordinate leaves first?
    1. Explain the departure. Near the minimum the gradients vanish, so
       $\hat{v}_t$ shrinks by the factor $\beta_2$ per step and the rate
       $\eta/\sqrt{\hat{v}_t}$ of a coordinate grows. With the scale held
       fixed, Adam on the coordinate is a momentum method with learning
       rate $(1 - \beta_1)\, \eta / \sqrt{\hat{v}_t}$, and by the ceiling
       of :numref:`sec_momentum` it is stable as long as
       $(1 - \beta_1)\, \eta \lambda / \sqrt{\hat{v}_t} < 2 (1 + \beta_1)$.
       Estimate from the size of the early gradients the step at which
       $x_2$ crosses this limit, and compare with your run.
    1. AMSGrad (:numref:`subsec_adam-variance-estimate`) divides by the
       running maximum of $v_t$. Add this maximum to the state of
       `adam_2d` and repeat the run of 20,000 steps. Does the iterate
       leave the minimum now? Explain with the limit of the previous part.
   :exlabel:`ex_adam-constant-rate`

1. **Rare gradients.** ● A coordinate that seldom receives a gradient can
   move by far more than $\eta$ in a step of Adam. Ignore $\epsilon$ and
   the bias correction throughout.
    1. A coordinate has received zero gradients for so long that $v_t$ has
       decayed to almost nothing, and then receives one gradient $g$. Show
       that its next step has the length $\eta/\sqrt{1 - \beta_2}$, about
       $32\,\eta$ at the default $\beta_2$, if $\beta_1 = 0$, and a tenth
       of that if $\beta_1 = 0.9$. Show that with $\beta_1 = 0.9$ the
       coordinate goes on moving while its momentum decays, and travels
       about as far in total as with $\beta_1 = 0$.
    1. Now let the gradient $g$ arrive once every $n$ steps, with zeros in
       between, for $n$ well above $1/(1 - \beta_1)$ and well below
       $1/(1 - \beta_2)$. Show that $v_t \approx g^2/n$ and that every
       arrival moves the coordinate by about $\sqrt{n}\,\eta$ in total.
       How far does SGD with momentum move it per arrival, and how does
       that depend on $n$?
    1. Which parameters of `TinyLM` receive their gradients in this way?
       Is the result of the previous part what AdaGrad was designed to do
       for rare features?

1. [extended] **Frequent and rare tokens.** Following
   :citet:`Kunstner.Yadav.Milligan.ea.2024`, split the 256 byte values of
   `data` into a frequent and a rare group by their share `freq` of the
   training text, for instance at 0.1%. Under SGD with momentum and under
   Adam, each at its best learning rate in `sgd_lm` and `adam_lm`, record
   every 100 steps the loss of `TinyLM` on the targets of each group,
   measured on a fixed set of validation windows.
    1. Which optimizer makes progress on the rare group, and does the gap
       of :numref:`subsec_adam-where-wins` come mainly from the rare group,
       from the frequent group, or from both?
    1. Add a third optimizer, sign descent with momentum, which steps by
       $\eta\, \mathrm{sign}(\mathbf{m}_t)$, and tune it on the grid of
       Adam. On each group, does it behave like Adam or like SGD?
    1. Repeat the measurement with subword tokens, among which rare tokens
       make up a larger share of the text. Train a `d2l.BPETokenizer`
       of :numref:`sec_text-sequence` with 1,024 tokens, the GPT-2
       pattern, and no special tokens on the first million bytes of the
       training text, encode the first ten million bytes and the
       validation split with it, store the results in `data.train` and
       `data.val`, and train `TinyLM` with a vocabulary of 1,024. Compare
       the two groups.
   :exlabel:`ex_adam-frequent-and-rare-tokens`

1. [code] **Per-group learning rates for SGD.** ● The three kinds of
   parameters of `TinyLM` in :numref:`subsec_tinylm` share one learning
   rate under SGD with momentum. In the table of this section the
   embeddings are the rows `token_emb` and `pos_emb`, the matrices are
   `qkv`, `proj`, `mlp`, and `head`, and the vectors are `LayerNorm` and
   `biases`.
    1. Give each of the three groups its own learning rate. Starting from
       the best rate in `sgd_lm`, scale the rate of one group by each
       factor in $\{1/10, 1/3, 3, 10\}$ while holding the other two fixed,
       and keep the best factor for each group. Then train once with the
       three best factors combined. Do the factors agree with the gradient
       scales that the table printed?
    1. How much of the gap between the best runs of `sgd_lm` and `adam_lm`
       does per-group SGD close? What does the result say about whether
       the advantage of Adam on this model comes from scale differences
       between the three groups or from differences among the coordinates
       within each group?
   :exlabel:`ex_adam-per-group-learning-rates`

1. [code] **Two repairs of the second-moment estimate.** Consider the
   sequence of losses $f_t(x) = 4x$ for $t \bmod 3 = 1$ and $f_t(x) = -x$
   otherwise, on the interval $x \in [-1, 1]$
   :cite:`Reddi.Kale.Kumar.2019`. Its gradients repeat the pattern
   $+4, -1, -1$.
    1. Which point of the interval has the lowest total loss over a period
       of three steps? Run Adam on the sequence with $\beta_1 = 0$,
       $\beta_2 = 1/17$, no bias correction, the learning rate
       $\eta_t = 0.5/\sqrt{t}$, and $x$ clipped back to $[-1, 1]$ after
       every step, for 15,000 steps from $x = 0$. Where does $x$ end?
    1. Once $v_t$ repeats with period three, compute its three values and
       from them the three displacements $-\eta_t\, g_t / \sqrt{v_t}$ in
       units of $\eta_t$. In which direction does one period move $x$?
    1. AMSGrad divides by the running maximum $\max(v_1, \ldots, v_t)$
       instead of $v_t$. Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` keeps the
       division and changes the estimate to
       $$v_t = v_{t-1} - (1 - \beta_2)\, g_t^2\, \mathrm{sign}(v_{t-1} - g_t^2),$$
       which moves $v_t$ by at most $(1 - \beta_2)\, g_t^2$ per step. Add
       both to your code and report where each ends. Explain the three
       outcomes from how far $v_t$ falls during the two gradients $-1$ of
       a period.
   :exlabel:`ex_adam-two-repairs-variance-estimate`

1. [code] **The $\epsilon$ ceiling.** Sweep
   $\epsilon \in \{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1\}$ for Adam on
   `TinyLM` at its best learning rate in `adam_lm`.
    1. From the gradient scales that the table of this section printed,
       predict the smallest $\epsilon$ of the sweep that changes the
       result. Then run the sweep and explain the trend at its upper end
       with the ceiling $\eta/\epsilon$ on the learning rate of each
       coordinate (:numref:`subsec_adam-variance-estimate`).
    1. At $\epsilon = 1$ the run resembles SGD with momentum at the
       learning rate $(1 - \beta_1)\eta/\epsilon$. Which $\eta$ would make
       it match the best run of `sgd_lm`? Try it.


:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1078)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1079)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.6]{.kicker}

Adam<br>
**per-coordinate learning rates · bias correction · the learning rate as a step length · gradient scales in a network · Adam against SGD**
:::
:::

::: {.slide title="One Learning Rate Leaves x₁ at −0.94 After 20 Steps"}
[The problem]{.kicker}

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$: curvatures $0.2$ and
$4$, start $(-5, -2)$.

- Gradient descent is stable only for $\eta < 2/4 = 0.5$.
- At $\eta = 0.4$, $x_1$ shrinks by $1 - 0.4 \cdot 0.2 = 0.92$ per step.
- After 20 steps: $x_1 = -5 \cdot 0.92^{20} \approx -0.94$.
- A rate of $1/\textrm{curvature}$ per coordinate would finish in one
  step.

::: {.d2l-note .warn}
A network does not tell us its curvatures. Can each coordinate set its own
step from the size of its own gradients?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute AdaGrad, RMSProp, and Adam steps on the valley by hand;
- explain why the first step of Adam is $\eta$ in every coordinate;
- show that Adam takes the same steps on $f$ and on $1000 f$, and say
  what $\eta$ then measures;
- read a learning-rate sweep for range, speed, and failure mode;
- say why Adam at a constant rate does not settle.

**Running example:** the valley, then `TinyLM`, a transformer with 0.47
million parameters that predicts the next byte of Wikipedia text.
**The idea:** one learning rate per coordinate, set from the size of its
own gradients.
:::

::: {.slide title="Adam Replaces the Single Rate by One Rate per Coordinate"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · **Adam** · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** the valley and its ceiling $\eta < 2/\textrm{curvature}$;
  momentum $\beta$ and its window of $1/(1-\beta)$ steps; `TinyLM`.
- **Skips:** weight decay (next deck), schedules, convergence proofs.
:::

::: {.slide title="AdaGrad Divides Each Step by Its Coordinate's Gradient History"}
[Per-coordinate learning rates]{.kicker}

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

($\mathbf{g}_t$: gradient at $\mathbf{x}_t$; elementwise;
$\mathbf{s}_0 = \mathbf{0}$; small $\epsilon > 0$)

- **Rare features:** a coordinate that is rarely touched keeps a large
  step for when its feature appears.
- **Ill-conditioning:** a gradient is curvature times distance, so the
  curvature cancels from the ratio.

::: {.d2l-note .rule}
Each coordinate gets its own clock: its step decays with its own activity,
not with $t$.
:::
:::

::: {.slide title="Under AdaGrad the Learning Rate Is a Travel Budget of 7.6η in 20 Steps"}
[Per-coordinate learning rates]{.kicker}

::: {.cols}
::: {.col}
$\eta = 0.4$: budget $3.0 < 5$

@!adam-per-coordinate-learning-rates-1
:::

::: {.col}
$\eta = 2$: budget $15$ (gradient descent diverges)

@!adam-per-coordinate-learning-rates-2
:::
:::

- Step 1 moves every coordinate by $\eta$, since
  $\mathbf{s}_1 = \mathbf{g}_1^2$.
- Step $t$ is at most $\eta/\sqrt{t}$:
  $\sum_{t=1}^{20} t^{-1/2} \approx 7.6$.
:::

::: {.slide title="RMSProp Forgets: Its Average Spans 1/(1 − β₂) Steps"}
[Per-coordinate learning rates]{.kicker}

::: {.cols .vc}
::: {.col}
@!adam-rmsprop-forgetting-on-purpose-1
:::

::: {.col}
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t^2$$

replaces $\mathbf{s}_t$ in AdaGrad.

- $\beta_2 = 0.9$: a window of 10 steps.
- No decay by construction: near the minimum after 20 steps.
- Step 1 too long: $\mathbf{v}_1 = 0.1\,\mathbf{g}_1^2$, a move of
  $\eta/\sqrt{0.1} \approx 3.2\,\eta$.
:::
:::
:::

::: {.slide title="After 200 Steps RMSProp Hops Between ±η/2 and AdaGrad Still Converges"}
[Per-coordinate learning rates]{.kicker}

Both at $\eta = 0.4$:

@!adam-rmsprop-1

- At $x = \pm a$ the gradient has a steady size, so $v_t = g_t^2$ and the
  step is $\eta\,\mathrm{sign}(g_t)$.
- A jump of $\eta$ takes $-a$ to $a$ for $a = \eta/2 = 0.2$, at any
  curvature.

::: {.d2l-note .rule}
A normalized step does not shrink with the gradient.
:::
:::

::: {.slide title="Check: Where Does RMSProp End at η = 0.1?"}
[Check]{.kicker}

RMSProp on the valley, $\beta_2 = 0.9$, now at $\eta = 0.1$. Where are
$x_1$ and $x_2$ after 2,000 steps?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Both hop between $-0.05$ and $0.05$: the band is
$\pm\eta/2$, and the curvatures $0.2$ and $4$ do not enter. To come
closer, $\eta$ has to fall during the run.
:::
:::
:::

::: {.slide title="Adam Averages the Gradient and Its Square, Then Corrects Both"}
[Adam]{.kicker}

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,
\qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

- $\mathbf{m}_t$: momentum, averaged over about $1/(1-\beta_1) = 10$
  steps.
- $\mathbf{v}_t$: second-moment estimate; $\sqrt{\hat{\mathbf{v}}_t}$ is
  the *scale* of a coordinate, averaged over about 1,000 steps.
- The hats are the *bias correction*. Defaults: $\beta_1 = 0.9$,
  $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
:::

::: {.slide title="Bias Correction Makes the First Step of Adam η in Every Coordinate"}
[Worked example]{.kicker}

Step 1 from $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$:

- $\mathbf{m}_1 = 0.1\,\mathbf{g}_1$, $\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2$.
- Uncorrected: $\mathbf{m}_1/\sqrt{\mathbf{v}_1} = (0.1/0.0316)\,
  \mathrm{sign}(\mathbf{g}_1) \approx 3.2\,\mathrm{sign}(\mathbf{g}_1)$.
- Corrected: $\hat{\mathbf{m}}_1 = \mathbf{g}_1$,
  $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the step is
  $\eta\,\mathrm{sign}(\mathbf{g}_1)$.
- After 10 steps $\mathbf{v}_t$ still holds only
  $1 - 0.999^{10} \approx 1\%$ of its stationary value.

::: {.d2l-note .rule}
The weights of $\mathbf{v}_t$ sum to $1 - \beta_2^t$; dividing by that sum
removes the deficit at every $t$.
:::
:::

::: {.slide title="On the Valley Adam Marches 0.4 per Step, Overshoots, and Spirals In"}
[The learning rate as a step length]{.kicker}

::: {.cols .vc}
::: {.col}
@!adam-adam-on-the-valley-1
:::

::: {.col .narrow}
$\eta = 0.4$; the plot is the default run.

- Start: $45^\circ$, $0.4$ per step in both coordinates.
- $\beta_2 = 0.999$ remembers the first gradients: within $10^{-4}$ at
  step 200.
- $\beta_2 = 0.9$ forgets: a hop of about $0.02$, for good.
- A longer window postpones the hop; it does not prevent it.
:::
:::
:::

::: {.slide title="Multiply the Loss by 1,000 and Adam Takes the Same Steps"}
[The learning rate as a step length]{.kicker}

$$f \to c f:\qquad \mathbf{g}_t \to c\,\mathbf{g}_t,\quad
\hat{\mathbf{m}}_t \to c\,\hat{\mathbf{m}}_t,\quad
\sqrt{\hat{\mathbf{v}}_t} \to c\,\sqrt{\hat{\mathbf{v}}_t},\qquad
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}} \ \textrm{unchanged}$$

(for $\epsilon$ negligible beside $\sqrt{\hat{\mathbf{v}}_t}$)

- True per coordinate: on the valley Adam takes the steps it takes on
  the round bowl $\frac{1}{2}(x_1^2 + x_2^2)$.
- The condition number 20 drops out, for a valley along the axes.

::: {.d2l-note .rule}
Under SGD $\eta$ multiplies a gradient. Under Adam $\eta$ is a step
length: about $\eta$ per coordinate and step.
:::
:::

::: {.slide title="Check: Which Optimizer Moves This Weight Farther?"}
[Check]{.kicker}

A weight whose gradient stays at $10^{-3}$. One step of SGD at
$\eta = 0.5$, or one step of Adam at $\eta = 0.003$: which is longer? And
if the gradient stays at $10$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** SGD moves $0.5 \cdot 10^{-3} = 5 \cdot 10^{-4}$, Adam about
$0.003$, six times as far. At a gradient of $10$, SGD moves $5$ and Adam
still $0.003$. The two rates have different units.
:::
:::
:::

::: {.slide title="Adam from Scratch Matches the Library to Rounding Error"}
[Implementation]{.kicker}

State per parameter: $\mathbf{m}$, $\mathbf{v}$, and a step count $t$.
Two copies of a linear model, same data, 100 steps: our Adam against the
framework's.

@!adam-adam-both-moments-debiased-2

- Every weight moves by $\eta$ on step 1, as the bias correction
  promises.
:::

::: {.slide title="TinyLM's Gradients Differ up to 40-Fold in Size, Adam's Steps 2-Fold"}
[Adam on a language model]{.kicker}

RMS gradient per kind of parameter (SGD's step per unit of $\eta$) and RMS
of Adam's step 501 over $\eta$. `qkv`, `proj`: attention; `mlp`: MLP;
`head`: output layer.

@!adam-gradient-scales-in-a-network-2

::: {.d2l-note}
A fifth of $\eta$, not $\eta$: minibatch gradients disagree, and the scale
remembers the larger early gradients. Pure noise gives
$\sqrt{(1-\beta_1)/(1+\beta_1)}\,\eta \approx 0.23\,\eta$.
:::
:::

::: {.slide title="Adam Trains over a Tenfold Range of Rates, SGD at One Rate"}
[Adam on a language model]{.kicker}

::: {.cols .vc}
::: {.col}
@!adam-the-race-on-the-language-model-2
:::

::: {.col}
`TinyLM`, 2,000 steps at a constant rate, three seeds, SGD with momentum
$0.9$; `inf`: diverged. *Usable*: mean final loss within $0.15$ nats of
the best rate's.

- Usable: Adam from $0.001$ to $0.01$; SGD at $0.3$ alone, next to
  divergence at $0.5$.
- Best rates 100-fold apart: Adam at $\eta$ steps like SGD with momentum
  at $0.1\,\eta/\textrm{scale}$.
- Adam at $0.03$ stalls near $2.8$ and stays finite.
:::
:::
:::

::: {.slide title="Adam Reaches a Loss of 1.8 in Half to Seven Tenths of SGD's Steps"}
[Adam on a language model]{.kicker}

Three runs at each best rate; the plot is the first seed.

@!adam-the-race-on-the-language-model-3

- Final loss: Adam lower in every run. The curves part in the first few
  hundred steps.
- Largest one-step rise: below $0.2$ nats for Adam; several tenths or
  more in SGD's worst run.
:::

::: {.slide title="Check: Why Does Adam Stall Where SGD Diverges?"}
[Check]{.kicker}

At ten times its best rate Adam ends at a loss of about $2.8$. SGD ends
at `inf` less than a factor of two above its best rate. Why?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Under SGD a large gradient makes a large step, which makes a
larger gradient. The step of Adam stays about $\eta$ whatever the
gradient does: too coarse at $0.03$, but bounded.
:::
:::
:::

::: {.slide title="With β₁ = β₂ = 0, Adam Is Sign Descent"}
[Sources of the advantage]{.kicker}

$$\beta_1 = \beta_2 = 0,\ \epsilon = 0:\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\frac{\mathbf{g}_t}{|\mathbf{g}_t|}
= \mathbf{x}_t - \eta\,\mathrm{sign}(\mathbf{g}_t)$$

- **Sign descent:** every coordinate moves by $\eta$ against the sign of
  its gradient, whatever the size.
- Valley: $\mathbf{g}_1 = (-1, -8)$ gives the step $(0.4, 0.4)$, the
  first step of Adam.
- On large transformers Adam leads SGD by far, and sign descent with
  momentum shares the lead, with full-batch gradients too.

::: {.d2l-note .rule}
Adam gains by equalizing the steps of coordinates whose gradients differ
in size.
:::

[Kunstner et al. 2023; Zhang et al. 2024]{.small}
:::

::: {.slide title="TinyLM Is a Mild Case: Two Blocks, and Its Rare Bytes Hardly Count"}
[Sources of the advantage]{.kicker}

Two sources of unequal gradients grow with a language model:

- blocks (embeddings, attention, MLPs) that differ in curvature far more
  than those of a CNN;
- rare tokens: among word pieces most are rare, yet together much of the
  text. Gradient descent stalls on their loss, Adam does not.

`TinyLM` predicts bytes:

@!adam-sources-of-the-gap-1

[Zhang et al. 2024; Kunstner et al. 2024]{.small}
:::

::: {.slide title="ε Caps the Learning Rate of Every Coordinate at η/ε"}
[Limits]{.kicker}

- $\epsilon$ is a floor on the denominator: the rate
  $\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ of coordinate $i$ never
  exceeds $\eta/\epsilon$.
- Large $\epsilon$: $\mathbf{x}_{t+1} \approx \mathbf{x}_t -
  (\eta/\epsilon)\,\hat{\mathbf{m}}_t$, SGD with momentum $\beta_1$ at
  rate $(1-\beta_1)\eta/\epsilon$.
- On `TinyLM` the scales, about $10^{-4}$ and above, are far larger than
  $\epsilon = 10^{-8}$: the floor is idle.

::: {.d2l-note}
$\epsilon$ interpolates between Adam and SGD with momentum: a tuning
knob, not a numerical detail.
:::
:::

::: {.slide title="Gradients +4, −1, −1 Should Move x Down; Adam Moves It Up"}
[Limits]{.kicker}

Losses $4x, -x, -x$, repeated, on $[-1, 1]$: the gradients sum to $+2$,
so the best point is $x = -1$. Adam with $\beta_1 = 0$,
$\beta_2 = 1/17$:

| gradient $g_t$ | $+4$ | $-1$ | $-1$ |
|---|---:|---:|---:|
| $\sqrt{v_t}$ | 3.89 | 1.35 | 1.02 |
| step $-\eta\, g_t/\sqrt{v_t}$ | $-1.03\,\eta$ | $+0.74\,\eta$ | $+0.98\,\eta$ |

Net $+0.69\,\eta$ per period: Adam ends at $x = +1$.

::: {.d2l-note .warn}
Normalization turns a vote weighted by gradient size into a head count.
AMSGrad divides by the running maximum of $v_t$.
:::

[Reddi et al. 2019]{.small}
:::

::: {.slide title="Check: How Far Does a TinyLM Weight Move per Step After 500 Steps?"}
[Check]{.kicker}

Adam at a constant $\eta = 0.003$. How long is the step of a typical
weight after 500 steps, and will it shrink at this rate?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Between $0.1\,\eta$ and $0.3\,\eta$ in root mean square,
about $6 \cdot 10^{-4}$. A normalized step does not shrink with the
gradient, so only a smaller $\eta$ shortens it.
:::
:::
:::

::: {.slide title="Under Adam the Learning Rate Is a Step Length, and It Is Ours to Choose"}
[Wrap-up]{.kicker}

- AdaGrad: a travel budget of $7.6\,\eta$ in 20 steps. RMSProp forgets,
  and ends hopping at $\pm\eta/2 = \pm 0.2$.
- Adam adds momentum and bias correction: step 1 is $\eta$ in every
  coordinate.
- Same steps on $f$ and on $1000 f$: $\eta$ is a step length, and the
  condition number 20 of the valley drops out.
- `TinyLM`: gradients differ 15- to 40-fold between kinds of parameters,
  the steps of Adam 2-fold.
- Against SGD: a tenfold range of usable rates, a stall in place of
  divergence, loss $1.8$ in half to seven tenths of the steps.

**Next:** weight decay under a per-coordinate method, in AdamW.
:::
