# Adam
:label:`sec_adam`

The modern training recipe of :numref:`sec_training_recipes` replaced SGD
with momentum by AdamW, and we used it there as one ingredient among many.
Adam differs from the optimizers so far in one respect: it gives every
coordinate of the parameter vector its own learning rate. This section asks
what such per-coordinate rates achieve, how Adam sets them without knowing
the curvature, and why weight decay must then bypass them.

The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of :numref:`sec_gd` shows
what a single rate costs. Gradient descent uses one learning rate for every
coordinate, and the stiffest direction decides how large that rate may be.
Here the curvatures are $0.2$ and $4$, so a step is stable only for $\eta <
2/4 = 0.5$. At $\eta = 0.4$ the flat coordinate shrinks by a factor of only
$1 - 0.4 \cdot 0.2 = 0.92$ per step, and twenty steps take $x_1$ from $-5$
to about $-0.94$. With a separate rate for each coordinate, the inverse of
that coordinate's curvature, a single step would reach the minimum. Momentum
averages gradients over time and so eases the problem, but it keeps the
single rate.

The curvatures of a neural network are unknown, and they change during
training. The methods of this section therefore estimate a scale for each
coordinate from the history of that coordinate's own gradients. AdaGrad
accumulates the squared gradients, RMSProp lets the accumulation forget, and
Adam :cite:`Kingma.Ba.2014` adds momentum and corrects both averages for
their start at zero. Adam is the standard optimizer for transformers. To see
where its advantage comes from, we build two small models, a transformer
language model and a convolutional network, and compare Adam with tuned SGD
with momentum on both. The section ends with AdamW, the form in which Adam
is used in practice. AdamW applies weight decay outside the per-coordinate
scaling. It also stores two extra numbers for every parameter, and we count
that memory at the end.

<!-- WRITER: after the split this section no longer builds the two models (they are in :numref:`sec_test_models`), no longer trains the CNN, and no longer ends with AdamW (now :numref:`sec_adamw`); the last four sentences of this paragraph and "why weight decay must then bypass them" in the first paragraph are false. -->

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
the word *learning* far more often than the word *preconditioning*. The
same imbalance appears in advertising and recommendation, where most items
interest only a few users. Parameters tied to rare features receive
informative gradients only when those features occur. Under the decaying
learning rate that stochastic convergence demands (:numref:`sec_sgd`), the
parameters of frequent features approach their optima quickly while the
rare ones are still waiting for evidence, and by the time it arrives the
shared step size has decayed too far to use it. One clock cannot serve both
populations.

AdaGrad :cite:`Duchi.Hazan.Singer.2011` gives every coordinate its own
clock. Instead of decaying the step with the iteration count $t$, it decays
each coordinate's step with that coordinate's own activity, measured by the
running sum of its squared gradients:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t,
$$
:eqlabel:`eq_adagrad`

with all operations elementwise, $\mathbf{s}_0 = \mathbf{0}$, and a small
$\epsilon > 0$ guarding the division. A coordinate whose gradients have been
large moves cautiously from now on. A coordinate that has barely been
touched keeps a large step and can respond as soon as its rare feature
appears.

The same rule addresses the valley. The ideal fix there divides the step in
each direction by the curvature of that direction. A transformation that
rescales the directions of a problem in this way is called a
*preconditioner*. Curvature is too expensive to compute at the scale of deep
learning. When the coordinate axes roughly align with the directions of
curvature, however, the size of a coordinate's past gradients is a usable
stand-in for its curvature. Then :eqref:`eq_adagrad` is a diagonal
preconditioner estimated from gradients alone.
:numref:`subsec_mdl-per-coordinate` makes both readings precise, through a
change of metric and through a regret bound.

We run AdaGrad on the valley from the starting point $(-5, -2)$ at the
learning rate $\eta = 0.4$, close to the stability limit of gradient
descent. The
helper `d2l.train_2d` of :numref:`sec_gd` carries the state $(s_1, s_2)$
from step to step and passes extra keyword arguments, such as `eta`, to the
update rule; `d2l.f_valley` and `d2l.valley_grad` are the valley and its
gradient.

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

The path is smooth: the steep direction $x_2$ is scaled down automatically,
so it does not oscillate. But the accumulation also has a drawback. On the
first step $\mathbf{s}_1 = \mathbf{g}_1^2$, so every coordinate moves by
exactly $\eta$, whatever its gradient. After that, a coordinate whose
gradient stays roughly the same size moves by about $\eta/\sqrt{t}$ at step
$t$, and twenty such steps add up to at most $\eta \sum_{t=1}^{20}
t^{-1/2} \approx 7.6\,\eta = 3.0$. That is not enough to cover the
distance of $5$ along $x_1$, and the iterate stalls far from the minimum.
Because the scaling adapts, we can compensate with a learning rate at which
gradient descent diverges, $\eta = 2$:

```{.python .input #adam-per-coordinate-learning-rates-2}
%%tab pytorch, jax
d2l.show_trace_2d(d2l.f_valley, d2l.train_2d(adagrad_2d, d2l.valley_grad,
                                              eta=2))
```

### RMSProp

AdaGrad's sum never forgets. Under persistent gradient noise
$\mathbf{s}_t$ grows roughly linearly in $t$, so the effective step decays
like $t^{-1/2}$ whether or not the optimization is anywhere near done. This
decay suits the convex problems that AdaGrad was designed for, but it can
stop a nonconvex optimization that still needs to move late in training.

RMSProp :cite:`Tieleman.Hinton.2012` keeps the per-coordinate scaling and
replaces the sum with an exponential moving average, the same leaky
averaging that momentum applies to the gradient itself:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_rmsprop`

The average $\mathbf{v}_t$ estimates the *current* squared-gradient scale
rather than the lifetime total, over a window of about $1/(1-\beta_2)$
steps, ten at the customary $\beta_2 = 0.9$. The step size no longer
decays by construction, and the learning rate $\eta$ becomes a setting we
control separately, to be changed over time by a schedule
(:numref:`sec_scheduler`). On the valley at $\eta = 0.4$, RMSProp gets
close to the minimum where AdaGrad stalled:

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

<!-- WRITER: new cell below (ADAM-3). Measured, both tabs identical (deterministic): at step 200 AdaGrad (eta = 0.4) is at x1 = -0.047, still moving toward 0 (last four x1: -0.050, -0.049, -0.048, -0.047); RMSProp is at (0.201, 0.200) and its last four x1 alternate -0.199, 0.200, -0.201, 0.201: it is not converging but jumping across the minimum by eta = 0.4, a band of eta/2 = 0.2 around it, in both coordinates. RMSProp's step after step 20 is still about eta: its ten-step average follows the gradient down, so g/sqrt(v) stays near 1. "Gets close to the minimum" is true at step 20 (x1 = -0.0106) and false from step 50 on. -->

```{.python .input #adam-rmsprop-1}
%%tab pytorch, jax
def last_x1(path, k=4):
    return ' '.join(f'{x1:.3f}' for x1, _ in path[-k:])

for name, rule in [('AdaGrad', adagrad_2d), ('RMSProp', rmsprop_2d)]:
    print(f'{name}, ', end='')
    path = d2l.train_2d(rule, d2l.valley_grad, steps=200)
    print(f'  x1 over the last four steps: {last_x1(path)}')
```

The first step of this run is larger than intended. Since
$\mathbf{v}_0 = \mathbf{0}$, the average after one step is
$\mathbf{v}_1 = 0.1\, \mathbf{g}_1^2$, a tenth of the squared gradient, and
each coordinate moves by $\eta/\sqrt{0.1} \approx 3.2\,\eta$ instead of
$\eta$. The average starts at zero and needs about ten steps to fill its
window. Adam corrects this startup error exactly.

### Adam

Adam applies the same exponential averaging to the gradient itself, giving
a momentum estimate $\mathbf{m}_t$, keeps RMSProp's second-moment estimate
$\mathbf{v}_t$, and divides each average by the total weight it has
accumulated so far:

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

The update divides the corrected momentum by the square root of the
corrected scale:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_adam-update`

Each coordinate thus moves by about $\eta$ times the ratio of its average
gradient to its typical gradient size. The momentum method of
:numref:`sec_momentum` *adds* each gradient to $\beta$ times its buffer.
The factor $1 - \beta_1$ turns that sum into an *average*, which amounts to
the same method with the learning rate multiplied by $1 - \beta_1$. The
defaults are $\beta_1 = 0.9$ and $\beta_2 = 0.999$: the direction estimate
averages over about ten steps, the scale estimate over about a thousand.
Language-model recipes since GPT-3 :cite:`brown2020language` often use
$\beta_2 = 0.95$, a window of twenty steps. The scale estimate then follows
changes in the gradients more quickly, but with more noise.

The corrections in :eqref:`eq_adam-moments` are exact rather than
heuristic.
Unrolled from $\mathbf{v}_0 = \mathbf{0}$, the average
$\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^{t} \beta_2^{t-s}\, \mathbf{g}_s^2$
is a weighted sum whose weights total $1 - \beta_2^t$, not $1$. If the
squared gradients have a steady mean, $\mathbf{v}_t$ therefore carries the
fraction $1 - \beta_2^t$ of it in expectation, and dividing by that
fraction removes the deficit at every $t$. The deficit is large: at
$\beta_2 = 0.999$, after ten steps $\mathbf{v}_t$ holds about $1\%$ of its
stationary value. Without the corrections, the first step would be
$(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/\sqrt{0.001} \approx 3.2$ times
$\eta$, the same overshoot as RMSProp's. With them, $\hat{\mathbf{m}}_1 =
\mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the first step
moves every coordinate by exactly $\eta$. :numref:`subsec_mdl-per-coordinate`
plots the transient and its cancellation.

Adam also adds $\epsilon$ *outside* the square root, unlike
:eqref:`eq_rmsprop`. Where $\sqrt{\hat{\mathbf{v}}_t}$ is of order one the
placement matters little. It decides the update where
$\sqrt{\hat{\mathbf{v}}_t}$ is small, and we return to it when we look at
the limits of the method. The usual default is $\epsilon = 10^{-8}$.

<!-- WRITER: new cell below (structure 6, "Adam on the valley"), placed here for the writer to move. Measured, both tabs identical: Adam at eta = 0.4 and the defaults reaches (0.996, -0.213) at step 20, overshooting the minimum through its momentum, and is at (0.00002, 0.00004) at step 200, still converging (last four x1 0.000). With RMSProp's ten-step window, beta2 = 0.9, Adam does not settle: its last four x1 are 0.006, -0.015, 0.017, -0.010, a band of about 0.017 that stays the same out to 20,000 steps. Reason: with beta2 = 0.999 the denominator still remembers the large gradients of the first steps, so the steps shrink with the gradient as in AdaGrad; development runs (not printed) show that this memory runs out: at 20,000 steps the defaults also oscillate, in a band of about 1e-3. With noisy gradients the denominator is held up by the noise instead (the network case), so a constant-rate Adam settles only as far as its step eta allows; that is the job of the schedule. -->

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

### Adam from Scratch

The update needs two buffers per parameter and a step counter for the
bias correction. We write it the way the framework writes its own
optimizers, so that it can train any model.

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
by $-\eta$. Keeping the two apart lets us insert weight decay between them
later in the section.
:end_tab:

<!-- WRITER: "later in the section" above: the AdamW chain is now in :numref:`sec_adamw`. -->

Before using our version we check it against the library on a small
least-squares problem: two copies of the same linear model, the same data,
100 steps each. The first step should move every weight by exactly $\eta$.

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

<!-- WRITER: the harness (TinyLM, FashionCNN, train_steps, lm_perplexity, final_loss, smooth, steps_to_reach, evaluate_accuracy, lr_sweep) moved to test-models.md (sec_test_models); the cells below call it from d2l. The first Adam run on TinyLM (perplexity about 4.9) is cut. The cell below loads the byte-level data (FashionMNIST is no longer used here). -->

```{.python .input #adam-adam-from-scratch-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

### Gradient Scales in a Network

<!-- WRITER: NEW subsection (structure 6, "gradient scales in TinyLM"); heading is a placeholder. The cells measure, for eight kinds of parameters of TinyLM, the RMS gradient (over 10 minibatches, the quantity sqrt(v) estimates) at initialization and after 500 Adam steps at eta = 3e-3, and the RMS of Adam's step divided by eta at step 501 (at step 1 that ratio is exactly 1 for every coordinate, by the bias correction, so it is read after 500 steps). Under SGD every step is eta times the gradient, so the first two columns are also SGD's steps per unit of eta. Measured, two development runs and one notebook run per tab (the last line of the table prints the ratios): RMS gradient at initialization from 1.6e-4 to 2.1e-4 (position and token embeddings) to 6.6e-3 to 7.3e-3 (biases), a ratio of 35 to 45; after 500 steps from 8.7e-5 to 1.2e-4 (token embeddings) to 1.2e-3 to 1.7e-3 (output layer and LayerNorm), a ratio of 13 to 18; Adam's step per unit of eta 0.11 to 0.28 for every kind, a ratio of 2.0 to 2.6. Notebook run: 39 / 14 / 2.3 (PyTorch), 40 / 16 / 2.4 (JAX). Tabs agree. The embeddings' small RMS comes partly from rows of bytes that do not occur in a minibatch and get no gradient. Read as the valley's condition number found in a network: one SGD rate moves the kinds by amounts that differ 15- to 40-fold; Adam's moves agree within a factor of about 2.5. -->

```{.python .input #adam-gradient-scales-in-a-network-1}
%%tab pytorch
def gradients(model, num_batches=10):
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

def gradients(model, num_batches=10):
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

grad_init = rms_by_kind(gradients(model))
d2l.train_steps(model, data, optimizer, num_steps=500)
grad_500 = rms_by_kind(gradients(model))
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

## Strengths and Limits of Adam
:label:`subsec_adam-where-wins`

Practitioners train transformers with Adam because SGD, however carefully
tuned, falls behind. On convolutional networks the case for Adam is far
weaker, and SGD with momentum was the standard for a decade of computer
vision. Claims that one optimizer beats another depend strongly on how much
effort went into tuning each :cite:`Schmidt.Schneider.Hennig.2021`, so the
protocol of the comparison matters as much as its result.

Ours is symmetric and small. Both optimizers train the same model from the
same initialization, a random draw shared by every run of the comparison,
for 2,000 steps at a constant learning rate, with no schedule, no clipping,
and no weight decay. Each gets a grid of four learning rates spaced by
factors of about three, and the run with the lowest final training loss
represents its method. Training loss is the right measure here, because
the question is which method optimizes faster, not which one generalizes
better. For the language model the two would agree anyway, since a run
rarely sees the same text twice. Below, "tuned" means the best of these
four rates.

### Comparison on the Language Model

<!-- WRITER: the comparison now runs each optimizer's grid three times (three seeds; within a seed both optimizers and all rates start from the same initialization) and prints every final loss. SGD's grid is 0.1, 0.3, 0.5, 1 (0.03 dropped; 0.5 added between the best rate and divergence, ADAM-12), Adam's 3e-4 to 3e-2 (3e-2 added: Adam's upper edge). The last cell prints, per optimizer, the best rate (lowest mean of the three runs), the range of its three final losses, the rates whose mean is within 0.15 nats of that best (the margin Muon uses, five times the largest spread printed in :numref:`sec_test_models`; that section's comment proposes 0.1, reconcile to one number), and the step at which each of its three best runs first reaches loss 1.8. Measured, six seeds per tab (three probe seeds plus one notebook run) and six more seeds at SGD 0.3 and 0.5 per tab:
- SGD with momentum: 0.1 ends at 1.836 to 1.882 (PyTorch), 1.838 to 1.900 (JAX), 0.19 to 0.23 above 0.3; 0.3 ends at 1.648 to 1.705 (12 runs) and 1.648 to 1.688 (12 runs); 0.5 diverged in 7 of 9 runs (PyTorch) and 9 of 9 (JAX), and the two runs that survived ended at 1.640 and 1.641, below every run at 0.3: SGD's optimum lies at its edge of divergence. 1 diverged in every run. (The old store's 2.247 at 0.3 did not recur in 24 runs.)
- Adam: 3e-4 ends at 1.92 to 1.96, 1e-3 at 1.651 to 1.691, 3e-3 at 1.580 to 1.614, 1e-2 at 1.631 to 1.659, 3e-2 at 2.73 to 2.85, finite in every run: at too large a rate Adam stalls instead of diverging, because its step is about eta whatever the gradient. Usable (within 0.15 nats): 1e-3, 3e-3, 1e-2 in every execution, a factor of ten; SGD: 0.3 only.
- At the best rates every Adam run ended below every SGD run of the same execution, by 0.04 to 0.09 nats per seed (smaller than the margin; do not build a conclusion on the endpoint).
- Steps to loss 1.8 at the best rates: Adam 625 to 775 (PyTorch), 650 to 875 (JAX); SGD 1,100 to 1,275 and 1,075 to 1,250. Adam gets there in about 0.6 of SGD's steps; the ranges do not overlap.
- Largest one-step rise of the loss (new last column; a single minibatch whose loss jumps above the previous one), 8 more seeds per tab: SGD at 0.3 0.18 to 1.72 (PyTorch) and 0.15 to 1.52 (JAX), above 0.5 in 3 and 4 of 8 runs, and the runs with a jump of 1 or more end highest (1.733, 1.710; 1.693); Adam at 3e-3 0.15 to 0.23 and SGD at 0.1 0.14 to 0.26 in every run, the minibatch-to-minibatch level. So SGD at its best rate trains at its edge of stability: single steps throw the loss up by up to 1.7 nats, the start of the divergence that 0.5 completes, and a run that meets one ends higher. The occasional stall (2.041 in one notebook run, 2.247 in the old store) is the far end of the same effect: two of 30 runs at 0.3.
- Notebook run (r3, final code): SGD 0.3: 1.648 to 1.659 (PyTorch), 1.650 to 1.685 (JAX); Adam 3e-3: 1.586 to 1.589 / 1.580 to 1.592; usable: SGD 0.3, Adam 0.001 0.003 0.01 in both; steps to 1.8: SGD 1050 to 1200 / 1100 to 1200, Adam 675 to 700 / 675 to 750; largest one-step rise: SGD 0.14, 1.40, 0.36 / 0.52, 0.15, 0.18, Adam 0.15 to 0.18 in both. Gradient-scale ratios 39 / 16 / 2.2 and 39 / 16 / 1.8.
- Notebook run (r2): SGD 0.3: 1.657 to 1.677 (PyTorch), 1.692 to 2.041 (JAX, one stall; 0.1 then falls within 0.15 of 0.3's mean); Adam 3e-3: 1.578 to 1.587 / 1.576 to 1.594; steps to 1.8: SGD 1150 to 1225 / 1200, 1825, never; Adam 675 to 700 / 725 to 775.
- Notebook run (r1): SGD 0.3: 1.652 to 1.661 / 1.651 to 1.669; Adam 3e-3: 1.583 to 1.589 / 1.582 to 1.614; steps to 1.8: SGD 1150 to 1250 in both tabs, Adam 700 to 750 / 700 to 875. -->

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

SGD's best learning rate sits directly below one at which training
diverges, printed as `inf`. Adam's best rate is interior, with stable
neighbors on both sides. We plot the two tuned runs:

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

Tuned Adam ends about 0.03 to 0.2 nats below tuned SGD, a perplexity 3
to 20% lower, and its curve lies below SGD's from the first hundred steps
to the end. No learning rate in SGD's grid closes the gap: the best one is
the last stable one. Repeated with fresh random draws, both tuned losses
move by a few hundredths of a nat, and SGD, run just below the rate at
which it diverges, can end up to about a tenth of a nat higher. The last
line of the output measures the gap in time rather than in loss. Adam
reaches the loss on which SGD ends 40% to 80% of the way through the run,
so it optimizes this model about 1.3 to 2.5 times as fast.

<!-- WRITER: the CNN comparison (old subsection "Comparison on a CNN": cells adam-the-same-race-on-a-cnn-2/-3/-4 and its three paragraphs) is cut (structure 6). For the record, the 2026-11-12 store: SGD with momentum 0.174 / test 0.901 against Adam 0.126 / 0.918 (PyTorch), 0.154 / 0.904 against 0.116 / 0.921 (JAX). -->

### Sources of the Gap

On our two small models Adam's speedup is modest and of similar size. On
large transformers it is decisive: SGD with momentum falls far behind Adam
there, while on CNNs of comparable scale the two stay close
:cite:`Zhang.Chen.Ding.ea.2024`. The reasons are still being researched,
but three findings point in the same direction.

First, the gap is not about minibatch noise.
:citet:`Kunstner.Chen.Lavington.ea.2023` found that it persists in
*full-batch* training, where there is no noise to average away, and that
Adam's advantage follows that of *sign descent*, the method that moves each
coordinate by a fixed amount in the direction opposite to the sign of its
gradient. The connection is visible in :eqref:`eq_adam-update`: with both
averages switched off ($\beta_1 = \beta_2 = 0$) and $\epsilon = 0$, the
update becomes $\eta\, \mathbf{g}_t / |\mathbf{g}_t| = \eta\,
\mathrm{sign}(\mathbf{g}_t)$. Adam is a smoothed sign method. On these
problems it mostly equalizes the step across coordinates whose gradients
differ by orders of magnitude.

Second, the differences being equalized come from the data. Word
frequencies are heavy-tailed: most words are rare, so the embedding rows
and output weights that serve them see small, infrequent gradients, while
those of frequent words see large, constant ones.
:citet:`Kunstner.Yadav.Milligan.ea.2024` showed that gradient descent makes
almost no progress on the loss of the rare classes while Adam makes
progress on all of them, even in linear models with imbalanced classes.

Third, they come from the architecture. Across the blocks of a
transformer, the embedding, attention, and MLP parameters have curvatures
so different that no single learning rate suits them all, whereas the
blocks of a CNN look much more alike. A single learning rate must be small
enough for the stiffest block, which slows all the others. A
per-coordinate method gives every block its own effective step. The three
parameter groups of `TinyLM` are a coarse version of this division: groups
with different roles and different gradient statistics, all sharing one
$\eta$ under SGD.

How much each finding explains for `TinyLM` is open, and the model is a
mild case of the last two. Its bytes are skewed, but the rare ones carry
little of the loss. Every lowercase letter appears more than twice in an
average minibatch of 4,096 predictions. About a hundred byte values, mostly
pieces of the UTF-8 encodings of non-English characters, appear less than
once in ten minibatches, but together they make up less than 0.1% of the
text, whereas the rare words of a word-level vocabulary together make up a
sizable share of it. And with two blocks of width 128 the model is far
smaller than the transformers of these studies. The exercises test both
explanations on this model.

<!-- WRITER: new cell below (ADAM-11): it prints the byte statistics of the paragraph above, which no cell printed before. Measured on the full training text (the cell counts every tenth byte and prints 99 values, 0.07%, 2.6): 101 byte values appear less than once in ten minibatches of 4,096 targets, together 0.07% of the text; the rarest lowercase letter, q, appears 2.6 times per minibatch; 62 byte values make up 99% of the text. All four statements of the paragraph hold. -->

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

### Limits of the Variance Estimate
:label:`subsec_adam-variance-estimate`

Adam's denominator $\sqrt{\hat{\mathbf{v}}_t}$ is an estimate built from
one noisy stream of gradients, and its errors are not symmetric: an
estimate that is too small produces a step that is too large. The worst
case is a coordinate whose gradient is usually small and occasionally
large. Between the large gradients the average $\mathbf{v}_t$ decays toward
the small ones, so when the next large gradient arrives, its step is far
larger than the scale of that coordinate warrants.
:citet:`Reddi.Kale.Kumar.2019` built a convex online problem from this
pattern on which Adam converges to the *worst* feasible point. The problem
is adversarial and says nothing about ordinary training data, but it shows
that Adam has no general convergence guarantee. Their repair, AMSGrad,
divides by the running maximum of $\hat{\mathbf{v}}_t$, so that the
estimate can never shrink. :numref:`subsec_mdl-adam-counterexample` works
through the construction and the repair. Neither AMSGrad nor later repairs
replaced Adam in practice.

The constant $\epsilon$ limits the damage. It is meant to guard against
division by zero, but it also acts as a floor on the denominator of
:eqref:`eq_adam-update`. The learning rate of a coordinate, the factor
$\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ that multiplies its momentum
$\hat{m}_{t,i}$, therefore never exceeds $\eta/\epsilon$. A small $\epsilon$
trusts the variance estimate even when it is tiny; a large $\epsilon$ does
not. In the limit of large $\epsilon$ the denominator is $\epsilon$ for
every coordinate, and Adam becomes $\mathbf{x}_{t+1} = \mathbf{x}_t -
(\eta/\epsilon)\, \hat{\mathbf{m}}_t$. Since $\mathbf{m}_t$ is $1 - \beta_1$
times the momentum buffer of :numref:`sec_momentum`, this is SGD with
momentum $\beta = \beta_1$ and learning rate $(1 - \beta_1)\eta/\epsilon$,
up to the bias correction of the first steps. Values of $\epsilon$ between
the default $10^{-8}$ and this limit interpolate between the two methods,
which makes $\epsilon$ a tuning knob rather than a numerical detail.

## Summary

A single learning rate must suit the stiffest direction of the loss. Adam
gives each coordinate its own step, scaled by an estimate of that
coordinate's gradient size. The estimate is AdaGrad's running sum of
squared gradients, made to forget by RMSProp's exponential average. Adam
combines it with momentum and corrects both averages exactly for their
start at zero. On a small transformer language model and on a CNN of the
same size, tuned Adam reaches tuned SGD's final training loss about 1.3
to 2.5 times as fast and ends a few hundredths to about two tenths of a nat
lower; on the CNN the test accuracies agree to within a point or two. On
large transformers
the gap is far wider, and the best current explanations point to gradient
scales that differ across coordinates, from heavy-tailed data and from
dissimilar blocks of the architecture. Under Adam an $\ell_2$ penalty is
not weight decay, because its gradient passes through the per-coordinate
scaling. AdamW applies the decay outside it, which gives $\lambda$ the
same meaning for every coordinate. At scale the product $\eta\lambda$
sets the effective learning rate of normalized layers. Adam stores two
extra numbers per parameter, which in mixed precision bring a parameter
to 16 to 20 bytes. What Adam does not remove is the choice of $\eta$ over
time, the subject of :numref:`sec_scheduler`.

<!-- WRITER: Summary predates the split and the new runs: the CNN, "1.3 to 2.5 times as fast", "a few hundredths to about two tenths of a nat", and the AdamW and memory sentences (now :numref:`sec_adamw`) no longer hold here. -->

## Exercises

1. [code] **Rotating the coordinates.** Per-coordinate methods are
   axis-aligned. Rotating the valley `f_valley` by $45^\circ$ gives
   $$f(\mathbf{x}) = 0.05 (x_1 + x_2)^2 + (x_1 - x_2)^2,$$
   whose Hessian has the same eigenvalues as that of `f_valley`, $0.2$
   and $4$.
    1. Write the gradient of the rotated valley as a function like
       `valley_grad`, and pass it as `f_grad` to `adagrad_2d`,
       `rmsprop_2d`, and the `gd_2d` of :numref:`sec_gd`. On both
       valleys, run AdaGrad at $\eta = 0.4$ and $\eta = 2$, RMSProp at
       $\eta = 0.4$, and gradient descent at $\eta = 0.4$ with
       `d2l.train_2d` and `steps=2000`. For each run, record the first
       step at which $\|\mathbf{x}\| \leq 10^{-3}$, or that no step within
       the 2,000 reaches it.
    1. How much of the adaptive methods' advantage over gradient descent
       survives the rotation? Explain the answer from the Hessian of the
       rotated valley.

1. **Bias correction by initialization.** Instead of the zero
   initialization and the corrections in :eqref:`eq_adam-moments`,
   initialize $\mathbf{m}_1 = \mathbf{g}_1$ and
   $\mathbf{v}_1 = \mathbf{g}_1^2$ and use the uncorrected averages from
   then on.
    1. Show that the two schemes coincide at $t = 1$ and derive the weights
       that each assigns to $\mathbf{g}_1, \ldots, \mathbf{g}_t$ for
       $t > 1$.
    1. What is lost relative to the exact correction, and for which range
       of $t$ does the difference matter?

1. [extended] **Frequent and rare tokens.** Following
   :citet:`Kunstner.Yadav.Milligan.ea.2024`, split the 256 byte values of
   `data` into a frequent and a rare group by their share of the training
   text, for instance at 0.1%. Under SGD with momentum and under Adam, each
   at the best learning rate of `sgd_lm` and `adam_lm`, record every 100
   steps the loss of `TinyLM` on the targets of each group, measured on a
   fixed set of validation windows.
    1. Which optimizer makes progress on the rare group, and does the gap
       of :numref:`subsec_adam-where-wins` come mainly from the rare group,
       from the frequent group, or from both?
    1. Repeat the measurement with subword tokens, among which rare
       tokens make up a larger share of the text. Train a
       `d2l.BPETokenizer` with 1,024 tokens, the GPT-2 pattern, and no
       special tokens (:numref:`sec_text-sequence`) on the
       first million bytes of the training text, encode the first ten
       million bytes and the validation split with it, store the results
       in `data.train` and `data.val`, and train `TinyLM` with a
       vocabulary of 1,024. Compare the two splits.

1. [code] **Per-group learning rates for SGD.** ● The three parameter
   groups of `TinyLM` in :numref:`subsec_tinylm`, embeddings, matrices, and
   vectors, share one learning rate under SGD with momentum.
    1. Give each group its own learning rate. Starting from the best rate
       of `sgd_lm`, scale one group's rate by each factor in
       $\{1/10, 1/3, 3, 10\}$ while holding the other two fixed, and keep
       the best factor for each group. Then train once with the three best
       factors combined.
    1. How much of the gap between the best runs of `sgd_lm` and `adam_lm`
       does per-group SGD close? What does the result say about whether
       Adam's advantage on this model comes from scale differences between
       the three groups or from differences among the coordinates within
       each group?

1. [code] **Two repairs of the variance estimate.** Adam's second-moment
   update can be written as
   $$\mathbf{v}_t = \mathbf{v}_{t-1} + (1 - \beta_2)(\mathbf{g}_t^2 - \mathbf{v}_{t-1}),$$
   which shrinks $\mathbf{v}_t$ whenever $\mathbf{g}_t^2 < \mathbf{v}_{t-1}$.
   Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` limits how fast it can shrink
   by replacing the increment with
   $$(1 - \beta_2)\, \mathbf{g}_t^2 \odot \mathrm{sgn}(\mathbf{g}_t^2 - \mathbf{v}_{t-1}).$$
   AMSGrad :cite:`Reddi.Kale.Kumar.2019` carries the running maximum
   $$\hat{\mathbf{v}}_t^{\max} = \max(\hat{\mathbf{v}}_{t-1}^{\max}, \hat{\mathbf{v}}_t)$$
   and uses it in place of $\hat{\mathbf{v}}_t$ in :eqref:`eq_adam-update`.
   (The function `online` below takes the maximum of the uncorrected
   $\mathbf{v}_t$ instead, as the original paper does; implement the
   version written here.)
    1. Implement both by changing the from-scratch Adam of this section,
       and check each against a hand computation of its first three steps
       on a scalar gradient sequence such as $(1, 0.1, 0.1)$.
    1. The function `online` of :numref:`subsec_mdl-adam-counterexample` runs
       Adam and AMSGrad on the losses :eqref:`eq_mdl-opt-reddi`, on which
       Adam converges to the worst point $x = +1$. Add the Yogi update to
       it, run it at $C = 4$, and report where its iterate ends. Explain the
       outcome from how far the $v_t$ of Adam and of Yogi can fall during the
       two $-1$ gradients of each period.
   :exlabel:`ex_adam-two-repairs-variance-estimate`

1. [code] **The $\epsilon$ ceiling.** Sweep
   $\epsilon \in \{10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}, 1\}$ for Adam on
   `TinyLM` at the best learning rate of `adam_lm`. Explain the trend at
   both ends of the sweep with the ceiling $\eta/\epsilon$ on each
   coordinate's learning rate of :numref:`subsec_adam-variance-estimate`,
   and identify the $\epsilon$
   from which the run resembles SGD with momentum, as in `sgd_lm`, at
   learning rate $(1 - \beta_1)\eta/\epsilon$.


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

Adam<br>**per-coordinate learning rates · bias correction · gradient scales in a network · Adam against SGD**
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
Curvatures are unknown in a network. Can each coordinate estimate its own
step from its own gradients?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute AdaGrad, RMSProp, and Adam steps on the valley by hand;
- show that bias correction makes Adam's first step exactly $\eta$;
- read a fair comparison of Adam and SGD on a transformer and a CNN;
- tell an $\ell_2$ penalty from decoupled weight decay under Adam;
- count the bytes of optimizer state per parameter.

**Running example:** the valley above, then `TinyLM`, a 0.47M-parameter
transformer on the bytes of WikiText-103.
:::

::: {.slide title="One Step Size per Coordinate"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| Gradient Descent | step size against curvature |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| Momentum | averaging over time |
| **Adam and AdamW** | **a step size per coordinate, decoupled decay** |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** the valley, momentum $\beta$, weight decay, the byte-level
  language model.
- **Skips:** convergence proofs (appendix), second-order methods.
:::
:::
:::

::: {.slide title="AdaGrad Divides Each Step by Its Coordinate's Gradient History"}
[Per-coordinate learning rates]{.kicker}

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

(elementwise, $\mathbf{s}_0 = \mathbf{0}$)

- **Rare features:** a coordinate that is rarely touched keeps a large
  step for when its feature appears.
- **Ill-conditioning:** large past gradients stand in for large curvature,
  so $\sqrt{\mathbf{s}_t}$ is a diagonal preconditioner.

::: {.d2l-note .rule}
Each coordinate gets its own clock: its step decays with its own activity,
not with $t$.
:::
:::

::: {.slide title="AdaGrad Stalls at η = 0.4 and Converges at η = 2"}
::: {.cols}
::: {.col}
$\eta = 0.4$

@!adam-per-coordinate-learning-rates-1
:::

::: {.col}
$\eta = 2$ (gradient descent diverges)

@!adam-per-coordinate-learning-rates-2
:::
:::

- Step 1 moves every coordinate by exactly $\eta$: $\mathbf{s}_1 = \mathbf{g}_1^2$.
- Then about $\eta/\sqrt{t}$: 20 steps cover at most $7.6\,\eta = 3.0 < 5$.
:::

::: {.slide title="RMSProp Forgets: Its Average Spans 1/(1 − β₂) Steps"}
::: {.cols .vc}
::: {.col .fig}
@!adam-rmsprop-forgetting-on-purpose-1
:::

::: {.col}
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t^2$$

- $\beta_2 = 0.9$: a window of 10 steps.
- Steps no longer shrink by construction.
- Step 1 overshoots: $\mathbf{v}_1 = 0.1\,\mathbf{g}_1^2$, so each
  coordinate moves $\eta/\sqrt{0.1} \approx 3.2\,\eta$.
:::
:::
:::

::: {.slide title="Adam Averages the Gradient and Its Square, Then Corrects Both"}
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,
\qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

- Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$:
  direction over ~10 steps, scale over ~1,000.
- Language-model recipes often use $\beta_2 = 0.95$: a window of 20 steps.
- $\mathbf{m}_t$ is momentum, averaged instead of summed.
:::

::: {.slide title="Bias Correction Makes Adam's First Step Exactly η"}
[Worked example]{.kicker}

Step 1 from $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$:

- $\mathbf{m}_1 = 0.1\,\mathbf{g}_1$, $\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2$.
- Uncorrected: $\mathbf{m}_1/\sqrt{\mathbf{v}_1} = (0.1/0.0316)\,
  \mathrm{sign}(\mathbf{g}_1) \approx 3.2\,\mathrm{sign}(\mathbf{g}_1)$.
- Corrected: $\hat{\mathbf{m}}_1 = \mathbf{g}_1$,
  $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the step is
  $\eta\,\mathrm{sign}(\mathbf{g}_1)$.
- After 10 steps $\mathbf{v}_t$ still holds only $1 - 0.999^{10} \approx 1\%$
  of its stationary value.

::: {.d2l-note .rule}
The weights of $\mathbf{v}_t$ sum to $1 - \beta_2^t$; dividing by that sum
removes the deficit at every $t$.
:::
:::

::: {.slide title="Check: How Large Is the First Uncorrected Step at β₂ = 0.99?"}
[Check]{.kicker}

Adam without bias correction, $\beta_1 = 0.9$, $\beta_2 = 0.99$. How far
does the first step move a coordinate, in units of $\eta$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $(1-\beta_1)/\sqrt{1-\beta_2} = 0.1/0.1 = 1$: exactly $\eta$,
by coincidence. From step 2 on, $\mathbf{m}_t$ fills its window ten times
faster than $\mathbf{v}_t$, and the uncorrected steps grow too large.
:::
:::
:::

::: {.slide title="Adam from Scratch Matches the Library to Rounding Error"}
Two copies of a linear model, same data, 100 steps: our Adam against the
framework's.

@!adam-adam-both-moments-debiased-2

- Every weight moves by $\eta$ on step 1, as the bias correction promises.
- Written as a framework optimizer, the from-scratch version trains any
  model; afterwards we call the library's.
:::

<!-- WRITER: deck to rebuild (worklist 6): chain, valley with Adam (@!adam-adam-on-the-valley-1, RMSProp's band @!adam-rmsprop-1), scale invariance, gradient scales in TinyLM (@!adam-gradient-scales-in-a-network-2), usable range (@!adam-the-race-on-the-language-model-3). The CNN half of the slide below is cut with its cells; its title and bullets still quote the old numbers. -->

::: {.slide title="On TinyLM, Adam Reaches SGD's Final Loss Faster"}
@!adam-the-race-on-the-language-model-3

- SGD trains best at or just below its largest stable rate; Adam's best
  rate lies inside its grid.
:::

::: {.slide title="Check: What Does a 0.1-Nat Gap Do to Perplexity?"}
[Check]{.kicker}

Adam's final loss is 0.1 nats below SGD's. By what factor is Adam's
perplexity lower?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Perplexity is $e^{\textrm{loss}}$, so the ratio is
$e^{-0.1} \approx 0.90$: about 10% lower.
:::
:::
:::

::: {.slide title="With β₁ = β₂ = 0, Adam Is Sign Descent"}
$$\beta_1 = \beta_2 = 0,\ \epsilon = 0:\qquad
\eta\,\frac{\mathbf{g}_t}{|\mathbf{g}_t|} = \eta\,\mathrm{sign}(\mathbf{g}_t)$$

Adam equalizes steps across coordinates whose gradients differ by orders
of magnitude. Three findings on why this helps transformers:

- The gap persists with full-batch gradients, so it is not noise.
- Rare tokens: gradient descent stalls on rare classes; Adam does not.
- Transformer blocks have very different curvatures; CNN blocks look
  alike.

[Kunstner et al. 2023, 2024; Zhang et al. 2024]{.small}
:::

::: {.slide title="ε Caps Every Coordinate's Learning Rate at η/ε"}
[Limits]{.kicker}

- Too small an estimate $\hat{\mathbf{v}}_t$ gives too large a step. A
  rare large gradient after many small ones hits exactly this case.
- Reddi et al. build a convex problem on which Adam converges to the worst
  point; AMSGrad divides by the running maximum of $\hat{\mathbf{v}}_t$.
- $\epsilon$ is a floor on the denominator, so a coordinate's learning
  rate $\eta/(\sqrt{\hat{v}_{t,i}} + \epsilon)$ is at most $\eta/\epsilon$.
- Large $\epsilon$: $\mathbf{x}_{t+1} \approx \mathbf{x}_t -
  (\eta/\epsilon)\hat{\mathbf{m}}_t$, SGD with momentum $\beta_1$ at rate
  $(1-\beta_1)\eta/\epsilon$.
:::

::: {.slide title="Wrap-up"}
- AdaGrad: one clock per coordinate. On the valley it stalls at
  $\eta = 0.4$ and converges at $\eta = 2$.
- RMSProp forgets; Adam adds momentum and bias correction, so step 1 is
  exactly $\eta$.
- Tuned Adam reaches tuned SGD's final loss about 1.3–2.5× faster on
  both models; the gap is about 0.03–0.2 nats on `TinyLM`, about
  0.03–0.1 on `FashionCNN`.
- AdamW decays outside the preconditioner: one $\lambda$ for every
  coordinate. At scale, $\eta\lambda$ sets the effective learning rate.
- Cost: 16–20 bytes per parameter.

**Next:** how to change $\eta$ during training, in Schedules.
:::
