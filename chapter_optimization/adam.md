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

<!-- WRITER: the harness (TinyLM, FashionCNN, train_steps, lm_perplexity, final_loss, smooth, steps_to_reach, evaluate_accuracy, lr_sweep) moved to test-models.md (sec_test_models); the cells below call it from d2l. The first Adam run on TinyLM (perplexity about 4.9) is cut. The cell below loads the two data sets. -->

```{.python .input #adam-adam-from-scratch-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
fashion = d2l.FashionMNIST(batch_size=256)
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

```{.python .input #adam-comparison-on-the-language-model-1}
%%tab pytorch
lm_seed = random.getrandbits(31)
sgd_lm, _ = d2l.lr_sweep(lambda: d2l.TinyLM(len(data.vocab)),
                     lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                                       momentum=0.9),
                     data, lrs=[0.03, 0.1, 0.3, 1.0], seed=lm_seed)
```

```{.python .input #adam-comparison-on-the-language-model-1}
%%tab jax
lm_seed = d2l.get_seed()
sgd_lm, _ = d2l.lr_sweep(lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs),
                     lambda lr: optax.sgd(lr, momentum=0.9), data,
                     lrs=[0.03, 0.1, 0.3, 1.0], seed=lm_seed)
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab pytorch
adam_lm, _ = d2l.lr_sweep(lambda: d2l.TinyLM(len(data.vocab)),
                      lambda model, lr: torch.optim.Adam(model.parameters(),
                                                         lr),
                      data, lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=lm_seed)
```

```{.python .input #adam-the-race-on-the-language-model-2}
%%tab jax
adam_lm, _ = d2l.lr_sweep(lambda rngs: d2l.TinyLM(len(data.vocab), rngs=rngs),
                      optax.adam, data, lrs=[3e-4, 1e-3, 3e-3, 1e-2],
                      seed=lm_seed)
```

SGD's best learning rate sits directly below one at which training
diverges, printed as `inf`. Adam's best rate is interior, with stable
neighbors on both sides. We plot the two tuned runs:

```{.python .input #adam-the-race-on-the-language-model-3}
%%tab pytorch, jax
best_sgd = min(sgd_lm, key=lambda lr: d2l.final_loss(sgd_lm[lr]))
best_adam = min(adam_lm, key=lambda lr: d2l.final_loss(adam_lm[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [d2l.smooth(sgd_lm[best_sgd]), d2l.smooth(adam_lm[best_adam])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd:g}',
                 f'Adam, lr {best_adam:g}'])
loss_sgd = d2l.final_loss(sgd_lm[best_sgd])
loss_adam = d2l.final_loss(adam_lm[best_adam])
print(f'final loss: SGD {loss_sgd:.3f}, Adam {loss_adam:.3f}, '
      f'gap {loss_sgd - loss_adam:.3f} nats')
print(f"Adam reaches SGD's final loss at step "
      f'{d2l.steps_to_reach(adam_lm[best_adam], loss_sgd)} of 2000')
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

### Comparison on a CNN

The same protocol trains `FashionCNN`. Adam's grid stays the same, SGD's
grid moves one step lower, and every run also records its test accuracy.

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab pytorch
def run_cnn(make_optimizer, lrs, seed):
    """lr_sweep of FashionCNN, with the test accuracy at the end."""
    return d2l.lr_sweep(d2l.FashionCNN, make_optimizer, fashion, lrs, seed,
                    evaluate=lambda model: d2l.evaluate_accuracy(
                        model, fashion.val_dataloader()),
                    metric='test accuracy')

seed = random.getrandbits(31)
sgd_cnn, sgd_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.SGD(model.parameters(), lr, momentum=0.9),
    lrs=[0.01, 0.03, 0.1, 0.3], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-2}
%%tab jax
def run_cnn(make_tx, lrs, seed):
    """lr_sweep of FashionCNN, with the test accuracy at the end."""
    return d2l.lr_sweep(d2l.FashionCNN, make_tx, fashion, lrs, seed,
                    evaluate=lambda model: d2l.evaluate_accuracy(
                        model, fashion.val_dataloader()),
                    metric='test accuracy')

seed = d2l.get_seed()
sgd_cnn, sgd_cnn_acc = run_cnn(lambda lr: optax.sgd(lr, momentum=0.9),
                               lrs=[0.01, 0.03, 0.1, 0.3], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab pytorch
adam_cnn, adam_cnn_acc = run_cnn(
    lambda model, lr: torch.optim.Adam(model.parameters(), lr),
    lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-3}
%%tab jax
adam_cnn, adam_cnn_acc = run_cnn(optax.adam,
                                 lrs=[3e-4, 1e-3, 3e-3, 1e-2], seed=seed)
```

```{.python .input #adam-the-same-race-on-a-cnn-4}
%%tab pytorch, jax
best_sgd_cnn = min(sgd_cnn, key=lambda lr: d2l.final_loss(sgd_cnn[lr]))
best_adam_cnn = min(adam_cnn, key=lambda lr: d2l.final_loss(adam_cnn[lr]))
d2l.plot(list(range(0, 2000, 25)),
         [d2l.smooth(sgd_cnn[best_sgd_cnn]),
          d2l.smooth(adam_cnn[best_adam_cnn])],
         'step', 'training loss',
         legend=[f'SGD + momentum, lr {best_sgd_cnn:g}',
                 f'Adam, lr {best_adam_cnn:g}'])
loss_sgd = d2l.final_loss(sgd_cnn[best_sgd_cnn])
loss_adam = d2l.final_loss(adam_cnn[best_adam_cnn])
print(f'final loss: SGD {loss_sgd:.3f}, Adam {loss_adam:.3f}, '
      f'gap {loss_sgd - loss_adam:.3f} nats')
print(f"Adam reaches SGD's final loss at step "
      f'{d2l.steps_to_reach(adam_cnn[best_adam_cnn], loss_sgd)} of 2000')
print(f'test accuracy: SGD {sgd_cnn_acc[best_sgd_cnn][-1]:.3f}, '
      f'Adam {adam_cnn_acc[best_adam_cnn][-1]:.3f}')
```

Two things repeat from the language model. SGD's best rate again sits
close to the edge of stability: at $\eta = 0.3$ training is unreliable.
Often the loss stays at $\log 10 \approx 2.30$, the loss of a uniform
guess over the ten classes, and when the run does train, it ends well
above the run at $\eta = 0.1$. On both models SGD trains best at or just
below the largest rate that trains reliably. And tuned Adam again reaches
a lower training loss.

Both losses are cross-entropies in nats per prediction, so the two gaps
can be compared directly: a few hundredths to about a tenth of a nat on
the CNN, about 0.03 to 0.2 on the language model. A gap of $\Delta$ nats
means that the better model assigns, on average, $e^\Delta$ times as much
probability to each correct answer: a few percent to about 10% more on the
CNN, about 3 to 20% more on the language model.

Measured in time, the question our protocol asks, the two models are
close as well: on both, Adam reaches SGD's final loss about 1.3 to 2.5
times as fast. After the first hundred steps the two CNN curves run
within about a tenth of a nat of each other, and the test accuracies of
the two tuned runs lie within a point or two of each other, between 90%
and 93%. On this problem SGD with momentum is a competitive choice, which
is why it served computer vision for a decade.

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

Adam and AdamW<br>**per-coordinate learning rates · bias correction · Adam against SGD · decoupled weight decay · optimizer memory**
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

::: {.slide title="On Both Models, Adam Reaches SGD's Final Loss 1.3–2.5× Faster"}
::: {.cols}
::: {.col}
`TinyLM`: gap about 0.03–0.2 nats

@!adam-the-race-on-the-language-model-3
:::

::: {.col}
`FashionCNN`: gap about 0.03–0.1 nats

@!adam-the-same-race-on-a-cnn-4
:::
:::

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
