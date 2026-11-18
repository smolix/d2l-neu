# Schedules
:label:`sec_scheduler`

So far a learning rate has been a single number, chosen before a run and
never touched again, and twice a run refused to finish because of it. On
the valley, SGD stopped short of the minimum and wandered in a noise ball
around it. Halving the rate halved the squared radius of the ball, but it
also doubled the number of steps needed to get there (:numref:`sec_sgd`).
Adam did not come to rest either. Its step has a length set by the learning
rate and not by the gradient, so at a constant rate it keeps hopping
(:numref:`sec_adam`). Both times the remedy was plain, and both times we
put it off: take large steps while the minimum is far away, and small ones
once it is near. A microscope has a coarse focusing knob and a fine one for
the same reason, and nobody turns the fine one first.

The recipes that train real models change the rate in this way, and in one
more. The recipe of :numref:`sec_training_recipes` let its rate fall along a
cosine curve, but first it *raised* the rate over the first few epochs, and
almost every large model today is trained with such a warmup followed by a
decay. That the rate should fall at the end we expected. Why should it rise
at the start? We have met a hint: Adam at ten times its best rate stalled
on `TinyLM`, far above the loss that it reaches otherwise. Such a run, we
will see, is lost within its first few steps, and twenty careful steps save
it.

A learning-rate *schedule* gives the learning rate $\eta_t$ for every step
$t$ of training. Of the three decisions that the chapter began with
(:numref:`chap_optimization`), it is the second, the size of a step over a
run. We begin on the valley, where we can compute what a decaying rate has
to achieve. The answer is two numbers: how far the schedule lets the
iterate travel, and how small it makes the last steps. We then train
`TinyLM` under three shapes of decay, look at what warmup does to the first
steps of a run, and end with the schedule of large language models, which
lets us decide late how long to train.

```{.python .input #lr-scheduler-a-testbed-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #lr-scheduler-a-testbed-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #lr-scheduler-schedules-1}
%%tab pytorch
import copy
import random
```

## Decay on the Valley
:label:`subsec_scheduler-valley`

A decaying schedule has two jobs: to carry the iterate to the minimum, and
to shrink the noise around it. On the valley
$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of the earlier sections, with
curvatures $0.2$ and $4$ and gradient noise of unit variance, both effects
can be computed.

### Progress and Noise

Under SGD each coordinate of the valley evolves on its own. Take one
coordinate with curvature $\lambda$ and gradient noise $\xi_t$ of mean zero
and variance $\sigma^2$, as in :numref:`sec_sgd`, and let the learning rate
$\eta_t$ change from step to step:

$$x_{t+1} = (1 - \eta_t \lambda)\, x_t - \eta_t\, \xi_t.$$

The first term pulls the iterate toward the minimum at $0$, and the second
pushes it in a random direction. Since the update is linear in $x_t$, we can
follow the two effects separately. Consider the pull first. The mean of
$x_t$ shrinks by the factor $1 - \eta_s \lambda \approx e^{-\eta_s \lambda}$
at every step $s$, so after $t$ steps it has shrunk by about
$e^{-\lambda B_t}$, where

$$B_t = \sum_{s < t} \eta_s$$

is the sum of all learning rates so far. We call $B_t$ the *budget* of the
schedule. If we draw the rate against the step, the budget is the area
under the curve. It measures how far the schedule lets the iterate travel:
a thousand steps at a rate of $0.01$ move the mean as far as a hundred
steps at $0.1$. Now the push. At a constant rate the noise holds the
iterate in a noise ball of squared radius $\eta\sigma^2 / (2\lambda)$, for
$\eta\lambda \ll 1$ (:numref:`sec_sgd`). If the rate changes slowly, the
ball follows it, where slowly means little change over the
$1/(\eta_t\lambda)$ steps that the iterate needs to forget where it was.
This is the assumption to keep an eye on; one of our schedules will break
it. Together,

$$\mathbb{E}\big[x_t^2\big] \approx e^{-2\lambda B_t}\, x_0^2 + \frac{\eta_t\, \sigma^2}{2\lambda}.$$
:eqlabel:`eq_scheduler-bias-noise`

The first term is what remains of the starting distance, and it shrinks
with the budget. The second is the noise ball of the current rate. On the
flat direction of the valley, $\lambda = 0.2$ and $x_0 = -5$, a budget of
$10$ leaves $e^{-2} \approx 14\%$ of the starting distance and a budget of
$25$ leaves $e^{-5} \approx 0.7\%$. A schedule that converges must
therefore let the budget grow without bound, so that the first term
vanishes, and let the rate go to zero, so that the second does. The
conditions of :citet:`Robbins.Monro.1951` ask for a little more, namely
$\sum_t \eta_t^2 < \infty$ (:numref:`sec_sgd`). The proof for general
objectives needs it. The quadratic does not, as one of the runs below will
show.

Three classical families of schedules are

$$
\begin{aligned}
\eta_t &= \eta_i \quad \textrm{for } t_i \leq t < t_{i+1} && \textrm{piecewise constant}, \\
\eta_t &= \eta_0\, \gamma^t && \textrm{exponential}, \\
\eta_t &= \eta_0\, (1 + t/\tau)^{-\alpha} && \textrm{polynomial},
\end{aligned}
$$
:eqlabel:`eq_scheduler-classical`

with a decay factor $\gamma \in (0, 1)$, a time scale $\tau$, and an
exponent $\alpha > 0$. The first keeps the rate fixed between chosen steps
$t_1 < t_2 < \ldots$, as the older recipe of :numref:`sec_training_recipes`
did when it divided its rate by ten twice. We leave it to Exercise
:exref:`ex_scheduler-step-decay`. Exponential decay multiplies the rate by
$\gamma$ at every step, so its budget is the geometric sum
$\eta_0 / (1 - \gamma)$, which is finite. Polynomial decay has an infinite
budget for $\alpha \leq 1$, and its squares have a finite sum for
$\alpha > 1/2$. The case $\alpha = 1$ is the rate $\eta_t \propto 1/t$ of
:numref:`sec_sgd`.

### Schedules on the Valley

We test the formula with 1,000 independent SGD runs on the valley, all
started at $(-5, -2)$. The function `sgd_valley` returns the mean squared
distance to the minimum after each step. Four schedules run for 3,000
steps: the constant rate $0.1$ that we used in :numref:`sec_sgd`;
exponential decay from $\eta_0 = 0.4$ with $\gamma = 0.98$; and polynomial
decay from $\eta_0 = 0.4$ with $\tau = 25$ and $\alpha = 1/2$ or
$\alpha = 1$. The start at $0.4$ is close to the stability limit
$2/4 = 0.5$ of the steep direction.

Before we run anything, let us ask the two numbers where each schedule
should end. Every budget covers the start: even the smallest,
$0.4/(1 - 0.98) = 20$ for exponential decay, shrinks the squared distance
along $x_1$ by $e^{-8}$, from $25$ to under $0.01$. So the final rates
should decide. The constant rate must stall at its noise ball, $0.25$ along
$x_1$ and $0.016$ along $x_2$, together $0.27$. (The second value is the
exact one of :numref:`sec_sgd`, since $\eta\lambda = 0.4$ is not small.)
Polynomial decay with $\alpha = 1/2$ ends at a rate of $0.036$, about a
third of $0.1$, and should end at a third of $0.27$, near $0.1$. With
$\alpha = 1$ the final rate is $0.0033$, a thirtieth of $0.1$, which
predicts about $0.01$. And exponential decay ends at a rate of about
$10^{-27}$. If the formula is to be believed, it should beat them all.

```{.python .input #lr-scheduler-four-schedules-on-the-valley}
%%tab pytorch, jax
def sgd_valley(schedule, num_steps=3000, num_runs=1000):
    """Mean squared distance to the minimum of SGD runs on the valley."""
    rng = np.random.default_rng()
    h = np.array(d2l.VALLEY_CURVATURES)  # Curvatures along x1 and x2
    x = np.tile([-5.0, -2.0], (num_runs, 1))
    dist = []
    for t in range(num_steps):
        x = x - schedule(t) * (h * x + rng.normal(size=x.shape))
        dist.append((x ** 2).sum(axis=1).mean())
    return dist

valley_schedules = {
    'constant': lambda t: 0.1,
    'exponential': lambda t: 0.4 * 0.98 ** t,
    'polynomial 1/2': lambda t: 0.4 / (1 + t / 25) ** 0.5,
    'polynomial 1': lambda t: 0.4 / (1 + t / 25)}
dists = {}
for name, schedule in valley_schedules.items():
    dists[name] = sgd_valley(schedule)
    budget = sum(schedule(t) for t in range(3000))
    print(f'{name:15s} budget {budget:6.1f}, final rate {schedule(2999):.2g}, '
          f'mean squared distance {dists[name][-1]:.3f}')
d2l.plot(np.arange(1, 3001), list(dists.values()), 'step',
         'mean squared distance', legend=list(dists), xscale='log',
         yscale='log')
```

Three of the predictions hold. The constant rate stalls within a few
hundredths of $0.27$. Polynomial decay with $\alpha = 1/2$ ends near $0.1$
and is still falling. This is the run that we promised: the squares of its
rates have an infinite sum, and it converges all the same. With
$\alpha = 1$ the distance falls like $1/t$ and ends at $0.011$, eight to
ten times closer, with a budget a quarter as large. Once the starting
distance has been covered, the size of the last steps decides.

The fourth prediction fails, and its failure teaches more than the three
successes. Exponential decay freezes at about $0.13$, more than ten times
what its two numbers allow. Its rate ends at $10^{-27}$, and it has lost to
a schedule whose rate ends at $0.036$. Its budget was enough for the start.
It was not enough for the noise. The noise that step $s$ adds can be
removed only by the steps that come after it, and under exponential decay
the budget still to come is about $\eta_s/(1 - \gamma) = 50\,\eta_s$. On
the flat direction, with $\lambda = 0.2$, the later steps remove less and
less once $50\, \eta_s \lambda$ falls below $1$, that is, from
$\eta_s = 0.1$ on, and most of the noise that is left at the end was added
after that point. The formula gave no warning because its second term
assumed a rate that changes slowly. Adding up what every step leaves behind
gives a frozen squared distance of $(1 - \gamma)\sigma^2/(4\lambda^2)$,
which is $0.125$ along $x_1$ (Exercise :exref:`ex_scheduler-frozen-noise`).
With the remnant of the start, this is the $0.13$ that the cell printed.

Noise is removed only by steps that are still to come. The rule holds for
every schedule, and it corrects a tempting reading of the two numbers: a
schedule that reaches zero has not thereby removed its noise. The rate must
also stay small for long enough, and a direction of curvature $\lambda$
needs a remaining budget of about $1/\lambda$.

Deep networks are trained for a fixed number of steps $T$, and for a fixed
$T$ conditions on infinite sums say little. What remains of them are the
two numbers at the end of the run: the budget $B_T$, which must cover the
distance to a good solution, and the size of the last steps, which must be
small for long enough to remove the noise. Worst-case theory, for what it
is worth on a network, points the same way: for the last iterate of SGD on
a convex problem, the schedule with the best guarantee decays linearly to
zero :cite:`Defazio.Cutkosky.Mehta.ea.2023`
(:numref:`subsec_mdl-fixed-budget-rates`). No theorem ranks one shape
against another on a deep network, so from here on we measure.

## Decay on a Language Model

The valley leaves us with two numbers to watch and one rule about noise.
How much of this survives on a network? The formula does not. It was
derived for SGD on a quadratic, and the schedules in use today were shaped
on language models trained with Adam. Under Adam the learning rate is a
step length. Every weight moves by a fraction of $\eta$ per step, whatever
the size of its gradient, and at a constant rate it keeps hopping at that
length (:numref:`sec_adam`). The hop takes the place of the noise ball, and
like the ball it shrinks only when $\eta$ does. The budget becomes a length
as well, about the longest path that a weight can travel. And where the
valley let us measure the distance to the minimum, a network lets us read
only the loss. So we keep the two questions, how far and how small, and put
them to an experiment.

### The Training Setup

The model is `TinyLM` of :numref:`sec_test_models`, a transformer with two
blocks and about 470,000 parameters that predicts the next byte of
Wikipedia text. A run takes 2,000 steps on minibatches of 64 windows of 64
bytes. It reads its text only once, so whatever it gains along the way is
prediction and not memorization. The optimizer is AdamW without weight
decay, which is Adam, at a peak rate of $0.003$, the best constant rate of
:numref:`sec_adam`. We read a run through its validation loss, in nats per
byte. Five runs of this setting ended within about a hundredth of a nat of
each other in :numref:`sec_test_models`. To be on the safe side, that
section calls a difference a result only from $0.05$ nats, and so shall we.

One caveat applies to every run below. Without weight decay the weight
matrices grow during training, from an RMS of $0.045$ to about $0.14$ over
these 2,000 steps (:numref:`sec_adamw`). Relative to the weights a step of
fixed length shrinks to a third, so even our constant rate is a decay in
effect.

From here on a schedule is a function of the step $t$ that returns the
learning rate, `adamw(model, schedule)` returns an optimizer that follows
it, `tiny_lm(seed)` builds the model, and `new_seed()` draws a fresh seed.
Later cells train with `d2l.train_steps` of :numref:`sec_test_models`,
which takes a given number of further steps and returns their losses.

```{.python .input #lr-scheduler-language-model-1}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=64)
new_seed = lambda: random.getrandbits(31)

def tiny_lm(seed):
    torch.manual_seed(seed)  # The initialization and the minibatch order
    return d2l.TinyLM(len(data.vocab)).to(d2l.try_gpu())

class Scheduled:
    """An optimizer whose learning rate at step t is schedule(t)."""
    def __init__(self, optimizer, schedule, start=0):
        self.optimizer = optimizer  # Its rate is 1, which LambdaLR multiplies
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, schedule, last_epoch=start - 1)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

def adamw(model, schedule):
    """AdamW without weight decay at the learning rate schedule(t)."""
    return Scheduled(torch.optim.AdamW(model.parameters(), lr=1.0,
                                       weight_decay=0), schedule)

```

```{.python .input #lr-scheduler-language-model-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
new_seed = d2l.get_seed

def tiny_lm(seed):
    return d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))

def adamw(model, schedule):
    """AdamW without weight decay at the learning rate schedule(t)."""
    return nnx.Optimizer(model, optax.adamw(schedule, weight_decay=0),
                         wrt=nnx.Param)

```

:begin_tab:`pytorch`
`LambdaLR` multiplies the optimizer's own rate, which is therefore set to
1, by `schedule(t)`, and `Scheduled` advances it after every optimizer
step. The seed of `tiny_lm` fixes the initial weights and the order of the
minibatches, so two runs with one seed read the same text as long as
nothing else draws minibatches in between.
:end_tab:

:begin_tab:`jax`
Optax takes the schedule in place of the learning rate and calls it inside
the compiled training step, which is why the schedules below use `jnp`.
The seed of `tiny_lm` fixes the initial weights only, and every run draws
its own minibatches.
:end_tab:

### Decay Shapes Compared
:label:`subsec_scheduler-decay-shapes`

We compare three shapes with the same peak, $0.003$, over $T = 2{,}000$
steps. The first is the constant rate. The second is *cosine decay*
:cite:`Loshchilov.Hutter.2016`, which follows half a cosine wave from the
peak rate $\eta_0$ at step $0$ down to a final rate $\eta_T$ at step $T$,

$$
\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2}\left(1 + \cos\frac{\pi t}{T}\right).
$$
:eqlabel:`eq_scheduler-cosine`

Here $\eta_T = 0$. Nothing singles out the cosine among curves that fall
smoothly from the peak to zero; it has stayed in use because it has no drop
times to tune. (Its authors restarted the wave periodically, a variant
that :numref:`sec_optimization-further-reading` describes.) The third
shape holds the peak for the first four fifths of the run and then falls
linearly to zero. We call it the *late decay*; in the code it is
`stable_decay`. The cell also defines `warmup`, a linear ramp that we put
to use in :numref:`subsec_scheduler-warmup`.

```{.python .input #lr-scheduler-language-model-2}
%%tab pytorch
def constant(t, eta=3e-3):
    return eta

def cosine(t, eta=3e-3, T=2000):
    return eta * (1 + math.cos(math.pi * t / T)) / 2

def stable_decay(t, eta=3e-3, decay_start=1600, T=2000):
    """eta until decay_start, then a linear decay to zero at step T."""
    return eta * min(1, (T - t) / (T - decay_start))

def warmup(schedule, num_steps):
    """schedule, ramped up linearly over its first num_steps steps."""
    return lambda t: min(1, (t + 1) / num_steps) * schedule(t)
```

```{.python .input #lr-scheduler-language-model-2}
%%tab jax
def constant(t, eta=3e-3):
    return eta

def cosine(t, eta=3e-3, T=2000):
    return eta * (1 + jnp.cos(jnp.pi * t / T)) / 2

def stable_decay(t, eta=3e-3, decay_start=1600, T=2000):
    """eta until decay_start, then a linear decay to zero at step T."""
    return eta * jnp.minimum(1, (T - t) / (T - decay_start))

def warmup(schedule, num_steps):
    """schedule, ramped up linearly over its first num_steps steps."""
    return lambda t: jnp.minimum(1, (t + 1) / num_steps) * schedule(t)
```

What do the two numbers say? Picture the three schedules as curves over
the 2,000 steps, with the budget as the area under each. The constant rate
is a rectangle. It has the largest budget, $2{,}000 \cdot 0.003 = 6$, and
it never lowers its rate. Cosine decay ends at zero, but half a cosine wave
covers half the rectangle, so its budget is $3$. The late decay is the
rectangle with a triangle cut from its last fifth. Its budget is
$4.8 + 0.6 = 5.4$, nine tenths of the largest, and it ends at zero as
well. If both numbers count on a network, the late decay should end lowest.
Between the other two the numbers cannot decide: the constant rate has the
budget, and the cosine has the small last steps.

Each shape is trained from two seeds. The cell prints the budgets and the
validation loss of both runs at step 1,500 and at the end, and it plots the
training loss of the second seed, averaged over blocks of 25 steps.

```{.python .input #lr-scheduler-decay-on-a-language-model-1}
%%tab pytorch, jax
schedules = {'constant': constant, 'cosine': cosine,
             'stable, then decay': stable_decay}
val_loss = lambda model: math.log(d2l.lm_perplexity(model,
                                                    data.val_dataloader()))
seeds, curves = [new_seed(), new_seed()], {}
print(f'{"schedule":<20}{"budget":>7}  validation loss at step 1500, 2000')
for name, schedule in schedules.items():
    at_1500, at_2000 = [], []
    for seed in seeds:
        model = tiny_lm(seed)
        optimizer = adamw(model, schedule)
        losses = d2l.train_steps(model, data, optimizer, 1500)
        at_1500.append(val_loss(model))
        losses += d2l.train_steps(model, data, optimizer, 500)
        at_2000.append(val_loss(model))
    curves[name] = d2l.smooth(losses)
    budget = sum(float(schedule(t)) for t in range(2000))
    print(f'{name:<20}{budget:7.2f}  ', ' '.join(f'{v:.3f}' for v in at_1500),
          '  ', ' '.join(f'{v:.3f}' for v in at_2000))
d2l.plot(list(range(25, 2001, 25)), list(curves.values()), 'step',
         'training loss', legend=list(curves), xlim=[500, 2000],
         ylim=[1.45, 1.95])
```

The loss drops when the rate falls. Until step 1,600 the late decay is the
constant rate. Over the last 500 steps the constant runs gain $0.04$ to
$0.05$ nats, and the runs with the late decay gain $0.10$. The plot shows
the moment: soon after step 1,600, where its rate begins to fall, the
curve of the late decay steepens, and it ends well below the other two.

Where do the runs end? The late decay ends near $1.50$, the constant rate
at $1.56$ to $1.60$, and cosine decay at about $1.55$ to $1.57$. The late
decay is the lowest in every run, as the two numbers said it should be.
How firmly? Against the constant rate, firmly: each of its runs ends
$0.05$ to $0.09$ nats below the constant run of the same seed, which is a
result by our rule. Against the cosine run of the same seed it leads by
$0.04$ to $0.06$. That is the size of the margin we ask for and no more,
so we do not call it a win.

Cosine decay and the constant rate end within $0.03$ of each other for
either seed, a tie by our rule. They tie for opposite reasons. The constant
run travels farthest and never stops hopping. The cosine run leads at step
1,500, when its rate is down to 15% of the peak, but with so little rate
left it hardly travels any more: it gains only $0.02$ to $0.03$ over the
last 500 steps, and its curve levels off. The constant rate has the budget,
the cosine has the small last steps, and on this run one counts about as
much as the other.

:begin_tab:`pytorch`
The constant run and the late decay of one seed read the same minibatches,
so until step 1,600 they are one run, and their losses at step 1,500 agree
to the last digit.
:end_tab:

:begin_tab:`jax`
The runs of one seed share their initial weights but not their
minibatches, so the constant run and the late decay differ at step 1,500
already, by up to a few hundredths.
:end_tab:

How much each number matters depends on the run. On a noisy quadratic that
reaches its noise ball within ten steps, travel no longer helps, and cosine
decay ends three to four times lower than the late decay, because it spends
more steps at small rates (:numref:`subsec_mdl-step-size-decay`; Exercise
:exref:`ex_scheduler-same-two-numbers` runs such a case on the valley).
`TinyLM` after 2,000 steps is at the other extreme: its loss is still
falling when the run ends, and the shape that gives up the least budget for
its decay ends lowest.

The table also says when a comparison should be read. At step 1,500 the
cosine runs are ahead of both other shapes. Five hundred steps later the
late decay has passed them, and it made its gain with less budget than the
constant rate had over those steps. What it shed was the hop. A loss that
is read while the rate is high includes the hop, so it understates how good
the weights are, by an amount that depends on the rate at that moment.
Rankings of optimizers that were read before the decay have been found to
flip by the end :cite:`Wen.Hall.Ma.ea.2025`. The earlier sections compared
at a constant rate on purpose, to vary one thing at a time. A final verdict
on two optimizers needs decayed runs.

## Warmup
:label:`subsec_scheduler-warmup`

Decay looks after the end of a run. *Warmup* looks after its beginning: it
raises the rate from near zero to its peak over the first steps. No run of
this chapter has needed one, since at a rate of $0.003$ `TinyLM` trains
from the first step. Higher rates are another matter. In
:numref:`sec_adam`, Adam at $0.03$ ended near $2.8$ nats, where its best
rate reached $1.6$. Was the rate simply too large? Only in part. Most of
the damage is done within a few steps, and Adam's own update says why.

### Cold Start and Ramp

Adam's first step moves every coordinate by $\eta$, on the evidence of a
single minibatch (:numref:`sec_adam`). The weight matrices of `TinyLM`
start with an RMS of about $0.045$. At $\eta = 0.003$ the first step
changes a weight by 7% of that size. At $\eta = 0.03$ it changes every
weight by two thirds of its size, in whatever direction 64 windows of text
happened to indicate. That is a rough way to treat an initialization that
was chosen with some care. Steps of this size are taken only at the very
start: once Adam's average of the gradients covers ten minibatches or so,
their disagreements cancel in part, and the step settles at a fraction of
$\eta$. The remedy can therefore be short as well.
`warmup(schedule, 20)` multiplies a schedule by $(t + 1)/20$ during its
first 20 steps, 1% of our run, so its first step should move the weights
by about 3% of their size.

We run the rate $0.03$ twice from the same initial weights, from a cold
start and with the ramp. How large are the steps against the weights? For
the weight matrices of the linear layers the cell divides the RMS of the
change in one step by the RMS of the weights, at steps 1, 5, 20 and 100,
and then it trains both runs to step 2,000.

```{.python .input #lr-scheduler-cold-and-ramped-1}
%%tab pytorch
def linear_weights(model):
    return [p.detach().clone() for name, p in model.named_parameters()
            if p.ndim == 2 and 'emb' not in name]
```

```{.python .input #lr-scheduler-cold-and-ramped-1}
%%tab jax
def linear_weights(model):
    params = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return [p[...] for path, p in params
            if p.ndim == 2 and 'emb' not in str(path)]
```

```{.python .input #lr-scheduler-cold-and-ramped-2}
%%tab pytorch, jax
def relative_change(old, new):
    """RMS change of a list of weight matrices relative to their RMS."""
    return math.sqrt(float(sum(((n - o) ** 2).sum() for o, n in zip(old, new))
                           / sum((o ** 2).sum() for o in old)))

peak = lambda t: constant(t, eta=3e-2)
seed = new_seed()
print('step size / weight size at steps 1, 5, 20, 100; final loss')
for name, schedule in [('cold start', peak),
                       ('20-step ramp', warmup(peak, 20))]:
    model = tiny_lm(seed)
    optimizer = adamw(model, schedule)
    losses, sizes = [], []
    for step in range(1, 101):
        before = linear_weights(model)
        losses += d2l.train_steps(model, data, optimizer, 1)
        if step in (1, 5, 20, 100):
            sizes.append(relative_change(before, linear_weights(model)))
    losses += d2l.train_steps(model, data, optimizer, 1900)
    print(f'{name:<14}', ' '.join(f'{s:.2f}' for s in sizes),
          f'  training {d2l.final_loss(losses):.3f},',
          f'validation {val_loss(model):.3f}')
```

The first step of the cold run changes the weights by two thirds of their
size, as predicted, and its fifth step still by about 15%. After that its
steps are small: at step 20 a step is a fiftieth of the weights, smaller
than the step of the ramped run at the same moment and the same rate. The
arithmetic of the first steps suggests why. Five steps of up to $0.03$ per
entry can add $0.15$ to entries that began near $0.045$, and a step of
fixed length is small against weights that have grown severalfold. (The
cell does not print the weights; Exercise
:exref:`ex_scheduler-warmup-length` does.) At none of the four steps does
the ramped run move its weights by more than a tenth of their size.

By step 100 both runs take steps of about a hundredth of their weights,
and from then on they are trained alike. They end far apart all the same.
The cold run stalls near $2.8$ nats, nearer to the $3.17$ of a model that
knows only how frequent each byte is (:numref:`sec_adamw`) than to any
loss that a trained `TinyLM` has shown us. The ramped run ends below
$2.0$, about a nat lower. The two schedules differ in twenty steps out
of two thousand.

Why can 1,900 further steps not repair what the first few did? We have
not taken that apart. The experiment shows when the run is lost and how
little it takes to save it.

### Warmup in Practice

Warmup did not make the high rate a good one. The ramped run at $0.03$
still ends more than a tenth of a nat above the constant-rate runs at
$0.003$ in the table of :numref:`subsec_scheduler-decay-shapes`. On this
model warmup rescues a rate that we would not choose anyway, since the best
rate trains from a cold start. It becomes necessary when the best rate lies
above what a cold start survives, as it often does for large models and
large minibatches. Large minibatches are where warmup entered practice. When
:citet:`Goyal.Dollar.Girshick.ea.2017` trained ImageNet classifiers on
minibatches of thousands of images, with learning rates scaled up to
match, the first epochs went wrong, and they introduced a *gradual*
warmup, a linear ramp over the first five epochs.

Their optimizer was SGD with momentum, whose first step is not normalized,
so Adam's first steps cannot be the whole explanation. The other part is
curvature, and we have met it. Under gradient descent the sharpness of the
loss, its largest curvature, cannot stay above $2/\eta$, so a rate that
rises gradually lowers this ceiling a little at a time, and the run reaches
its peak in a region flat enough for it
(:numref:`subsec_gd-edge-of-stability`).
:citet:`Kalra.Barkeshli.2024` find that warmup acts in this way under Adam
as well, where the curvature that counts is that of the problem after Adam
has rescaled every coordinate. We did not measure it. That sharpness is
laborious to compute, and the size of a step can be read off.

The two choices that a warmup asks for are easy ones. Its length is
customarily one to a few percent of the run, and it costs almost nothing: a
linear ramp over $W$ steps gives up the budget of about $W/2$ steps at the
peak, ten steps out of 2,000 here. For the peak, sweep rates a factor of
three apart with the ramp in place, as in :numref:`sec_test_models`, and
take the best. As a rule of thumb, expect it a factor of three to ten
below the first rate that stalls: in the sweep of :numref:`sec_adam`,
without a ramp, the runs stalled at $0.03$ and the best rate was $0.003$.
A rate that only just trains is not the one to choose, as the ramped run
at $0.03$ has just shown.

## Warmup–Stable–Decay

Cosine decay has a practical defect that our comparison could not show: it
fixes the length of training, the horizon $T$, at the first step. A
checkpoint taken before $T$ is not a finished model, since its rate has not
yet come down. And if we decide after the fact that training should run
twice as long, the cosine has already reached zero, and the choices are to
start over or to splice on a new schedule. For a single run of known length
none of this matters. It matters when data keep arriving, when the loss is
still falling at $T$, and when one run should yield models of several
training lengths, as a study of how the loss falls with the amount of
training requires :cite:`Hagele.Bakouch.Kosson.ea.2024`.

The late decay has no such defect. With a warmup in front it is the
*warmup–stable–decay* (WSD) schedule :cite:`Hu.Tu.Han.ea.2024`: it warms
up, holds the peak rate for most of the run, and decays only in a short
final phase, typically the last 10–20% of the steps. We call the long
stretch at the peak rate the *plateau*. It is a plateau of the learning
rate and not of the loss, which keeps falling on it. Every checkpoint of
the plateau can start a decay, so the decision how long to train can be
made late.

### The River Valley

Why should a short decay at the end suffice? The published loss curves of
large WSD runs show the same abrupt drop as ours, and
:citet:`Wen.Li.Wang.ea.2024` explain it with the geometry of a *river
valley*: a loss surface with steep walls on both sides of a floor that
winds gently downhill. At the high constant rate the iterate bounces
between the walls while it drifts quickly along the floor. The bouncing
raises the measured loss, since the iterate is always partway up a wall,
but the drift is real progress, over a distance along the floor that small
steps could not cover in the same time. The decay stops the bouncing: the
iterate settles onto the part of the floor it has already reached, and the
loss falls quickly.

This is the valley of the beginning of the section, bent into a river. The
travel along the floor is the budget, and the bouncing between the walls
is the noise of the current rate. The walls are steep, and a direction of
large curvature needs only a small remaining budget to lose its noise,
which is why a short decay is enough.

A reader who remembers the exponential schedule will object: its noise
froze in the flat direction, and the floor of the river is the flat
direction. The objection stands. A short decay removes the noise across
the river and leaves the noise along it. In the river-valley model that
does little harm, because nearly all the loss that the noise costs is
across the river. On the valley we were stricter: we measured distance,
and the flat direction held $0.125$ of the frozen $0.13$.

Nobody has surveyed the loss surface of a language model, so the river
valley remains a conjecture, though one that accounts for the drop we
measured. :numref:`fig_opt_river_valley` draws two runs of SGD on such a
valley. The same noise drives both, and the only difference between
bouncing indefinitely and landing on the floor is what the learning rate
does at the end.

![The river-valley model of warmup–stable–decay. Stochastic gradient descent on a winding valley with steep walls: after a brief warmup, the constant high learning rate bounces off the walls yet keeps moving downstream, while the branch whose learning rate decays settles onto the valley floor it had already reached.](../img/mdl-opt-river-valley.svg)
:label:`fig_opt_river_valley`

### Branching Off the Plateau
:label:`subsec_scheduler-branching`

If the bouncing is all that separates a checkpoint of the plateau from a
finished model, then a decay started from *any* such checkpoint should
finish it. We test this the way WSD is used. One run stays on the plateau
and trains at the constant rate $0.003$ for 2,000 steps. At steps 500,
1,000 and 1,500 we copy it, and the copy, a *branch*, decays linearly to
zero over 250 steps while the plateau run carries on. A branch must copy
more than the weights. Adam's two averages and its step count belong to
the state of a run (:numref:`subsec_adamw-memory`), and a branch that
started them afresh would open with the full-length first steps of a cold
start. `branch` therefore copies the model together with the optimizer.

```{.python .input #lr-scheduler-branching-1}
%%tab pytorch
def branch(model, optimizer, schedule):
    """Copies of model and optimizer that continue under schedule."""
    model, adam = copy.deepcopy((model, optimizer.optimizer))  # With moments
    return model, Scheduled(adam, schedule, optimizer.scheduler.last_epoch)
```

```{.python .input #lr-scheduler-branching-1}
%%tab jax
def branch(model, optimizer, schedule):
    """Copies of model and optimizer that continue under schedule."""
    model = nnx.clone(model)
    twin = adamw(model, schedule)
    nnx.update(twin, nnx.clone(nnx.state(optimizer)))  # Moments, step count
    return model, twin
```

For comparison one more run is planned from the start to last 1,250 steps,
constant to step 1,000 and then the same decay, from the initial weights
of the plateau run. If branching works, the branch taken at step 1,000
should land where this run does.

```{.python .input #lr-scheduler-branching-2}
%%tab pytorch, jax
seed = new_seed()  # The planned run below starts from the same weights
model = tiny_lm(seed)
optimizer = adamw(model, constant)
plateau, branched = {}, {}
for step in range(250, 2001, 250):
    d2l.train_steps(model, data, optimizer, 250)
    plateau[step] = val_loss(model)
    if step in (500, 1000, 1500):  # A 250-step decay on a copy
        decay = lambda t, s=step: stable_decay(t, decay_start=s, T=s + 250)
        twin, twin_optimizer = branch(model, optimizer, decay)
        d2l.train_steps(twin, data, twin_optimizer, 250)
        branched[step + 250] = val_loss(twin)
planned = tiny_lm(seed)  # Decays from step 1000 to 1250 by design
d2l.train_steps(planned, data, adamw(planned, lambda t: stable_decay(
    t, decay_start=1000, T=1250)), 1250)
print('step  plateau  branch')
for step, v in branched.items():
    print(f'{step:>4} {plateau[step]:8.3f} {v:7.3f}')
planned_loss = val_loss(planned)
print(f'planned run of 1250 steps: {planned_loss:.3f}')
d2l.plot([list(plateau), [1250]] + [[s - 250, s] for s in branched],
         [list(plateau.values()), [planned_loss]] +
         [[plateau[s - 250], v] for s, v in branched.items()],
         'step', 'validation loss', legend=['plateau', 'planned', 'branches'],
         fmts=('-', 'ko', 'r--', 'r--', 'r--'))
```

Each branch ends below the plateau run at the same step, by $0.02$ to
$0.03$ nats at step 750 and by $0.05$ to $0.06$ at steps 1,250 and 1,750. A
branch shares its weights and its optimizer state with the plateau up to
the branch point, so only 250 steps separate the two numbers in a row. The
two later gains are about the $0.05$ that we ask even of unrelated runs.
The gain at step 750 is too small to count by that rule, though it has the
same sign. A reason for its size is at hand: at step 500 the plateau is
still falling steeply, and a branch that lowers its rate there gives up
travel that it still needs.

Does a branch land where a planned run would? The branch taken at step
1,000 ends within a few hundredths of a nat of the run that was planned for
1,250 steps, less than the $0.05$ we ask of a difference, although nobody
chose its length when training started.

This property made WSD popular for training large models. One long
plateau, with short decays branched off it whenever a model is needed,
yields models of many training lengths for little more than the compute
of the longest. Cosine decay needs a separate full run for each
:cite:`Hagele.Bakouch.Kosson.ea.2024`. Our three models needed the plateau
up to step 1,500 and three decays of 250 steps, 2,250 steps in all; three
separate runs would have taken 3,750.

Which decay is best when the length is known remains open, and the
differences are small. On large language models,
:citet:`Hagele.Bakouch.Kosson.ea.2024` found a constant rate with a final
decay on par with cosine decay at the training lengths that they tried. In
the sweeps of :citet:`Bergsma.Dey.Gosal.ea.2025` a linear decay to zero
over the whole run was consistently ahead of the customary cosine decay to
a tenth of the peak. Our own comparison could not separate the late decay
from the cosine either. What the reports and our runs agree on is the end:
a schedule should finish at zero or close to it.

All of this concerns the size of the step. The noise that the step has to
contend with, the third of the three decisions, we took as given.
:numref:`sec_batch_size` asks what changes when a step averages over more
examples.

## Summary

A schedule gives the learning rate at every step, and two numbers describe
what its decay achieves: the budget $\sum_s \eta_s$, the area under the
schedule, which sets how far the iterate can travel, and the size of the
last steps, which sets the noise it ends with. Reaching zero is not
enough, as exponential decay showed on the valley: noise is removed only
by steps that are still to come, so the rate must stay small for a while.
On `TinyLM` with AdamW the loss dropped when the rate fell, and a linear
decay over the last fifth of the run ended $0.05$ to $0.09$ nats below the
constant rate. At the other end of a run, a ramp over the first 20 steps
let a rate train that stalls from a cold start. For a new run this
suggests warmup–stable–decay: a ramp over about 1% of the steps, a peak
found by a sweep with the ramp in place, and a decay to zero that starts
late, or on demand from any checkpoint of the plateau, optimizer state
included.

## Exercises

1. [code] **Frozen noise.** The run with exponential decay in
   :numref:`subsec_scheduler-valley` froze at a mean squared distance of
   about $0.13$.
    1. Under the rates $\eta_s = \eta_0 \gamma^s$, step $s$ adds noise of
       variance $\eta_s^2 \sigma^2$ to a coordinate of curvature
       $\lambda$, and the later steps contract it by about
       $e^{-2\lambda R_s}$, where $R_s \approx \eta_s/(1 - \gamma)$ is the
       budget that remains after step $s$. Sum these contributions, by an
       integral over $\eta_s$, and show that the noise left at the end has
       variance about $(1 - \gamma)\sigma^2/(4\lambda^2)$ as long as
       $2\lambda\eta_0 \gg 1 - \gamma$.
    1. Predict the final mean squared distance on the valley for
       $\gamma \in \{0.98, 0.99, 0.995\}$, with $\eta_0 = 0.4$ and 3,000
       steps, including what remains of the start, and check the three
       predictions with `sgd_valley`.
    1. Which $\gamma$ would match the $0.011$ of polynomial decay with
       $\alpha = 1$, and about how many steps does that schedule need
       before its rate is too small to matter? What does the answer say
       about choosing $\gamma$ without knowing the length of the run?
   :exlabel:`ex_scheduler-frozen-noise`

1. [code] **Step decay.** Piecewise constant decay with finitely many
   drops violates one of the two Robbins–Monro conditions.
    1. Which one, and why is the schedule a sensible choice nonetheless
       when the number of steps is fixed in advance?
    1. Use :eqref:`eq_scheduler-bias-noise` to say when the next drop
       should come, and how long the last stage must be for its noise ball
       to be reached.
    1. Design a schedule of three stages for 3,000 steps on the valley and
       run it with `sgd_valley`. How close can you come to the $0.011$ of
       polynomial decay with $\alpha = 1$?
   :exlabel:`ex_scheduler-step-decay`

1. [code] **Two schedules with the same two numbers.** ● Consider linear
   decay to zero, $\eta_t = \eta_0 (1 - t/T)$, and cosine decay
   :eqref:`eq_scheduler-cosine` with $\eta_T = 0$.
    1. Show that both have the budget $\eta_0 T/2$, up to rounding, and
       the final rate zero.
    1. For each schedule, express the budget that remains after step $t$
       as a function of the current rate $\eta_t$, for $t$ close to $T$.
       Which schedule has more budget left when its rate has fallen to a
       given small value?
    1. Run both on the valley with `sgd_valley`, with $\eta_0 = 0.4$ and
       $T = 3{,}000$, together with a constant rate of $0.4$ that decays
       linearly to zero over the last fifth of the run. Order the three
       final distances, and explain the order with the rule that noise is
       removed only by steps that are still to come.
   :exlabel:`ex_scheduler-same-two-numbers`

1. [code] **The peak under a decay.** Train `TinyLM` with `adamw` at the
   peaks $\eta \in \{0.001, 0.003, 0.01\}$, once at a constant rate and
   once with `stable_decay`, each from two seeds.
    1. For each peak, compute the gain of the decayed run over the
       constant run of the same seed. Does the gain grow with the peak, as
       a hop of length proportional to $\eta$ suggests?
    1. Is the best peak under the decay the best constant rate? State
       which of your differences exceed the $0.05$ nats that
       :numref:`sec_test_models` asks for.

1. [code] **The length of the decay.** ● `stable_decay(t, decay_start=s)`
   begins its linear decay at step `s`. With `s = 0` it is a linear decay
   to zero over the whole run, the schedule that
   :citet:`Defazio.Cutkosky.Mehta.ea.2023` derive from a worst-case
   analysis and that :citet:`Bergsma.Dey.Gosal.ea.2025` found ahead of
   cosine decay on large language models.
    1. Compute the budgets for $s \in \{0, 1000, 1600, 1900\}$ and
       $T = 2{,}000$, and predict the order of the final validation
       losses on `TinyLM`.
    1. Train each schedule from two seeds and compare with the cosine runs
       of :numref:`subsec_scheduler-decay-shapes`. Which differences
       exceed $0.05$ nats? Does `TinyLM` side with the valley of
       Exercise :exref:`ex_scheduler-same-two-numbers` or with the reports
       from large runs?
    1. Repeat the comparison for runs of 500 steps, with every `s` and
       `T` divided by four. Does the best start of the decay move, and
       why?

1. [code] **Schedule-free SGD.** :citet:`Defazio.Yang.Mehta.ea.2024`
   replace decay by averaging. With an interpolation weight $\beta = 0.9$
   and averaging weights $c_{t+1} = 1/(t+1)$, the iteration is
   $$\mathbf{y}_t = (1-\beta)\mathbf{z}_t + \beta \mathbf{x}_t, \qquad
   \mathbf{z}_{t+1} = \mathbf{z}_t - \eta\, \mathbf{g}_t, \qquad
   \mathbf{x}_{t+1} = (1 - c_{t+1})\mathbf{x}_t + c_{t+1}\mathbf{z}_{t+1},$$
   where $\mathbf{g}_t$ is a stochastic gradient at $\mathbf{y}_t$, and
   $\mathbf{x}_t$ is the point that is evaluated.
    1. Implement it on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$
       with unit-variance Gaussian gradient noise, starting from
       $(-5, -2)$, and plot $f(\mathbf{z}_t)$ and $f(\mathbf{x}_t)$,
       averaged over 100 runs, for 3,000 steps at the constant rate
       $\eta = 0.1$. Compare with the constant and decaying schedules of
       `sgd_valley`.
    1. Explain the difference between the two curves with
       :eqref:`eq_scheduler-bias-noise`, and state which quantity in the
       iteration plays the role of a decaying learning rate.

1. [code] **The length of a warmup.** ● The cold start at $\eta = 0.03$ in
   :numref:`subsec_scheduler-warmup` stalls near $2.8$ nats, and a ramp
   over 20 steps ends below $2.0$. Count a run as trained if its
   validation loss after 2,000 steps is below $2.3$.
    1. Vary the length of the ramp over 2, 5, 10, 20 and 50 steps at
       $\eta = 0.03$. What is the shortest ramp that trains? Compare it
       with the $1/(1 - \beta_1) = 10$ steps that Adam's average of the
       gradients covers.
    1. Keep the ramp of 20 steps and raise the peak from $0.03$ in
       factors of about three until the run no longer trains. A cold
       start trains at $0.01$ (:numref:`sec_adam`). By what factor does
       the ramp raise the largest rate that trains?
    1. For every run of the first two parts, record the largest value of
       `relative_change` over the first 100 steps, the RMS of the weight
       matrices at step 100, and the final validation loss. Is there a
       size of step above which a run fails? Do the weights of the runs
       that fail grow severalfold within their first steps, as the text
       supposed for the cold start?
   :exlabel:`ex_scheduler-warmup-length`

1. [code] **Branching at several lengths.** ● Continue the plateau run of
   :numref:`subsec_scheduler-branching` to step 4,000.
    1. Branch a decay over 250 steps off it at steps 2,000, 3,000 and
       4,000, and compare each branch with the plateau at the same step,
       as in the text. Does the gain from the decay grow, shrink, or stay
       the same as the branch point moves later? What do the two numbers
       of a schedule lead you to expect?
    1. At step 1,000, branch decays over 50, 125, 250 and 500 steps. How
       short can a decay be before its gain shrinks?
    1. Branch once more at step 1,000, but give the copy of the model a
       new optimizer from `adamw`, whose schedule decays from its first
       step, in place of the copied one. Measure `relative_change` over
       its first steps and its validation loss after 10 and after 250
       steps. What does the result imply for checkpoints that store the
       weights only?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1080)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1081)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.9]{.kicker}

Schedules<br>**budget and final rate · decay shapes · warmup · warmup–stable–decay**
:::
:::

::: {.slide title="A Constant Rate Never Settles, and a Cold Start at a High Rate Never Recovers"}
[The problem]{.kicker}

| run | learning rate | what happens |
|:--|:--|:--|
| SGD on the valley $0.1x_1^2 + 2x_2^2$, unit gradient noise | constant 0.1 | wanders around the minimum with standard deviation 0.5 along $x_1$, forever |
| `TinyLM` with AdamW, 2,000 steps | 0.03 from the first step | stalls near 2.8 nats per byte |
| the same | 0.03, raised over the first 20 steps | trains to below 2.0 |

::: {.d2l-note .warn}
The learning rate has to change during a run: up at the start, down at
the end.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- describe a decay by two numbers and predict where it ends;
- explain why exponential decay freezes;
- compute the budget of a decay shape from its formula;
- choose the length of a warmup and the peak rate;
- branch a decay off a constant-rate checkpoint.

**Idea:** travel far, then keep the last steps small for long enough.

**Running examples:** the valley with unit noise; `TinyLM` (0.47 million
parameters, predicts the next byte of Wikipedia text).
:::

::: {.slide title="Schedules Set the Size of the Step over a Run"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | **Schedules** · Batch Size · Scaling Up · Practice |

- **Uses:** the noise ball of constant-rate SGD; Adam's rate as a step
  length; `TinyLM`.
- **Skips:** convergence proofs, warm restarts, schedule-free methods.
:::

::: {.slide title="The Error Splits into a Travel Term and a Noise Term"}
[Decay on the valley]{.kicker}

One coordinate $x$, started at $x_0$: curvature $\lambda$, noise variance
$\sigma^2$, rate $\eta_t$ at step $t$, changing slowly:

$$\mathbb{E}\big[x_t^2\big] \approx e^{-2\lambda B_t}\, x_0^2 + \frac{\eta_t\, \sigma^2}{2\lambda},
\qquad B_t = \sum_{s<t} \eta_s$$

- **Budget** $B_t$, the area under the schedule: how far the iterate can
  travel. $\lambda = 0.2$, $B = 25$: $e^{-5} \approx 0.7\%$ of the start
  is left.
- **Noise ball** $\eta\sigma^2/(2\lambda)$: the spread at a constant rate
  $\eta$. It follows the current rate $\eta_t$.

::: {.d2l-note}
To converge: $B_t \to \infty$ and $\eta_t \to 0$.
:::
:::

::: {.slide title="The Classical Decays Differ in Budget and Final Rate"}
[Decay on the valley]{.kicker}

3,000 steps on the valley (curvatures 0.2 and 4) from $(-5, -2)$; the
decays start at $\eta_0 = 0.4$:

| schedule | $\eta_t$ | budget | final rate |
|:--|:--|--:|--:|
| constant | $0.1$ | 300 | 0.1 |
| exponential, $\gamma = 0.98$ | $\eta_0 \gamma^t$ | $\eta_0/(1-\gamma) = 20$ | $\approx 10^{-27}$ |
| polynomial, $\alpha = 1/2$ | $\eta_0 (1 + t/25)^{-1/2}$ | 200 | 0.036 |
| polynomial, $\alpha = 1$ | $\eta_0 (1 + t/25)^{-1}$ | 48 | 0.0033 |

::: {.d2l-note}
Every budget covers the start: even 20 shrinks $x_1^2$ from 25 by
$e^{-8}$, to under 0.01. The final rates should decide.
:::
:::

::: {.slide title="Check: Where Does Each Schedule End?"}
[Check]{.kicker}

At the constant rate 0.1 the mean squared distance to the minimum stalls
at the noise ball, 0.27. The polynomial schedules end at the rates 0.036
and 0.0033. Where should they end? And exponential decay, at
$10^{-27}$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The noise ball is proportional to the rate:
$0.27 \cdot 0.36 \approx 0.1$ and $0.27 \cdot 0.033 \approx 0.01$. By the
same formula exponential decay should end lowest of all, under 0.01.
:::
:::
:::

::: {.slide title="Three Predictions Hold; Exponential Decay Freezes at 0.13"}
[Decay on the valley]{.kicker}

1,000 runs from $(-5, -2)$; mean squared distance to the minimum:

@!lr-scheduler-four-schedules-on-the-valley
:::

::: {.slide title="Noise Is Removed Only by Steps Still to Come"}
[Decay on the valley]{.kicker}

- Step $s$ adds noise $\eta_s^2\sigma^2$; later steps shrink it by
  $e^{-2\lambda R_s}$ ($R_s$: budget remaining after step $s$).
- Exponential decay: $R_s \approx \eta_s/(1-\gamma) = 50\,\eta_s$, too
  little on $x_1$ ($\lambda = 0.2$) once $\eta_s < 0.1$.
- Left at the end:

$$\frac{(1-\gamma)\,\sigma^2}{4\lambda^2} = \frac{0.02}{4 \cdot 0.04} = 0.125$$

::: {.d2l-note .rule}
A schedule that reaches zero has not thereby removed its noise.
:::
:::

::: {.slide title="On a Network We Keep the Two Questions and Measure"}
[Decay on a language model]{.kicker}

| | on the valley | on `TinyLM` |
|:--|:--|:--|
| optimizer | SGD | AdamW, no weight decay |
| what $\eta$ is | a factor on the gradient | a step length per weight |
| at a constant rate | noise ball | hop of fixed length |
| measure | distance to the minimum | validation loss, nats per byte |

- 2,000 steps, 64 windows of 64 bytes, text read once: no memorization.
- Five runs of one setting ended within about 0.01 (Two Test Models); a
  difference counts from 0.05.
:::

::: {.slide title="Three Shapes with One Peak Have Budgets 6, 3 and 5.4"}
[Decay on a language model]{.kicker}

Peak $\eta_0 = 0.003$, $T = 2{,}000$ steps; a schedule is a function of
the step $t$, and its budget is the area under it:

| shape | $\eta_t$ | area | budget | final rate |
|:--|:--|:--|--:|--:|
| constant | $\eta_0$ | rectangle | 6.0 | 0.003 |
| cosine | $\frac{\eta_0}{2}\big(1 + \cos\frac{\pi t}{T}\big)$ | half of it | 3.0 | 0 |
| late decay | $\eta_0 \min\big(1, \frac{T - t}{0.2\,T}\big)$ | less a triangle | 5.4 | 0 |

::: {.d2l-note}
If both numbers count on a network, the late decay should end lowest:
90% of the budget and a final rate of zero.
:::

[Cosine decay: Loshchilov and Hutter, 2016]{.small}
:::

::: {.slide title="The Late Decay Ends Lowest; Cosine Ties the Constant Rate"}
[Decay on a language model]{.kicker}

Two seeds per shape; validation loss at steps 1,500 and 2,000 (late
decay: "stable, then decay"):

@!lr-scheduler-decay-on-a-language-model-1
:::

::: {.slide title="The Loss Drops When the Rate Falls"}
[Decay on a language model]{.kicker}

| shape | final validation loss | gain over the last 500 steps |
|:--|:--|:--|
| constant | about 1.56 to 1.60 | 0.04 to 0.05 |
| cosine | about 1.55 to 1.57 | 0.02 to 0.03 |
| late decay | about 1.50 | 0.10 |

- Late decay: 0.05 to 0.09 below the constant run of the same seed, a
  result. Ahead of cosine by 0.04 to 0.06, our margin: no verdict.
- Cosine leads at step 1,500, then hardly travels: a tie with the
  constant rate.

::: {.d2l-note .rule}
A loss read at a high rate includes the hop. Compare two methods at the
end of a decayed run.
:::

[Wen et al., 2025]{.small}
:::

::: {.slide title="Check: Which Shape Wins When Nothing Is Left to Travel?"}
[Check]{.kicker}

A noisy quadratic reaches its noise ball within ten steps of a 2,000-step
run. Cosine decay or the late decay: which ends lower, and why?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Cosine. With nothing left to travel, only the removal of
noise counts, and cosine spends far more steps at small rates. Computed
exactly, it ends three to four times lower. `TinyLM` is the opposite
case: its loss is still falling at step 2,000.
:::
:::
:::

::: {.slide title="Adam's First Step Moves Every Weight by η"}
[Warmup]{.kicker}

First step of Adam: $\eta \cdot \mathrm{sign}(\textrm{gradient})$ in every
coordinate, from one minibatch. Weight matrices of `TinyLM` start at RMS
0.045:

| start | first step | relative to the weights |
|:--|--:|--:|
| cold, $\eta = 0.003$ | 0.003 | 7% |
| cold, $\eta = 0.03$ | 0.03 | 67% |
| $\eta = 0.03$, ramp $(t+1)/20$ | 0.0015 | 3% |

::: {.d2l-note}
After ten steps or so the average of the gradients fills and the step
settles at a fraction of $\eta$: a short ramp is enough.
:::
:::

::: {.slide title="A Cold Start at 0.03 Stalls; a 20-Step Ramp Trains"}
[Warmup]{.kicker}

Same initial weights, 2,000 steps; step size relative to the weights:

@!lr-scheduler-cold-and-ramped-2

::: {.d2l-note .warn}
The two schedules differ in 20 steps out of 2,000.
:::
:::

::: {.slide title="Warmup Rescues a High Rate; It Does Not Make It the Best One"}
[Warmup]{.kicker}

- Ramped 0.03 ends above 1.7; cold 0.003 ends near 1.56 to 1.60.
- Needed when the best rate fails cold: large models, large minibatches.

| choice | rule of thumb |
|:--|:--|
| ramp length | 1% to a few percent of the steps |
| peak rate | sweep in factors of 3 with the ramp in place; expect the best 3 to 10 times below the first rate that stalls |

[Goyal et al., 2017]{.small}
:::

::: {.slide title="Check: What Does a Ramp Cost?"}
[Check]{.kicker}

A linear ramp to the peak $\eta = 0.03$ over 20 steps,
$\eta_t = 0.03 \min(1, (t + 1)/20)$, in a run of 2,000 steps. How much
budget does the ramp give up?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The ramp misses $1 - (t+1)/20$ of the peak at step $t$; the
sum over 20 steps is 9.5 peak steps, or $0.285$ of a budget of $60$:
under half a percent.
:::
:::
:::

::: {.slide title="WSD Holds the Peak and Decays Only at the End"}
[Warmup–stable–decay]{.kicker}

| | cosine decay | warmup–stable–decay (WSD) |
|:--|:--|:--|
| length of training | fixed at the first step | chosen when the decay starts |
| checkpoint mid-run | rate still high: unfinished | start a decay from it |
| training longer | restart or splice a schedule | continue the plateau |

WSD: warm up, hold the peak (the *plateau*), decay over the last 10–20%
of the steps.

[Hu et al., 2024; Hägele et al., 2024]{.small}
:::

::: {.slide title="The Decay Settles the Iterate onto the Valley Floor"}
[Warmup–stable–decay]{.kicker}

![](../img/mdl-opt-river-valley.svg){width=60%}

Constant rate: bounce between steep walls, travel along the floor
(budget). Decay: the bouncing stops (noise), and the loss drops.

[A model of the loss surface, not a measurement. Wen et al., 2024]{.small}
:::

::: {.slide title="A Decay Branched off the Plateau Finishes the Model"}
[Warmup–stable–decay]{.kicker}

Plateau: constant 0.003 for 2,000 steps. At steps 500, 1,000 and 1,500 a
copy of the model and of Adam's state decays to zero over 250 steps:

@!lr-scheduler-branching-2
:::

::: {.slide title="The Branch Lands Where a Planned Run Does"}
[Warmup–stable–decay]{.kicker}

| branch taken at step | gain over the plateau at the same step |
|--:|:--|
| 500 | 0.02 to 0.03 nats |
| 1,000 | 0.05 to 0.06 |
| 1,500 | 0.05 to 0.06 |

- Branch and plateau share everything up to the branch point.
- The branch from step 1,000 ends within a few hundredths of a run
  planned for 1,250 steps.

::: {.d2l-note}
Nobody chose the length of the branch when training started.
:::
:::

::: {.slide title="A Branch Copies Adam's State with the Weights"}
[Warmup–stable–decay]{.kicker}

Adam's two averages and its step count belong to the state of a run:

@-lr-scheduler-branching-1

::: {.d2l-note .warn}
A branch with a fresh optimizer opens with the full-length first steps
of a cold start.
:::
:::

::: {.slide title="Check: Three Models from One Run"}
[Check]{.kicker}

You want models trained for 750, 1,250 and 1,750 steps, each ending in a
decay over 250 steps. How many steps do three separate runs take, and how
many does one plateau with three branches take?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Separate runs: $750 + 1{,}250 + 1{,}750 = 3{,}750$ steps. One
plateau to step 1,500 plus three decays of 250: $2{,}250$ steps.
:::
:::
:::

::: {.slide title="A Schedule Travels Far, Then Makes Its Last Steps Small"}
[Wrap-up]{.kicker}

- Error: $e^{-2\lambda B_t} x_0^2 + \eta_t \sigma^2/(2\lambda)$; $B_t$:
  sum of the rates.
- Valley: exponential decay freezes at 0.13. Noise is removed only by
  steps still to come.
- `TinyLM`: a late linear decay ends 0.05 to 0.09 nats below the constant
  rate; cosine ties the constant rate.
- Cold start at 0.03 stalls near 2.8; a 20-step ramp trains.
- WSD: a decay off any plateau checkpoint finishes the model.

**Next:** Batch Size, which lowers the noise at its source.
:::
