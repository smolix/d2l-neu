# Schedules
:label:`sec_scheduler`

The modern recipe of :numref:`sec_training_recipes` did not keep the
learning rate fixed. It raised the rate linearly over the first few epochs
and then let it fall along a cosine curve. Almost every large model today
is trained with a warmup followed by a decay. This section asks why the
learning rate should change during a run, and what shape the change should
have.

The two ends of a run need different things. At the end, a constant learning
rate keeps stochastic gradient descent from settling. On the valley
$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ with unit gradient noise, the iterates
at $\eta = 0.1$ keep wandering around the minimum with a standard deviation
of 0.5 along the flat direction, however long they run (:numref:`sec_sgd`).
Halving the rate halves the squared radius of that cloud, but it also halves
the speed at which the iterates approach it. The start of training has the
opposite problem. The convolutional network of this section trains at a
learning rate of 7.5 if the rate is raised to that value gradually, while a
run that starts at 7.5 stays close to chance accuracy.

<!-- WRITER: the sentence above is false now (no convolutional network, no rate 7.5). New fact: the transformer language model `TinyLM` trained with AdamW at 3e-2, ten times its best constant rate, stalls near 2.8 nats per byte from a cold start (2.70-2.92 over all development runs), while the same peak reached by a linear ramp over the first 20 steps trains to 1.74-1.83. -->

A learning-rate *schedule* addresses both problems. It is a function that
gives the learning rate $\eta_t$ at each point $t$ of training, and it can
be changed without touching the rest of the training loop. An optimizer
makes three decisions (:numref:`sec_optimization-intro`): the descent
direction, the step size over time, and the treatment of noise. The
schedule is the second. We first ask what a decaying rate has to
achieve, on the valley, where the answer can be computed: it must carry
the iterate far enough to reach the minimum, and it must end small enough
to quench the noise. We then train one small network under the classical
decay shapes and under *warmup*, which raises the rate gradually at the
start.
<!-- WRITER: "one small network under the classical decay shapes" is false now: the network half trains `TinyLM` with AdamW under a constant rate, cosine decay and a constant rate with a linear decay over the last fifth; the classical shapes stay on the valley. --> Last comes the warmup–stable–decay schedule used to train large
language models. It holds the rate constant for most of the run, which
lets us decide late in training how long to train.

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

At the end of a run a schedule has two jobs: carry the iterate to the
minimum and shrink the noise around it. On the valley both effects can be
computed in closed form, so we start there.

### Progress and Noise

On the valley each coordinate of SGD evolves on its own. Take one
coordinate with curvature $\lambda$ and gradient noise $\xi_t$ of mean zero
and variance $\sigma^2$, and let the learning rate $\eta_t$ change from step
to step. One step of SGD reads

$$x_{t+1} = (1 - \eta_t \lambda)\, x_t - \eta_t\, \xi_t.$$

The first term pulls the iterate toward the minimum at $0$. The second
pushes it in a random direction. Their effects separate. The mean of $x_t$
shrinks by the factor $1 - \eta_s \lambda \approx e^{-\eta_s \lambda}$ at
every step $s$, so after $t$ steps it has shrunk by about
$e^{-\lambda B_t}$, where

$$B_t = \sum_{s < t} \eta_s$$

is the sum of all learning rates so far, which we call the *budget* of the
schedule. For a constant rate, :numref:`sec_sgd` showed that the noise
keeps the iterate in a *noise ball* of squared radius
$\eta\sigma^2 / (2\lambda)$ around the minimum, for $\eta\lambda \ll 1$.
When the rate changes slowly, compared with the $1/(\eta_t\lambda)$ steps
the iterate needs to forget its past, the ball follows the current rate.
Together,

$$\mathbb{E}\big[x_t^2\big] \approx e^{-2\lambda B_t}\, x_0^2 + \frac{\eta_t\, \sigma^2}{2\lambda}.$$
:eqlabel:`eq_scheduler-bias-noise`

The first term is what remains of the starting distance, and it shrinks
with the budget; the second is the noise ball of the current rate. On the
flat direction of the valley, $\lambda = 0.2$ and $x_0 = -5$, a budget of
$B = 10$ leaves $e^{-2} \approx 14\%$ of the starting distance and a
budget of $25$ leaves $e^{-5} \approx 0.7\%$. A schedule that converges
must therefore do two things: let the budget grow without bound, so that
the first term vanishes, and let the rate go to zero, so that the second
does.

The conditions of :citet:`Robbins.Monro.1951` that :numref:`sec_sgd`
stated, $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$, are a
sufficient form of these two requirements. The second is stronger than
$\eta_t \to 0$, and it is what their proof for general objectives needs. On
the quadratic, :eqref:`eq_scheduler-bias-noise` shows that a rate that goes
to zero with an infinite budget already suffices, as long as it falls
slowly enough for the noise ball to follow it. Polynomial decay with
$\alpha = 1/2$ below confirms this.

Three families of schedules have been used since the early days of
stochastic approximation:

$$
\begin{aligned}
\eta_t &= \eta_i \quad \textrm{for } t_i \leq t < t_{i+1} && \textrm{piecewise constant}, \\
\eta_t &= \eta_0\, \gamma^t && \textrm{exponential}, \\
\eta_t &= \eta_0\, (1 + t/\tau)^{-\alpha} && \textrm{polynomial},
\end{aligned}
$$
:eqlabel:`eq_scheduler-classical`

with a decay factor $\gamma \in (0, 1)$, a time scale $\tau$, and an
exponent $\alpha > 0$. Piecewise constant decay keeps the rate fixed and
cuts it at chosen steps $t_1 < t_2 < \ldots$. Exponential decay multiplies
the rate by $\gamma$ at every step, so its budget is the geometric sum
$\eta_0 / (1 - \gamma)$, which stays finite: it violates the first
condition. Polynomial decay has an infinite budget for $\alpha \leq 1$, and
its squares have a finite sum for $\alpha > 1/2$; the case $\alpha = 1$ is
the rate $\eta_t \propto 1/t$ of :numref:`sec_sgd`.

### Four Schedules on the Valley

We test these predictions with 1,000 independent SGD runs on the valley,
the rows of one array, all started at $(-5, -2)$ with unit gradient noise.
The function `sgd_valley` returns the mean squared distance to the minimum
after each step. We compare four schedules over 3,000 steps: the constant
rate $0.1$; exponential decay from $\eta_0 = 0.4$ with $\gamma = 0.98$; and
polynomial decay from $\eta_0 = 0.4$ with $\tau = 25$ and $\alpha = 1/2$ or
$\alpha = 1$. The decaying schedules start at $0.4$, close to the stability
limit $2/4 = 0.5$ of the steep direction.

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

The three decaying schedules start at the larger rate $0.4$ and approach
the minimum faster than the constant rate $0.1$. While the iterate is far
away, every schedule makes progress; the curves part once the noise
dominates. The constant rate stalls at the noise ball
$\eta\sigma^2 / (\lambda(2 - \eta\lambda))$ of :numref:`sec_sgd`: summed
over both coordinates it predicts a mean squared distance of $0.27$, and
the 1,000 runs land within a few hundredths of it.
Exponential decay does better for a while, then freezes at about $0.13$.
Its budget, $0.4/(1 - 0.98) = 20$, is enough to cover the starting
distance, but its rate falls by 2% per step, faster than the iterate can
follow. Once $\eta_t \lambda$ drops below $0.02$, the iterate no longer
moves far enough to shed the noise it has collected, and that noise stays. Both polynomial schedules keep improving. With
$\alpha = 1/2$ the distance falls in proportion to the rate, as the second
term of :eqref:`eq_scheduler-bias-noise` predicts, although the squares of
this schedule have an infinite sum. With $\alpha = 1$ it falls like $1/t$
and ends eight to nine times closer, with a budget a quarter as large: once
the starting distance has been covered, the final rate decides.

That last observation carries over to deep networks, which do not train
forever but for a fixed number of steps $T$. The conditions on infinite
sums then say little. What matters are the two numbers of
:eqref:`eq_scheduler-bias-noise` at the end of the run: the budget $B_T$,
which must be large enough to cover the distance to a good solution, and
the final rate $\eta_T$, which must be small enough to quench the noise.
Worst-case theory for a fixed $T$ even favors a constant rate if the
average of the iterates is the output. For convex problems, SGD at a
constant rate proportional to $1/\sqrt{T}$, with the average of its
iterates as the output, reaches an error of order $1/\sqrt{T}$. No method
that sees only noisy gradients has a better worst-case rate. For smooth
nonconvex problems the same constant rate is optimal for driving the
gradient toward zero (:numref:`subsec_mdl-fixed-budget-rates`). Training, however, returns the last
iterate, and for the last iterate the worst-case optimal schedule decays
linearly to zero :cite:`Defazio.Cutkosky.Mehta.ea.2023`, again a statement
about the final rate. No theorem ranks one decay shape against another on
a deep network, so we turn to experiments.

## Decay on a Language Model

<!-- WRITER: NEW network half (structure 9; worklist 9.A.3), replacing the batch-normalized LeNet testbed, its tab notes and every reading of it. The model is `TinyLM` of :numref:`sec_test_models` on WikiText-103, trained with AdamW without weight decay at the best constant rate of :numref:`sec_adam`, 3e-3, for 2,000 steps of 64 windows of 64 bytes: the yardstick setting of :numref:`sec_test_models`, whose five repeats spread over 0.026 in validation loss. Single-pass text makes the training loss a held-out loss, so the "budget" here is travel, not memorization (SCHED-9). A schedule is now a function from the step t to the learning rate (not a multiplier, not epochs). Per tab, at most one sentence on how the framework receives it: PyTorch's `LambdaLR` multiplies the optimizer's rate, which is therefore set to 1, and `Scheduled` steps it after every optimizer step, so that `d2l.train_steps` needs no change; Optax takes the schedule as the learning rate and calls it with the step count inside the compiled step, which is why the JAX schedules use `jnp`. In PyTorch, `tiny_lm(seed)` seeds the initialization and the order of the minibatches, so two runs with one seed see the same data; in JAX the seed fixes the initialization only. -->

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

### Decay Shapes Compared
:label:`subsec_scheduler-decay-shapes`

<!-- WRITER: the three decaying shapes, all with the peak 3e-3 and T = 2,000 steps: constant; cosine to zero, :eqref:`eq_scheduler-cosine` with eta_T = 0 (keep the equation, with t in steps); and `stable_decay`, the constant rate followed by a linear decay to zero over the last fifth (steps 1,600 to 2,000), the shape the WSD subsection names. `warmup` multiplies a schedule by a ramp that reaches 1 at its last step (it starts at 1/num_steps, not 0, so that no step is wasted). Budget sum_t eta_t, printed by the next cell: 6.00, 3.00 and 5.40. Predictions from the two numbers before the run: the constant run has the largest budget and the largest final rate; cosine has the smallest final rate but half the budget; stable-then-decay keeps 90% of the budget and ends at zero. -->

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

<!-- WRITER: the decay comparison. Two seeds per schedule; the cell prints each schedule's budget and the validation loss of both runs at step 1,500 and at the end, and draws the smoothed training loss of the second seed. Measured, all development runs (PyTorch: 4 notebook runs and 3 probes, 9 seeds per schedule; JAX: 3 notebook runs and 3 probes, 9 seeds): constant ends at 1.560-1.588 (PyTorch) and 1.558-1.589 (JAX); cosine ends level with it, cosine minus constant between -0.018 and +0.011 (PyTorch) and -0.019 and +0.023 (JAX), so never claim that cosine beats or trails the constant rate here; stable-then-decay ends at 1.503-1.533 (PyTorch) and 1.500-1.518 (JAX), 0.054-0.060 below the constant run of the same seed in PyTorch, where the two runs are identical until step 1,600, and 0.055-0.071 below in JAX, where only the initialization is shared; the ranges of the two shapes do not overlap in either tab. At step 1,500 cosine is ahead of the constant run by 0.013-0.033 in PyTorch and by up to 0.037 in JAX (one JAX run level), and the constant run closes that gap in the last quarter (the plot shows it). Firm lessons: (1) the loss drops when the rate falls: the stable-then-decay run gains 0.05-0.07 in its last 400 steps, every run of both tabs, a paired difference whose range across seeds is under 0.02, against a spread of 0.026 between independent constant runs (:numref:`sec_test_models`); (2) both numbers matter: cosine has the small final rate but gives up half the budget and ends no better than the constant rate; the shape that keeps 90% of the budget and ends at zero is best. Old prose ("Four Decay Shapes", "Five Schedules on the Network", their readings, square-root, exponential and piecewise decay on the network) is gone; the classical shapes stay on the valley. Add the sentence that the appendix computes the same ranking without sampling (worklist 9.B). -->

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

## Warmup
:label:`subsec_scheduler-warmup`

Decay sets the end of the schedule. *Warmup* sets its beginning: it raises
the rate from near zero to its peak over the first part of training. At
initialization the weights are random, and the loss surface around them
can curve sharply in some directions. A learning rate that the network
tolerates after a few epochs of training can then be too large in the
first steps, and the updates can push the weights into a region from
which training at that rate does not recover.

:citet:`Goyal.Dollar.Girshick.ea.2017` met this problem when they scaled
ImageNet training to large minibatches and correspondingly
large learning rates, and their fix, a linear ramp of the rate over the
first five epochs, became standard practice. :citet:`Kalra.Barkeshli.2024`
studied why it works. Early training at a growing rate reduces the
sharpness of the loss around the iterate, its largest curvature, so by the
time the rate reaches its peak, the network has moved to a region where
that peak is stable. Adaptive optimizers such as Adam (:numref:`sec_adam`)
have a second, smaller reason to warm up: Adam's first update has size
$\eta$ in every coordinate, several times its stationary size on a
coordinate whose gradient is mostly noise, until the momentum average has
filled (:numref:`subsec_mdl-schedules-warmup`). Our
network uses plain SGD, so it isolates the first reason.

<!-- WRITER: the sentence above ("Our network uses plain SGD, so it isolates the first reason") is now false: the model is trained with AdamW, so both reasons are present. Tie to :numref:`sec_adam` instead: Adam's first update is eta * sign(g) in every coordinate (its normalized step starts at 1 and falls within tens of steps), so a cold start at a large peak moves every weight by a large fraction of its own size at once. Measured in the next cell (linear layers of TinyLM, RMS change in one step divided by the RMS of the weights before it), identical in all eight notebook runs of both tabs: at 3e-2 the cold run's steps 1, 5, 20, 100 move the weights by 0.67, 0.15-0.16, 0.02, 0.01 of their size; the 20-step ramp's by 0.03, 0.09-0.10, 0.05-0.06, 0.01-0.02. The cold run's first steps are the largest of either run by far (its first step alone changes every weight by two thirds of its size, Adam's sign step of :numref:`sec_adam` at eta = 0.03 against weights of RMS about 0.045); after them its steps shrink to a fiftieth of the weights (the weights have grown several-fold) and the run stalls; the ramp never takes a step above a tenth of the weights. SHARPNESS (SCHED-7): I measured the largest Hessian eigenvalue on a fixed minibatch by power iteration (20-30 HVPs) at steps 0 to 500 of cold and ramped runs at 3e-3, 1e-2, 3e-2, three seeds, both tabs. Raw sharpness is about 0.8 at initialization, rises to 13-23 within ten steps and falls to 5-10 by step 200, in every run of both tabs; it never approaches the plain-SGD ceiling 2/eta = 67 of eta = 3e-2, so it does not explain why Adam fails at that rate (Adam's ceiling concerns the curvature seen through its preconditioner, eta * lambda_max(P^-1 H) < 38 for beta_1 = 0.9, Cohen et al. 2022 "Adaptive gradient methods at the edge of stability", not in d2l.bib). The preconditioned estimate varied by factors of 10-1000 between neighbouring steps and seeds (dominated by coordinates with tiny second moments), so I did not put it in the notebook. The appendix sentence ("Right after initialization that curvature can be large and change quickly ... :numref:`sec_scheduler` measures this in a real model", mdl-adaptive-stochastic-methods.md, subsec_mdl-schedules-warmup) should change: what this section measures is the size of Adam's first steps, and the raw curvature is small at initialization and grows twentyfold in the first ten steps. -->

<!-- WRITER: the LeNet cold start at 7.5, the five-epoch ramp, `warmup` in epochs and the warmup-plus-cosine run are gone. New experiment: AdamW at the constant peak 3e-2, ten times the best constant rate, from a cold start and with a linear ramp over the first 20 steps (1% of the run). Measured final training loss (development, all runs of both tabs): cold 2.70-2.92 (notebook runs 2.785-2.915, validation 2.78-2.93), it stalls near 2.8 nats per byte from about step 50 on and never recovers (with cosine decay instead of the constant rate it still ends at 2.61-2.66); 20-step ramp 1.736-1.826 (notebook runs 1.736-1.787, validation 1.72-1.82). Every ramp from 20 to 200 steps rescues the run (PyTorch W=20/50/100/200, three seeds: 1.72-1.81 except one JAX 200-step run at 1.98), so a ramp of about 1% of the steps suffices here, the length of Adam's inflated first steps. The ramped run at 3e-2 still ends 0.15-0.25 above the cold run at 3e-3 (1.56-1.59): warmup lets a run survive a peak that fails cold, it does not make that peak good. Choose the peak from a short sweep (one grid point below the edge), the ramp at one to a few percent of the steps. At the rates one would choose on this model the ramp's effect is within the spread (3e-3 with cosine: ramp vs cold -0.03 to +0.01; 1e-2 with cosine: gains 0.015-0.06 over three seeds, PyTorch), so do not claim warmup improves the best run here; say it matters when the best peak lies above what a cold start tolerates (large batches, large models; :numref:`sec_practice`). BATCH-17: Goyal et al. introduced *gradual* warmup. -->

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

## Warmup–Stable–Decay

Cosine decay fixes the length of training, the horizon $T$, at the first
step. Every checkpoint before $T$ comes from a run whose rate has not yet
come down. Such a checkpoint is fine for resuming training, but it is not
a finished model, because its noise has not been quenched. If we decide
after the fact that training should run twice as long, the cosine schedule
has already reached its end, and the only clean options are to start over
or to splice on a new schedule. For a single run of known length this is
no limitation. It becomes one when the length is not known in advance,
because data keep arriving or the loss is still falling at $T$, and when
one run should produce models for several budgets, as in a study of
scaling laws (:numref:`sec_scaling`).

The *warmup–stable–decay* (WSD) schedule avoids the fixed horizon
:cite:`Hu.Tu.Han.ea.2024`. It warms up, holds the peak rate constant for
most of the run, and decays only in a short final phase, typically the
last 10–20% of the steps. Every checkpoint of the constant phase can start
a decay, so the decision of how long to train can be made late.
:citet:`Hagele.Bakouch.Kosson.ea.2024` found that WSD matches cosine decay
across budgets when language models are trained this way.

<!-- WRITER: the function is now `stable_decay` (defined with the decay shapes above; with `warmup` in front it is the full schedule). The LeNet WSD-versus-cosine run and its reading (SCHED-9) are gone: the decay comparison above already contains the WSD shape on single-pass text, where the extra progress is not memorization. -->

The same abrupt drop appears in the published loss curves of large WSD
runs, and :citet:`Wen.Li.Wang.ea.2024` explain it with the geometry of a
*river valley*: a loss surface with steep walls on both sides of a floor
that winds gently downhill. At the high constant rate, the iterate bounces
between the walls while it drifts quickly along the floor. The bouncing
raises the measured loss, since the iterate is always partway up a wall,
but the drift is real progress, covering a distance along the floor that
small steps could not cover in the same time. The decay stops the
bouncing: the iterate settles onto the part of the floor it has already
reached, and the loss falls quickly.

In this model the constant phase
travels along the shallow direction and the decay removes the oscillation
across the steep one, which is why the constant phase can run for as long
as training continues and a short decay suffices. It is the noise ball of
:eqref:`eq_scheduler-bias-noise` on a winding valley instead of a bowl:
the travel along the floor is the budget, and the bouncing between the
walls is the noise of the current rate. :numref:`fig_opt_river_valley`
draws two runs on such a valley. The same noise drives both runs, and the
only difference between bouncing indefinitely and landing on the floor is
what the learning rate does at the end.

![The river-valley model of warmup–stable–decay. Stochastic gradient descent on a winding valley with steep walls: after a brief warmup, the constant high learning rate bounces off the walls yet keeps moving downstream, while the branch whose learning rate decays settles onto the valley floor it had already reached.](../img/mdl-opt-river-valley.svg)
:label:`fig_opt_river_valley`

### Branching Off the Plateau
:label:`subsec_scheduler-branching`

If the oscillation across the valley is what keeps a constant-rate
checkpoint from being a finished model, then a decay started from *any*
checkpoint of the constant phase should finish it. We test this with the
workflow WSD is used for.

<!-- WRITER: the branching experiment, rebuilt on TinyLM (structure 9.3). `branch` copies the model together with Adam's moments and step count (PyTorch: one `deepcopy` of model and optimizer, so that the copied moments belong to the copied weights, and a new `LambdaLR` that continues from the step count; JAX: `nnx.clone` and the optimizer state copied into a new optimizer, whose schedule is then called with the copied step count). One plateau run at the constant rate 3e-3 for 2,000 steps, validated every 250 steps; at steps 500, 1,000 and 1,500 a copy of the model and of Adam's state branches off and decays linearly to zero over 250 steps; separately, one planned run of 1,250 steps with the same final decay (constant to step 1,000, then down to zero at 1,250), from another seed. The table prints the validation loss of plateau and branch at the same step; the plot draws the plateau and the three branches. Measured (six notebook runs and two probes, both tabs): each branch ends below the plateau at the same step, by 0.018-0.030 at step 750 (the loss is still falling fast there), 0.050-0.058 at step 1,250 and 0.050-0.061 at step 1,750 (a paired comparison from one checkpoint, so the gain is not run-to-run noise); the branch at 1,250 lands within 0.03 of the planned 1,250-step run from another seed (branch minus planned: -0.016 to +0.027), the spread between independent runs, although nobody chose its length at the start. "for 24 epochs", "We now decide on a budget of 30 epochs" (SCHED-1: write length, not budget), "the test accuracy of the plateau run keeps jumping" are gone with the LeNet runs. -->

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

```{.python .input #lr-scheduler-branching-2}
%%tab pytorch, jax
model = tiny_lm(new_seed())
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
planned = tiny_lm(new_seed())  # Decays from step 1000 to 1250 by design
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

This is the property that made WSD popular for training large models: one
long constant run, with short decays branched off it whenever a model is
needed, gives models at many budgets for little more than the compute of
the longest one, where cosine decay needs a separate full run for each budget
:cite:`Hagele.Bakouch.Kosson.ea.2024`.

<!-- WRITER: the removed sentences ("our optimizer is plain SGD, which keeps no state, so copying the parameters was enough. With momentum or Adam, the state of the optimizer must be copied along with them") are now false: the branches above copy Adam's moments and step count with the weights (`branch`); say that a branch needs both. -->

Which late-decaying shape is best remains open. In published comparisons
on language models, cosine decay, WSD, and linear decay to zero end close
to each other when each is well tuned, with linear decay to zero slightly
ahead in recent sweeps
:cite:`Hagele.Bakouch.Kosson.ea.2024,Bergsma.Dey.Gosal.ea.2025`. What is
settled is a practical point: a constant phase whose checkpoints can be decayed
on demand is more flexible than a schedule that fixes its horizon at the
first step.

## Summary

<!-- WRITER: the network sentences of this Summary describe the deleted LeNet runs and are false (training-loss ranking of piecewise/cosine, "two to three points" of test accuracy, SCHED-14). Measured facts for the new Summary (at most six sentences): on TinyLM with AdamW at 3e-3, a constant rate followed by a linear decay over the last fifth ends 0.05-0.07 nats below the constant rate in every run, while cosine, with half the budget, ends level with the constant rate; a cold start at 3e-2 stalls near 2.8 nats, a 20-step ramp trains it; a decay branched off any plateau checkpoint ends 0.02-0.06 below the plateau and within the run-to-run spread of a run planned for that length. -->

A learning-rate schedule is a function from training time to the learning
rate. On a noisy quadratic the expected squared error splits into what
remains of the starting distance, which shrinks with the budget $\sum_s
\eta_s$, and the noise ball of the current rate, which shrinks with
$\eta_t$. A schedule therefore needs a large budget and a small final
rate. Exponential decay stops moving too early and keeps the noise it has
collected; polynomial decay keeps improving. On a network trained for a
fixed number of epochs, the shapes that stay high and decay late,
piecewise constant and cosine, reach a lower training loss than the shapes
that decay early and match the constant rate's with half its budget. The
schedules that end at a small rate settle their test accuracy, about two to
three points above the constant rate's. Warmup ramps the
rate up at the start and widens the range of usable peak rates.
Warmup–stable–decay holds the peak for most of the run and decays at the
end; a decay branched off any checkpoint of its constant phase yields a
finished model, so the length of training can be chosen late.

## Exercises

<!-- WRITER: Exercises 1, 3, 5 and 6 refer to the deleted LeNet testbed (`train`, epochs, the rate 7.5) and must be rewritten for `TinyLM` with the helpers `tiny_lm`, `adamw`, `branch`, `warmup`, `stable_decay` (worklist 9.B), or dropped. Development data for them: AdamW with cosine decay at peak 1e-2, cold vs 100- and 200-step ramps (PyTorch, three seeds): cold 1.555-1.571, ramp 200 1.512-1.550, gain 0.015-0.059 (a good exercise question: is the gain above the spread?); ramps of 20, 50, 100, 200 steps at 3e-2 all train (1.71-1.81). Exercise 7 (Langevin) moves to sgd.md (prose step). Exercise 8 (linear decay to zero) needs `stable_decay(t, decay_start=0)` and the TinyLM runs instead of `warmup(wsd)` over 30 epochs. -->

1. [code] **The constant baseline.** Train the network of
   :numref:`sec_test_models` with `train` at a constant
   $\eta \in \{0.03, 0.1, 0.3, 0.5\}$ for 30 epochs.
    1. Which rate gives the best final test accuracy, and at which epoch
       does each run's test accuracy stop improving?
    1. How does the gap between the test loss and the training loss at
       epoch 30 change as $\eta$ shrinks? Relate the trend to the noise term
       of :eqref:`eq_scheduler-bias-noise`.

1. [code] **Schedules on the valley.** Use the function `sgd_valley` of
   :numref:`subsec_scheduler-valley` for the following.
    1. Show that without noise the coordinate $x_1$ shrinks over a run by
       a factor of about $e^{-h_1 B}$, where $B = \sum_t \eta_t$ is the
       budget and $h_1 = 0.2$. Choose a decay factor $\gamma$ for
       exponential decay from $\eta_0 = 0.4$ whose predicted final $|x_1|$
       without noise is below $0.01$, run it, and compare its final mean
       squared distance with that of polynomial decay with $\alpha = 1$.
    1. Run the schedule $\eta_t = c/(t + \tau)$ with $\tau = 25$ for
       $c \in \{1, 2.5, 10\}$ for 30,000 steps, and estimate from a
       log–log plot the exponent $p$ in the decay of the mean squared
       distance, $t^{-p}$. Show that along a coordinate of curvature
       $\lambda$ the decay is like $1/t$ only if $c\lambda > 1/2$, and
       explain which coordinate of the valley limits the small values of
       $c$.
    1. Piecewise constant decay with finitely many drops is not covered by
       the Robbins–Monro conditions. Which of the two conditions does it
       violate, and why is it nonetheless a sensible choice when the number
       of steps is fixed in advance? Use :eqref:`eq_scheduler-bias-noise`
       to say when the next drop should come, and test your rule on the
       valley with `sgd_valley`.

1. [code] **Polynomial decay on the network.** Train the network with the
   polynomial decay $\eta_t = 0.3\, (1 + t)^{-\alpha}$, with $t$ in epochs,
   for $\alpha \in \{0.25, 1, 2\}$, and compare with the square-root decay
   of :numref:`subsec_scheduler-decay-shapes`, which is the case
   $\alpha = 1/2$.
    1. Compute the budget $\int_0^{30} \eta_t\, dt$ and the final rate of
       each of the four schedules, and relate them to the final training
       losses and test accuracies you measure.
    1. Polynomial decay satisfies both Robbins–Monro conditions exactly
       when $\alpha \in (1/2, 1]$. Does the best $\alpha$ on this network
       lie in that window? How far do conditions on infinite sums constrain
       a run of 30 epochs?

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

1. [code] **Warmup length and the stability ceiling.** ● The cold start at
   $\eta = 7.5$ in :numref:`subsec_scheduler-warmup` fails, while a
   five-epoch ramp trains. Count a run as trained when its test accuracy
   after ten epochs exceeds 50%.
    1. Vary the length of the ramp in `warmup` from 0.5 to 8 epochs at
       $\eta = 7.5$ and report the test accuracy after ten epochs for
       each. What is the shortest ramp that trains?
    1. Keep the five-epoch ramp. Double the target rate from 7.5 until the
       ramped run no longer trains, and halve it from 7.5 until a cold
       start trains. By what factor does the ramp raise the largest rate
       that trains?
   :exlabel:`ex_scheduler-warmup-length`

1. [code] **Branching at several budgets.** ● Continue the plateau run of
   :numref:`subsec_scheduler-branching` at the constant rate to epoch 50,
   and branch a six-epoch decay off it at each of epochs 30, 40, and 50.
   This is the design of :citet:`Hagele.Bakouch.Kosson.ea.2024`.
    1. For each branch point, record the training loss and the test
       accuracy of the plateau checkpoint, averaged over its last three
       epochs, and of the branch after its decay. Does the gain from the
       decay grow, shrink, or stay constant as the branch point moves
       later?
    1. For the branch at epoch 40, train a fresh run with
       `warmup(cosine)` over the same 46 epochs at the same peak rate,
       with the argument `T` of `cosine` set to 46. Does the branched model
       match it? Compare the epochs spent on the plateau run and its three
       branches with the epochs of three separate cosine runs at the same
       budgets.

1. **From noise ball to sampler.** ● Constant-rate SGD does not converge;
   its iterates fluctuate in the noise ball of the second term of
   :eqref:`eq_scheduler-bias-noise`, a ball whose size is set by the rate.
   Stochastic gradient Langevin dynamics
   :cite:`Welling.Teh.2011` adds Gaussian noise to every step so that the
   iterates approximately sample the density proportional to $e^{-f}$:
   $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{g}_t + \sqrt{2\eta}\,
   \boldsymbol{\zeta}_t, \qquad \boldsymbol{\zeta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
   where $\mathbf{g}_t$ is a stochastic gradient of $f$.
    1. For $f(x) = \frac{\lambda}{2}x^2$ with the exact gradient, derive
       the stationary variance of $x_t$ and compare it with the variance
       $1/\lambda$ of the density proportional to $e^{-f(x)}$.
    1. Compare the injected noise $\sqrt{2\eta}\, \zeta_t$ with the
       gradient-noise term $\eta\, \xi_t$ of SGD. For which gradient-noise
       variance $\sigma^2$ do the two contributions have equal variance,
       and which of them dominates as $\eta$ decays?

1. [code] **Linear decay to zero.** ● :citet:`Defazio.Cutkosky.Mehta.ea.2023`
   argue from a worst-case analysis that the learning rate should decay
   linearly to exactly zero, and :citet:`Bergsma.Dey.Gosal.ea.2025` found
   that warmup followed by linear decay to zero matched or beat cosine and
   WSD in large sweeps of language-model training.
    1. Choose the arguments of `wsd` that turn `warmup(wsd)` into a
       five-epoch warmup to $\eta = 0.3$ followed by linear decay to zero
       over a 30-epoch budget, with no stable phase. Train with it and
       compare the final training loss and test accuracy with the
       `warmup(cosine)` and `warmup(wsd)` runs.
    1. Vary the final rate $\eta_T$ of :eqref:`eq_scheduler-cosine` over
       $\{0.03, 0.01, 0\}$ at the same budget, and repeat the cosine run
       at $\eta_T = 0$ twice more. Measure the effect of the final rate by
       the spread of the three cosine runs, and the effect of the shape by
       the gap between the cosine run at $\eta_T = 0$ and the linear decay
       to zero. Which effect is larger on this network, and does either
       exceed the spread between repeated runs?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1080)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1081)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.6]{.kicker}

Schedules<br>**budget and final rate · decay shapes · warmup · warmup–stable–decay**
:::
:::

::: {.slide title="A Constant Rate Never Settles, and a Cold Start at the Peak Rate Fails"}
[The problem]{.kicker}

- The valley $0.1x_1^2 + 2x_2^2$ with unit gradient noise, $\eta = 0.1$:
  the iterates wander with standard deviation 0.5 along $x_1$ forever.
  Halving $\eta$ halves the squared spread and doubles the time to get
  there.
- A LeNet with batch normalization on Fashion-MNIST: started at
  $\eta = 7.5$ it stays near chance accuracy; ramped up to $7.5$ it
  trains.

::: {.d2l-note .warn}
The learning rate has to change during training: up at the start, down at
the end.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- split the error of SGD under a schedule into a budget term and a noise
  term;
- predict which classical schedules converge on a quadratic;
- write a schedule as a function and hand it to an optimizer;
- explain warmup and choose its length;
- branch a decay off a constant-rate checkpoint.

**Running example:** the valley with unit noise; then a 62,000-parameter
LeNet with batch normalization on Fashion-MNIST, 30 epochs.
**The idea:** a large budget, then a small final rate.
:::

::: {.slide title="Schedules Are the Step Size over Time"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| Gradient Descent | step size against curvature |
| Stochastic and Minibatch GD | sampled gradients, noise ball, batching |
| Momentum | averaging over time |
| Adam and AdamW | a step size per coordinate, decoupled decay |
| **Schedules** | **warmup and decay** |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** the noise ball of SGD; the `d2l.Trainer`; batch
  normalization.
- **Skips:** convergence proofs (math appendix); schedule-free methods.
:::
:::
:::

::: {.slide title="The Error Splits into a Budget Term and a Noise Term"}
[Decay on the valley]{.kicker}

One coordinate, curvature $\lambda$, noise variance $\sigma^2$, rate
$\eta_t$ changing slowly:

$$\mathbb{E}\big[x_t^2\big] \approx e^{-2\lambda B_t}\, x_0^2 + \frac{\eta_t\, \sigma^2}{2\lambda},
\qquad B_t = \sum_{s<t} \eta_s$$

- Budget $B_t$: how far the iterate can travel. $\lambda = 0.2$, $B = 25$:
  $e^{-5} \approx 0.7\%$ of the start remains.
- Rate $\eta_t$: the noise ball of the moment.
- Converge: $B_t \to \infty$ and $\eta_t \to 0$ (Robbins–Monro:
  $\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$ suffices).
:::

::: {.slide title="Exponential Decay Has a Finite Budget"}
[Decay on the valley]{.kicker}

$$
\begin{aligned}
\eta_t &= \eta_i \ \textrm{ for } t_i \leq t < t_{i+1} && \textrm{piecewise constant} \\
\eta_t &= \eta_0\, \gamma^t && \textrm{exponential: } B_\infty = \tfrac{\eta_0}{1-\gamma} \\
\eta_t &= \eta_0\, (1 + t/\tau)^{-\alpha} && \textrm{polynomial: } B_\infty = \infty \textrm{ for } \alpha \leq 1
\end{aligned}
$$

- Exponential: once the rate is tiny the iterate stops moving; the
  distance and the noise left at that point stay.
- Polynomial with $\alpha = 1$: $\eta_t \propto 1/t$.
:::

::: {.slide title="On the Valley Only Polynomial Decay Keeps Improving"}
[Decay on the valley]{.kicker}

1,000 runs, 3,000 steps, start $(-5, -2)$:

@!lr-scheduler-four-schedules-on-the-valley
:::

::: {.slide title="Check: How Far Does Exponential Decay Get?"}
[Check]{.kicker}

On the valley, exponential decay from $\eta_0 = 0.4$ with $\gamma = 0.99$.
What is its budget, and without noise, what fraction of $x_1 = -5$ is left
at the end?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $B = 0.4/0.01 = 40$; $x_1$ shrinks by
$e^{-0.2 \cdot 40} = e^{-8} \approx 3 \times 10^{-4}$, to about $0.002$.
The budget suffices; the noise frozen in when the rate gets too small is
what remains.
:::
:::
:::

<!-- WRITER: the network slides below were rebuilt from the deleted LeNet slides with placeholder text only; rewrite them on TinyLM (slide guide; structure Slides). The problem slide (LeNet at 7.5), the wrap-up slide ("a cold start at 7.5 fails", "two to three points"), the "Where this fits" table and the Goals/Running-example slide still describe the old order (Muon after Schedules, no Two Test Models, no AdamW section) and the LeNet testbed. -->

::: {.slide title="A Schedule Is a Function of the Step"}
[Decay on a language model]{.kicker}

@lr-scheduler-language-model-2

`TinyLM` on WikiText-103, AdamW, 2,000 steps; the peak 3e-3 is the best
constant rate.
:::

::: {.slide title="The Loss Drops When the Rate Falls"}
[Decay on a language model]{.kicker}

@!lr-scheduler-decay-on-a-language-model-1
:::

::: {.slide title="A Cold Start at 3e-2 Stalls; a 20-Step Ramp Trains"}
[Warmup]{.kicker}

@!lr-scheduler-cold-and-ramped-2
:::

::: {.slide title="Warmup Lowers the Sharpness Before the Peak Arrives"}
[Warmup]{.kicker}

- At initialization the loss can curve sharply; the peak rate overshoots.
- Training at a growing rate reduces the sharpness, the largest
  curvature, so the peak becomes stable.
- Adam has a second reason: its step sizes come from the first few
  gradients.
- Warmup widens the range of usable peak rates; it does not remove the
  upper limit.

[Goyal et al., 2017; Kalra and Barkeshli, 2024]{.small}
:::

::: {.slide title="Check: The First Steps of a Ramp"}
[Check]{.kicker}

<!-- WRITER: rewrite this check for the 20-step ramp to 3e-2 (rate at step 1 and step 10; the ramp reaches 1 at its last step: min(1, (t + 1)/20)). -->

A linear ramp to $\eta = 0.03$ over 20 steps, $\eta_t = 0.03 \min(1, (t + 1)/20)$.
What rate does the first step use, and which step is the first at the peak?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta_0 = 0.0015$; step $t = 19$, the twentieth, is the first at
$0.03$.
:::
:::
:::

::: {.slide title="WSD Holds the Peak and Decays Only at the End"}
[Warmup–stable–decay]{.kicker}

<!-- WRITER: the LeNet WSD-versus-cosine plot is gone; the decay comparison (cell lr-scheduler-decay-on-a-language-model-1) shows the WSD shape: level with the constant run until step 1,600, then an abrupt drop. -->

A constant rate for most of the run, a short linear decay at the end.

[Hu et al., 2024; Hägele et al., 2024]{.small}
:::

::: {.slide title="The Decay Settles the Iterate onto the Valley Floor"}
[Warmup–stable–decay]{.kicker}

![](../img/mdl-opt-river-valley.svg){width=60%}

Constant rate: bounce between the walls, travel along the floor (budget).
Decay: stop the bouncing (noise).

[Wen et al., 2024]{.small}
:::

::: {.slide title="A Decay Branched off the Plateau Finishes the Model"}
[Warmup–stable–decay]{.kicker}

A constant-rate plateau of 2,000 steps; at steps 500, 1,000 and 1,500 a copy
of the model and of Adam's state decays to zero over 250 steps:

@!lr-scheduler-branching-2
:::

::: {.slide title="Check: Three Budgets"}
[Check]{.kicker}

You want models trained for 10, 20, and 40 thousand steps. How many steps
do three cosine runs cost, and how many does one WSD run with decays of
20% of each budget cost?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Cosine: $10 + 20 + 40 = 70$ thousand. WSD: a constant phase to
32 thousand plus decays of 2, 4, and 8 thousand: 46 thousand. With momentum
or Adam, branch the optimizer state too.
:::
:::
:::

::: {.slide title="A Schedule Needs a Large Budget and a Small Final Rate"}
[Wrap-up]{.kicker}

- Error under a schedule: $e^{-2\lambda B_t} x_0^2 + \eta_t \sigma^2/(2\lambda)$.
- Exponential decay freezes; polynomial decay keeps improving.
- On the network, late decay (piecewise, cosine) beats early decay in
  training loss; a small final rate settles the test accuracy, two to
  three points above the constant rate.
- Warmup: a cold start at 7.5 fails, a five-epoch ramp trains.
- WSD: a decay off any plateau checkpoint gives a finished model.

**Next:** Muon, which changes the direction of the step for matrices.
:::
