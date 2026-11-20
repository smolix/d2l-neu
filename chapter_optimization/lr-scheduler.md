# Schedules
:label:`sec_scheduler`

Throughout the preceding sections, we held the learning rate $\eta$ constant across training—and twice observed why a fixed step size is suboptimal. Under minibatch SGD (:numref:`sec_sgd`), a constant learning rate traps the iterate in a stationary noise ball of variance $O(\eta\sigma^2 / \lambda)$: halving $\eta$ halves the noise floor but doubles the transient steps required to reach it. Under normalized optimizers such as AdamW (:numref:`sec_adam`) and Muon (:numref:`sec_muon`), the update magnitude remains $O(\eta)$ even as the gradient approaches zero, preventing the iterate from settling at a minimum until $\eta$ is reduced.

Modern training recipes resolve this trade-off with a **learning-rate schedule** $\eta_t$ that varies across steps $t = 0, \ldots, T-1$. Almost every contemporary schedule combines two phases: an initial **warmup** ramp from zero to a peak learning rate $\eta_0$, followed by a **decay** phase toward zero. In this section, we first analyze learning-rate decay on the noisy quadratic valley in terms of two quantities—the **cumulative learning-rate budget** $B_T = \sum_{t=0}^{T-1}\eta_t$ and the **terminal learning rate** $\eta_T$. We then compare constant, cosine, and warmup–stable–decay (WSD) schedules on `TinyLM`, demonstrate why linear warmup prevents early step instability at high peak learning rates, and show how branching short decay phases off a constant-rate plateau produces competitive checkpoints across multiple training horizons.

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

## Learning-Rate Decay on the Noisy Valley
:label:`subsec_scheduler-valley`

Consider a single coordinate of the quadratic valley with curvature $\lambda > 0$ and zero-mean gradient noise $\xi_t$ of variance $\sigma^2$ under time-varying step sizes $\eta_t$:

$$x_{t+1} = (1 - \eta_t \lambda)\, x_t - \eta_t\, \xi_t.$$

Over $t$ steps, the deterministic mean contracts by $\prod_{s < t}(1 - \eta_s\lambda) \approx e^{-\lambda B_t}$, where

$$B_t = \sum_{s=0}^{t-1} \eta_s$$

is the **cumulative learning-rate budget** (the area under the learning-rate curve up to step $t$). Provided that $\eta_t$ changes slowly over the local relaxation window $1/(\eta_t\lambda)$, the variance tracks the instantaneous noise ball $\eta_t\sigma^2 / (2\lambda)$, yielding the mean-squared error decomposition

$$\mathbb{E}\big[x_t^2\big] \approx e^{-2\lambda B_t}\, x_0^2 + \frac{\eta_t\, \sigma^2}{2\lambda}.$$
:eqlabel:`eq_scheduler-bias-noise`

Equation :eqref:`eq_scheduler-bias-noise` highlights the two requirements on any finite-horizon schedule over $T$ steps:
1. **Sufficient cumulative budget $B_T$:** The total budget $B_T = \sum_{t=0}^{T-1}\eta_t$ must be large enough that the initial bias $e^{-2\lambda_{\min} B_T} x_0^2$ vanishes along the flattest direction.
2. **Small terminal learning rate $\eta_T$ with sufficient remaining budget:** The terminal rate $\eta_T$ must decay toward zero slowly enough that noise injected at intermediate steps $s$ is contracted by the remaining budget $R_s = \sum_{u=s}^{T-1}\eta_u \gtrsim 1/\lambda_{\min}$.

Three classical parametric schedule families are piecewise-constant (step) decay, exponential decay, and polynomial decay:

$$
\begin{aligned}
\eta_t &= \eta_i \quad \textrm{for } t_i \leq t < t_{i+1} && \textrm{(piecewise constant)}, \\
\eta_t &= \eta_0\, \gamma^t && \textrm{(exponential)}, \\
\eta_t &= \eta_0\, (1 + t/\tau)^{-\alpha} && \textrm{(polynomial)},
\end{aligned}
$$
:eqlabel:`eq_scheduler-classical`

with decay factor $\gamma \in (0, 1)$, timescale $\tau > 0$, and exponent $\alpha > 0$. Simulating $1{,}000$ independent SGD runs on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ over $T = 3{,}000$ steps illustrates how the cumulative budget $B_T$ and terminal rate $\eta_T$ govern convergence—and why overly rapid exponential decay freezes noise:

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

The constant rate ($\eta = 0.1$) plateaus at its stationary noise ball ($0.25 + 0.016 \approx 0.27$). Polynomial decay with $\alpha = 1/2$ ($\eta_T \approx 0.036$) and $\alpha = 1$ ($\eta_T \approx 0.0033$) track :eqref:`eq_scheduler-bias-noise` down to $\sim 0.10$ and $\sim 0.011$, respectively. By contrast, exponential decay ($\gamma = 0.98$) has a terminal rate of $10^{-27}$ yet freezes at a mean-squared error of $0.13$: because its remaining budget after step $s$ is only $R_s \approx \eta_s / (1-\gamma) = 50\eta_s$, noise injected once $\eta_s \lesssim 0.1$ ($R_s \lambda_1 \lesssim 1$) is never contracted away, leaving a frozen variance of $(1-\gamma)\sigma^2 / (4\lambda_1^2) = 0.125$ (Exercise :exref:`ex_scheduler-frozen-noise`).

## Decay Shapes on a Language Model
:label:`subsec_scheduler-decay-shapes`

On a fixed step budget $T$, two decay profiles dominate modern deep learning:
* **Cosine decay** :cite:`Loshchilov.Hutter.2016`, which follows a half-cosine wave from peak rate $\eta_0$ at $t = 0$ to terminal rate $\eta_T$ at $t = T$:
  $$
  \eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2}\left(1 + \cos\frac{\pi t}{T}\right).
  $$
  :eqlabel:`eq_scheduler-cosine`
  When $\eta_T = 0$, cosine decay has cumulative budget $B_T = \frac{1}{2}\eta_0 T$ (half the budget of a constant rate $\eta_0$).
* **Stable–then–decay (late linear decay)**, which holds the peak rate $\eta_0$ constant until step $t_{\text{decay}}$ (here $0.8\,T = 1{,}600$) and then decays linearly to $0$ over the final $20\%$ of training, preserving $90\%$ of the constant-rate budget ($B_T = 0.9\,\eta_0 T$) while still finishing at $\eta_T = 0$.

We compare constant, cosine, and stable–then–decay schedules ($\eta_0 = 3 \times 10^{-3}, T = 2{,}000$) with AdamW ($\lambda = 0$) on `TinyLM` across two seeds, recording validation loss at step $1{,}500$ and step $2{,}000$:

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

At step $1{,}500$, cosine decay leads because its learning rate has already fallen to $15\%$ of the peak, damping high-frequency oscillation. However, between steps $1{,}600$ and $2{,}000$, when `stable_decay` linearly reduces $\eta_t \to 0$, its loss drops sharply by $\sim 0.10$ nats—finishing $0.05\text{--}0.09$ nats below the constant schedule and slightly ahead of cosine decay, which had only half the cumulative travel budget ($B_T = 3.0$ vs. $5.4$). Notice that comparing checkpoints before learning-rate decay completes can reverse the true end-of-training ranking :cite:`Wen.Hall.Ma.ea.2025`.

## Learning-Rate Warmup
:label:`subsec_scheduler-warmup`

While decay settles the iterate at the end of training, **learning-rate warmup** stabilizes the opening steps by ramping $\eta_t$ linearly from $0$ to $\eta_0$ over $W$ steps (typically $1\%\text{--}5\%$ of $T$).

Why is warmup needed? First, under Adam (:numref:`sec_adam`), step $1$ moves every coordinate by $\pm\eta_0$ based on a single minibatch before the momentum average $\mathbf{m}_t$ and second-moment average $\mathbf{v}_t$ have accumulated multi-batch statistics. On `TinyLM`, where initial linear weight matrices have $\text{RMS}(\mathbf{W}_0) \approx 0.045$, a cold start at $\eta_0 = 0.03$ alters every weight by $\sim 67\%$ of its initial magnitude on step $1$, corrupting the initialization and causing the run to stall near $2.8$ nats. Second, as shown in :numref:`subsec_gd-edge-of-stability`, gradually raising $\eta_t$ lowers the sharpness ceiling $2/\eta_t$ progressively, guiding the network into a flat region that can tolerate the peak learning rate :cite:`Goyal.Dollar.Girshick.ea.2017,Kalra.Barkeshli.2024`.

Adding a $20$-step linear ramp (`warmup(peak, 20)`, just $1\%$ of the $2{,}000$-step run) caps the initial relative weight change below $10\%$ and rescues training at $\eta_0 = 0.03$, lowering the final validation loss by nearly a full nat:

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

## Warmup–Stable–Decay (WSD) and Checkpoint Branching

Cosine decay :eqref:`eq_scheduler-cosine` requires fixing the total step horizon $T$ before step $1$: intermediate checkpoints are still at elevated learning rates, and extending training beyond $T$ requires restarting or splicing schedules. Combining a short warmup with a constant peak-rate plateau and a short final decay yields the **warmup–stable–decay (WSD)** schedule :cite:`Hu.Tu.Han.ea.2024,Hagele.Bakouch.Kosson.ea.2024`.

Geometrically, :citet:`Wen.Li.Wang.ea.2024` model the WSD trajectory via a **river valley** landscape (:numref:`fig_opt_river_valley`): during the high-rate plateau, the iterate oscillates across the steep transverse walls while making rapid progress down the flat longitudinal river bed; the short final decay damps the transverse oscillations (which have large curvature $\lambda_{\text{steep}}$ and therefore require only a small remaining budget $R \gtrsim 1/\lambda_{\text{steep}}$) and drops the iterate directly onto the valley floor.

![The river-valley geometry of warmup–stable–decay (WSD): the high constant plateau makes rapid progress downstream while oscillating across steep transverse walls, and a short decay branch quickly settles onto the valley floor.](../img/mdl-opt-river-valley.svg)
:label:`fig_opt_river_valley`

### Branching Off the Plateau
:label:`subsec_scheduler-branching`

Because the plateau phase does not depend on the target horizon $T$, we can maintain a single continuous plateau run and **branch** a short linear cooldown ($250$ steps) from any checkpoint (copying both model weights and AdamW moment buffers $\mathbf{m}_t, \mathbf{v}_t$) to obtain a fully cooled model for that token budget:

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

Every branched cooldown drops the validation loss sharply below the continuing plateau ($0.05\text{--}0.06$ nats at steps $1{,}250$ and $1{,}750$), and the branch at step $1{,}000 \to 1{,}250$ matches a dedicated $1{,}250$-step run planned from scratch—delivering finished models across three training horizons in $2{,}250$ total steps instead of $3{,}750$.

## Summary

* A learning-rate schedule $\eta_t$ is governed by its **cumulative learning-rate budget** $B_T = \sum_{t=0}^{T-1}\eta_t$ (how far the iterate can travel along flat directions) and its **terminal learning rate** $\eta_T$ (the final noise floor), subject to retaining sufficient post-step budget $R_s \gtrsim 1/\lambda$ to damp earlier noise.
* On `TinyLM`, decaying the learning rate to zero at the end of training eliminates the $O(\eta)$ oscillation of constant-rate AdamW, reducing validation loss by $0.05\text{--}0.09$ nats.
* **Linear warmup** over the first $1\%\text{--}5\%$ of steps prevents oversized initial Adam steps before $(\mathbf{m}_t, \mathbf{v}_t)$ stabilize and steers the trajectory below the sharpness threshold $2/\eta_0$.
* **Warmup–stable–decay (WSD)** holds $\eta_0$ constant after warmup and decays only over the final $10\%\text{--}20\%$ of steps, allowing short cooldown branches to be forked from any intermediate checkpoint.

## Exercises

1. [code] **Frozen noise under exponential decay.** Under $\eta_s = \eta_0\gamma^s$ with remaining budget $R_s \approx \eta_s/(1-\gamma)$, integrate $\sum_{s} \eta_s^2 \sigma^2 e^{-2\lambda R_s}$ to show that the asymptotic frozen variance along curvature $\lambda$ is $(1-\gamma)\sigma^2 / (4\lambda^2)$ when $2\lambda\eta_0 \gg 1-\gamma$. Verify this prediction with `sgd_valley` for $\gamma \in \{0.98, 0.99, 0.995\}$.
   :exlabel:`ex_scheduler-frozen-noise`

1. [code] **Piecewise-constant step decay.** Use :eqref:`eq_scheduler-bias-noise` to design a $3$-stage piecewise-constant schedule over $3{,}000$ steps on `sgd_valley` and compare its final mean-squared distance against $\alpha = 1$ polynomial decay ($0.011$).
   :exlabel:`ex_scheduler-step-decay`

1. [code] **Schedules with identical cumulative budget $B_T$ and terminal rate $\eta_T$.** ● Compare full-horizon linear decay $\eta_t = \eta_0(1 - t/T)$ and cosine decay :eqref:`eq_scheduler-cosine` with $\eta_T = 0$ (both of which have $B_T = \eta_0 T / 2$ and $\eta_T = 0$) on `sgd_valley` ($\eta_0 = 0.4, T = 3{,}000$), and explain why cosine decay finishes at a lower terminal distance by comparing their remaining budgets $R_t$ near $t \approx T$.
   :exlabel:`ex_scheduler-same-two-numbers`

1. [code] **Warmup duration and maximum tolerable peak rate.** ● On `TinyLM`, vary the linear warmup length across $W \in \{2, 5, 10, 20, 50\}$ steps at $\eta_0 = 0.03$, compare the minimum stable warmup against Adam's first-moment window $1/(1-\beta_1) = 10$, and find how much a $20$-step warmup increases the maximum trainable peak learning rate $\eta_0$.
   :exlabel:`ex_scheduler-warmup-length`

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
