# Adam and AdamW
:label:`sec_adam`

In :numref:`sec_test_models`, plain minibatch SGD required a learning rate of $\eta = 0.5$ on `TinyLM` to reach a loss of about $2.1$ nats/byte, whereas `AdamW` with $\eta = 3 \times 10^{-3}$—a learning rate more than a hundred times smaller—reached $1.59$ nats/byte in the same $2{,}000$ steps. Why does an optimizer with a hundredfold smaller nominal learning rate train faster, and why is it the universal default for Transformers?

Momentum (:numref:`sec_momentum`) averages gradients over time but still applies a single scalar step size to every parameter. Yet in deep networks—and especially Transformers—gradient magnitudes vary by orders of magnitude across embeddings, attention projections, MLP layers, and normalization parameters. **Adam** (*Adaptive Moment Estimation*, :citet:`Kingma.Ba.2014`) combines momentum with a per-coordinate root-mean-square (RMS) gradient preconditioner developed through **AdaGrad** :cite:`Duchi.Hazan.Singer.2011` and **RMSProp** :cite:`Tieleman.Hinton.2012`. Dividing each coordinate's update by its own historical gradient scale makes the learning rate $\eta$ a direct step length in parameter space rather than a multiplier on raw gradients.

That same per-coordinate division, however, breaks standard $\ell_2$ regularization: adding $\frac{\lambda}{2}\|\mathbf{x}\|^2$ to the loss rescales each weight's decay rate by the inverse of its gradient scale. :citet:`Loshchilov.Hutter.2019` resolved this by decoupling weight decay from the adaptive preconditioner, creating **AdamW**. In this section, we trace the progression from AdaGrad and RMSProp to Adam, measure coordinate gradient scales and compare Adam against SGD with momentum on `TinyLM`, derive AdamW and its weight-decay timescale $\tau = 1/(\eta\lambda)$ and equilibrium norm, and detail which parameter groups must be exempt from weight decay.

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

## Per-Coordinate Adaptive Learning Rates

### AdaGrad and RMSProp

In models with sparse or heterogeneous features—such as token embeddings where common tokens appear in every minibatch and rare tokens appear only occasionally—a single global learning rate cannot serve all coordinates. **AdaGrad** :cite:`Duchi.Hazan.Singer.2011` accumulates the running sum of squared gradients $\mathbf{s}_t$ elementwise and scales each coordinate's step inversely by $\sqrt{\mathbf{s}_t + \epsilon}$:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t,
$$
:eqlabel:`eq_adagrad`

where $\mathbf{s}_0 = \mathbf{0}$ and $\epsilon > 0$ prevents division by zero. On an axis-aligned valley, scaling each coordinate by its own gradient magnitude automatically cancels the directional curvature difference, acting as a gradient-only approximation to the diagonal preconditioner of :numref:`subsec_gd-preconditioning` (:numref:`subsec_mdl-per-coordinate`).

```{.python .input #adam-per-coordinate-learning-rates-1}
%%tab pytorch, jax
def adagrad_2d(x1, x2, s1, s2, f_grad, eta=2.0, eps=1e-6):
    g1, g2 = f_grad(x1, x2)
    s1, s2 = s1 + g1 ** 2, s2 + g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

d2l.show_trace_2d(d2l.f_valley, d2l.train_2d(adagrad_2d, d2l.valley_grad))
```

Because $\mathbf{s}_t$ accumulates squared gradients monotonically without bound, AdaGrad's effective step size decays as $O(t^{-1/2})$ under persistent stochastic gradients, eventually halting progress on non-convex deep networks. **RMSProp** :cite:`Tieleman.Hinton.2012` replaces the cumulative sum with an exponential moving average (EMA) $\mathbf{v}_t$ of squared gradients over a window of $\sim 1/(1-\beta_2)$ steps:

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t.
$$
:eqlabel:`eq_rmsprop`

*(Note: following standard optimization notation, $\mathbf{v}_t$ here denotes the second-moment squared-gradient average rather than the velocity vector of :numref:`sec_momentum`.)* By forgetting ancient gradients, RMSProp maintains a non-vanishing step size even from a moderate learning rate $\eta = 0.4$:

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

Notice two properties of :eqref:`eq_rmsprop`:

1. **Early step amplification without bias correction:** Because $\mathbf{v}_0 = \mathbf{0}$, the first step has $\mathbf{v}_1 = (1 - \beta_2)\mathbf{g}_1^2$, causing an oversized initial jump of magnitude $\eta/\sqrt{1-\beta_2} \approx 3.16\,\eta$ at $\beta_2 = 0.9$ (and $31.6\,\eta$ at $\beta_2 = 0.999$).
2. **Normalized step length:** Whenever a coordinate's gradient has roughly steady magnitude, $\sqrt{v_t} \approx |g_t|$ and the update has magnitude $\approx \eta$ regardless of how small $|g_t|$ becomes. Near a minimum, a constant-rate adaptive method therefore oscillates within an $O(\eta)$ neighborhood until $\eta$ is decayed by a schedule (:numref:`sec_scheduler`).

### Adam and Bias Correction

**Adam** :cite:`Kingma.Ba.2014` combines an EMA of the first moment (momentum, $\mathbf{m}_t$) with an EMA of the uncentered second moment ($\mathbf{v}_t$), and applies exact **bias correction** to both estimators. Unrolling $\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^t \beta_2^{t-s}\mathbf{g}_s^2$ from $\mathbf{v}_0 = \mathbf{0}$ shows that the weights sum to $1 - \beta_2^t$ rather than $1$. Dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ yields unbiased moment estimators $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\, \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t,
\qquad &
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, \\
\mathbf{v}_t &= \beta_2\, \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^2,
\qquad &
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t}.
\end{aligned}
$$
:eqlabel:`eq_adam-moments`

The parameter update divides the bias-corrected first moment by the coordinate RMS scale $\sqrt{\hat{\mathbf{v}}_t}$:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$
:eqlabel:`eq_adam-update`

Standard defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$ (or $\beta_2 = 0.95$ in large language model training :cite:`brown2020language`), and $\epsilon = 10^{-8}$. At $t = 1$, $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so the initial step is exactly $-\eta\,\operatorname{sign}(\mathbf{g}_1)$.

More generally, rescaling any coordinate of the objective by a positive constant $c > 0$ multiplies both $\hat{m}_{t,i}$ and $\sqrt{\hat{v}_{t,i}}$ by $c$, leaving their ratio invariant (for $\epsilon \to 0$). Thus, **under Adam, $\eta$ specifies a direct coordinate step length in parameter space**: a coordinate with consistent gradient sign moves by $\approx \eta$ per step, whereas a coordinate dominated by zero-mean noise moves by $\approx \eta\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23\,\eta$.

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

path = d2l.train_2d(adam_2d, d2l.valley_grad, steps=200, beta2=0.999)
d2l.show_trace_2d(d2l.f_valley, path)
```

Implementing Adam from scratch requires maintaining the step counter $t$ and the two buffers $(\mathbf{m}, \mathbf{v})$ for each parameter tensor:

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

## Adam on a Language Model

### Gradient Scales and Comparison with SGD on `TinyLM`
:label:`subsec_adam-where-wins`

Why is per-coordinate scaling so critical for Transformers? To measure gradient scales across `TinyLM` (:numref:`subsec_tinylm`), we first load the `WikiText103` dataset ($b = 64$, window length $64$):

```{.python .input #adam-adam-from-scratch-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

Next we define `gradients(model, data)` to compute the root-mean-square (RMS) gradient of each parameter tensor over $10$ minibatches, `snapshot(model)` to clone parameter values, and an initial `TinyLM` instance with Adam ($\eta = 3 \times 10^{-3}$):

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

Grouping tensors into the $8$ functional parameter kinds via `rms_by_kind`, we compare their RMS gradient magnitudes at step $0$ and step $500$ alongside the normalized Adam step length $\|\Delta\mathbf{x}\|_{\text{RMS}} / \eta$:

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

Raw gradient magnitudes span a **$40\times$ range** at initialization (from $\sim 2 \times 10^{-4}$ on token embeddings to $\sim 7 \times 10^{-3}$ on biases) and a **$15\times$ range** at step $500$. Under SGD, a single scalar learning rate moves the groups with the largest gradients (biases, LayerNorm, the output head) $15\text{--}40\times$ faster than the embeddings; under Adam, the normalized step size across all eight parameter groups lies within a narrow $2\times$ band ($0.1\eta\text{--}0.3\eta$, close to the pure-noise ratio $\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23$).

Because gradient scales in `TinyLM` lie around $10^{-4}\text{--}10^{-2}$ and Adam's EMA includes $(1-\beta_1) = 0.1$, Adam at $\eta \sim 3 \times 10^{-3}$ produces parameter steps comparable in magnitude to SGD with momentum $0.9$ at $\eta \sim 0.3$. To compare both optimizers across $3$ shared seeds, we first set up the model and optimizer closures `make_lm`, `make_sgd`, and `make_adam`:

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

We then sweep SGD with momentum across $\eta \in \{0.1, 0.3, 0.5, 1.0\}$ and Adam across $\eta \in \{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}\}$ from identical initial weights per seed:

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

Summarizing each optimizer at its best learning rate (`best_rate`), we compare the usable learning-rate range (within $0.15$ nats of the optimum), steps to reach $1.8$ nats, maximum single-step loss spikes, and smoothed loss trajectories:

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

On `TinyLM`, Adam reaches a target loss of $1.8$ nats in nearly half the steps of SGD with momentum ($650\text{--}800$ vs. $1{,}100\text{--}1{,}500$), finishes at a lower final loss, operates across a $10\times$ usable learning-rate window ($10^{-3}$ to $10^{-2}$), and remains finite even at $\eta = 3 \times 10^{-2}$ because its normalized update length is bounded by $\eta$. SGD's best rate, $\eta = 0.3$, sits one grid step below divergence: its worst one-step loss spikes are several times Adam's, and at $\eta = 0.5$ most runs diverge to `inf`. On larger Transformers with subword vocabularies, this gap widens further due to block-level Hessian heterogeneity :cite:`Zhang.Chen.Ding.ea.2024` and heavy-tailed token frequencies :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024`.

:begin_tab:`pytorch`
In this run, one spike sets an SGD seed back so far that it never reaches $1.8$.
:end_tab:

### Role of $\epsilon$ and Limits of the Second-Moment Estimate
:label:`subsec_adam-variance-estimate`

In :eqref:`eq_adam-update`, $\epsilon$ caps the maximum effective coordinate learning rate at $\eta / \epsilon$; in the limit $\epsilon \gg \sqrt{\hat{v}_{t,i}}$, Adam reduces to SGD with momentum at step size $(1-\beta_1)\eta/\epsilon$. Conversely, when $\beta_2$ is too small, a short-memory second-moment estimate can weight frequent small gradients more heavily than occasional large opposing gradients—a phenomenon used by :citet:`Reddi.Kale.Kumar.2019` to construct a synthetic periodic convex counterexample ($+4, -1, -1$) on which small-$\beta_2$ Adam fails to converge (:numref:`subsec_mdl-adam-counterexample`; Exercise :exref:`ex_adam-two-repairs-variance-estimate`). In practice, keeping $\beta_2 \in [0.95, 0.999]$ prevents this pathology without requiring monotonic running maxima (AMSGrad).

## Decoupled Weight Decay (AdamW)
:label:`sec_adamw`

### Coupled $\ell_2$ Regularization vs. Decoupled Weight Decay

Under plain SGD (:numref:`sec_weight_decay`), adding an $\ell_2$ penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ to the objective adds $\lambda\mathbf{x}_t$ to the gradient, which is mathematically identical to shrinking the weights by $(1 - \eta\lambda)$ before each step:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,(\mathbf{g}_t + \lambda \mathbf{x}_t)
= (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t.
$$
:eqlabel:`eq_l2-is-decay`

When the penalized gradient $\mathbf{g}_t + \lambda\mathbf{x}_t$ is fed into Adam, however, the penalty term is divided by the coordinate preconditioner $\sqrt{\hat{v}_{t,i}} + \epsilon$:

$$
\underbrace{\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}}_{\textrm{coupled $\ell_2$ penalty in Adam}}
\qquad \textrm{versus} \qquad
\underbrace{\eta\,\lambda\; x_{t,i}\vphantom{\frac{\eta}{\sqrt{\hat{v}_{t,i}}}}}_{\textrm{decoupled weight decay}}.
$$
:eqlabel:`eq_coupled-decay`

Under coupled $\ell_2$ regularization, weights with small loss gradients are decayed up to $10^3\times$ more aggressively than weights with large gradients—or, when $\lambda|x_{t,i}|$ dominates the loss gradient, Adam normalizes the penalty by its own magnitude and drives the weight toward zero at speed $\eta$ per step until $\lambda|x| \sim |g_{\text{loss}}|$.

:citet:`Loshchilov.Hutter.2019` introduced **AdamW**, which decouples weight decay from the adaptive gradient step:

$$
\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
$$
:eqlabel:`eq_adamw`

where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are computed solely from the data loss gradient $\mathbf{g}_t$.

```{.python .input #adamw-adamw-from-scratch-1}
%%tab pytorch
class AdamW(torch.optim.Adam):
    """Adam with decoupled weight decay: shrink the weights, then step."""
    def __init__(self, params, lr=1e-3, weight_decay=0.01):
        super().__init__(params, lr)  # Adam's own weight_decay stays 0
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.mul_(1 - group['lr'] * self.weight_decay)
        super().step()
```

```{.python .input #adamw-adamw-from-scratch-1}
%%tab jax
def adamw(learning_rate, weight_decay):
    return optax.chain(optax.scale_by_adam(),
                       optax.add_decayed_weights(weight_decay),
                       optax.scale(-learning_rate))
```

To compare no decay, coupled `Adam + l2`, and decoupled `AdamW` on `TinyLM`, we first define `matrix_rms` to measure the entrywise RMS of the 2D non-embedding weight matrices and `train_tracked` to record both the step losses and the matrix RMS every $250$ steps:

```{.python .input #adamw-coupled-and-decoupled-decay-compared-2}
%%tab pytorch
def matrix_rms(model):
    """RMS of the entries of the weight matrices, embeddings excluded."""
    W = torch.cat([p.detach().flatten() for name, p in model.named_parameters()
                   if p.ndim == 2 and 'emb' not in name])
    return float(W.square().mean().sqrt())

def train_tracked(make_optimizer, data, seed, num_steps, every=250):
    """Train TinyLM from seed with make_optimizer(params); return the losses,
    the matrix RMS at the start and every `every` steps, and the model."""
    torch.manual_seed(seed)
    model = d2l.TinyLM(len(data.vocab))
    optimizer = make_optimizer(model.parameters())
    losses, rms = [], [matrix_rms(model)]
    for _ in range(num_steps // every):
        losses += d2l.train_steps(model, data, optimizer, every)
        rms.append(matrix_rms(model))
    return losses, rms, model
```

```{.python .input #adamw-coupled-and-decoupled-decay-compared-2}
%%tab jax
def matrix_rms(model):
    """RMS of the entries of the weight matrices, embeddings excluded."""
    leaves = jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param))
    W = jnp.concatenate([p.ravel() for path, p in leaves
                         if p.ndim == 2 and 'emb' not in str(path)])
    return float(jnp.sqrt((W ** 2).mean()))

def train_tracked(tx, data, seed, num_steps, every=250):
    """Train TinyLM from seed with the optax transformation tx; return the
    losses, the matrix RMS at the start and every `every` steps, the model."""
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses, rms = [], [matrix_rms(model)]
    for _ in range(num_steps // every):
        losses += d2l.train_steps(model, data, optimizer, every)
        rms.append(matrix_rms(model))
    return losses, rms, model
```

Running all three configurations for $1{,}000$ steps at $\eta = 3 \times 10^{-3}$ and $\lambda = 0.1$ from a shared seed shows the dramatic difference between coupled $\ell_2$ penalty and decoupled weight decay:

```{.python .input #adamw-one-number-two-meanings}
%%tab pytorch
seed = random.getrandbits(31)
curves = {}
for name, make_optimizer in [
        ('no decay', lambda params: torch.optim.Adam(params, 3e-3)),
        ('Adam + ℓ2', lambda params: torch.optim.Adam(params, 3e-3,
                                                       weight_decay=0.1)),
        ('AdamW', lambda params: torch.optim.AdamW(params, 3e-3,
                                                    weight_decay=0.1))]:
    curves[name], rms, _ = train_tracked(make_optimizer, data, seed, 1000)
    print(f'{name:<10} final loss {d2l.final_loss(curves[name]):.3f}, RMS of '
          f'the weight matrices {rms[0]:.4f} -> {rms[-1]:.4f}')
d2l.plot(list(range(0, 1000, 25)), [d2l.smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

```{.python .input #adamw-one-number-two-meanings}
%%tab jax
def coupled(lr, wd):  # The penalty's gradient enters Adam's moments
    return optax.chain(optax.add_decayed_weights(wd), optax.adam(lr))

seed = d2l.get_seed()
curves = {}
for name, tx in [('no decay', optax.adam(3e-3)),
                 ('Adam + ℓ2', coupled(3e-3, 0.1)),
                 ('AdamW', optax.adamw(3e-3, weight_decay=0.1))]:
    curves[name], rms, _ = train_tracked(tx, data, seed, 1000)
    print(f'{name:<10} final loss {d2l.final_loss(curves[name]):.3f}, RMS of '
          f'the weight matrices {rms[0]:.4f} -> {rms[-1]:.4f}')
d2l.plot(list(range(0, 1000, 25)), [d2l.smooth(c) for c in curves.values()],
         'step', 'training loss', legend=list(curves))
```

Coupled `Adam + l2` at $\lambda = 0.1$ collapses the hidden weight matrices from an initial RMS of $0.045$ down to $< 0.003$, destroying the network's representation and stalling at the unigram byte-frequency entropy ($\sim 3.2$ nats/byte). Decoupled `AdamW` trains cleanly alongside the unregularized baseline while bounding weight growth.

### The Timescale of Decay and Equilibrium Norm
:label:`subsec_wd-at-scale`

Writing $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ for Adam's normalized update direction and unrolling :eqref:`eq_adamw` over $t$ steps yields

$$
\mathbf{x}_t = (1 - \eta\lambda)^t\, \mathbf{x}_0
- \eta \sum_{s=0}^{t-1} (1 - \eta\lambda)^{t-1-s}\, \mathbf{u}_s.
$$
:eqlabel:`eq_adamw-unrolled`

Thus, AdamW expresses the parameters $\mathbf{x}_t$ as an exponentially weighted moving average of past updates with decay factor $1 - \eta\lambda \approx e^{-\eta\lambda}$ and characteristic **decay timescale**

$$\tau = \frac{1}{\eta\lambda}$$

steps :cite:`Wang.Aitchison.2025`. At $\eta = 3 \times 10^{-3}$ and $\lambda = 0.1$, the decay timescale is $\tau = 1 / 0.0003 \approx 3{,}333$ steps. Over a $T$-step run, weight decay acts across $T / \tau = T\eta\lambda$ timescales.

Over runs longer than $\tau = 1/(\eta\lambda)$, the initial weights decay away and each coordinate $x_{t,i}$ reaches a stochastic **equilibrium norm** where per-step shrinkage $\eta\lambda x_{t,i}$ balances the injection of normalized updates $-\eta n_{t,i}$ (with $n_{t,i} = g_{t,i}/\sqrt{\hat{v}_{t,i}}$ of unit mean square $\sigma^2 \approx 1$). By the exact same variance recursion as the SGD noise ball :eqref:`eq_sgd-noise-ball` with $\lambda_{\text{decay}}$ in place of curvature:

$$
\begin{aligned}
\mathbb{E}\big[x_{t+1}^2\big] &= (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2 \\
\Longrightarrow \quad \mathbb{E}\big[x_\infty^2\big] &\approx \frac{\eta\, \sigma^2}{2\lambda} \approx \frac{\eta}{2\lambda}.
\end{aligned}
$$
:eqlabel:`eq_adamw-equilibrium`

Stabilizing the root-mean-square weight magnitude at $\text{RMS}(\mathbf{x}_\infty) \propto \sqrt{\eta / (2\lambda)}$ stabilizes the **relative angular update rate** of normalized weight matrices at $\eta / \text{RMS}(\mathbf{x}_\infty) \propto \sqrt{2\eta\lambda}$ (*rotational equilibrium*, :citet:`Kosson.Messmer.Jaggi.2024`), preventing the uncontrolled $\sqrt{t}$ random-walk growth of unregularized weights that would otherwise decay the effective angular step size as $1/\sqrt{t}$ over long pretraining runs :cite:`DAngelo.Andriushchenko.Varre.ea.2024`. When sweeping $\eta$, holding the per-step shrinkage $s = \eta\lambda$ (and thus the timescale $\tau = 1/s$) fixed decouples the learning-rate search from the decay timescale :cite:`Loshchilov.Hutter.2019,Wang.Aitchison.2025,Bergsma.Dey.Gosal.ea.2025b`.

### Parameters Exempt from Weight Decay and Optimizer State Memory
:label:`subsec_adamw-what-not-to-decay`

In standard Transformer recipes, decoupled weight decay is applied **only to 2D internal projection matrices** ($90\%$ of `TinyLM`), while two parameter groups are explicitly exempted ($\lambda = 0$):

1. **1D biases and LayerNorm/RMSNorm gain and shift vectors:** Normalization scales start near $1$ and directly control feature variance; decaying them toward zero distorts layer activations rather than regulating angular update speed.
2. **Token and positional embedding tables:** Token embedding rows receive gradients only when their token appears in a minibatch, whereas decoupled decay shrinks every row on every step, collapsing rare token representations and inflating downstream LayerNorm gradients :cite:`OLMo.2025`.

```{.python .input #adamw-what-not-to-decay-1}
%%tab pytorch
model = d2l.TinyLM(len(data.vocab))
decay = [p for name, p in model.named_parameters()
         if p.ndim == 2 and 'emb' not in name]
no_decay = [p for name, p in model.named_parameters()
            if p.ndim != 2 or 'emb' in name]
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0}], lr=3e-3)
n_decay = sum(p.numel() for p in decay)
n = n_decay + sum(p.numel() for p in no_decay)
print(f'decayed: {n_decay} of {n} parameters')
```

```{.python .input #adamw-what-not-to-decay-1}
%%tab jax
model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(0))

def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

optimizer = nnx.Optimizer(
    model, optax.adamw(3e-3, weight_decay=0.1, mask=decay_mask),
    wrt=nnx.Param)
params = nnx.state(model, nnx.Param)
n_decay = sum(int(p.size) for p, m in zip(
    jax.tree.leaves(params), jax.tree.leaves(decay_mask(params))) if m)
n = sum(int(p.size) for p in jax.tree.leaves(params))
print(f'decayed: {n_decay} of {n} parameters')
```

### Optimizer State Memory
:label:`subsec_adamw-memory`

Finally, maintaining the two per-parameter buffers $(\mathbf{m}_t, \mathbf{v}_t)$ doubles the static parameter memory footprint. In 32-bit floating point ($4$ bytes per scalar), each trainable parameter requires **$16$ bytes** of device memory: $4$ bytes for the parameter $\mathbf{x}$, $4$ bytes for the gradient $\mathbf{g}$, and $8$ bytes for the AdamW state $(\mathbf{m}, \mathbf{v})$ (:numref:`sec_memory_precision`). For a $10^9$-parameter model, the optimizer state alone consumes $8\text{ GB}$ before storing any activations. Freezing parameters (leaving them out of the optimizer) or quantizing/factoring $(\mathbf{m}, \mathbf{v})$ directly eliminates or compresses this $8$-byte-per-parameter overhead.

While AdamW normalizes each weight entry independently, 2D weight matrices act as linear operators on activation vectors. :numref:`sec_muon` asks whether we can do even better by normalizing the update of each 2D weight matrix as a whole operator rather than entry by entry.

## Summary

* **AdaGrad** scales each coordinate by the inverse root of its cumulative squared gradients, **RMSProp** replaces the cumulative sum with an EMA $\mathbf{v}_t$, and **Adam** adds first-moment momentum $\mathbf{m}_t$ and bias corrections $1 - \beta_1^t, 1 - \beta_2^t$.
* Because Adam divides $\hat{\mathbf{m}}_t$ by $\sqrt{\hat{\mathbf{v}}_t}$, its update is invariant to diagonal rescaling of the gradient, making the learning rate $\eta$ a direct step length in parameter space and equalizing update magnitudes across Transformer parameter groups whose raw gradients differ by $15\text{--}40\times$.
* **AdamW** decouples weight decay $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$ from the adaptive preconditioner, establishing a uniform decay timescale $\tau = 1/(\eta\lambda)$, an equilibrium weight RMS $\propto \sqrt{\eta/(2\lambda)}$, and a steady relative angular step $\propto \sqrt{2\eta\lambda}$.
* Only 2D internal weight matrices are decayed; 1D biases, normalization parameters, and embedding tables are exempted. Storing $(\mathbf{m}_t, \mathbf{v}_t)$ in fp32 accounts for $8$ of the $16$ bytes per trainable parameter.

## Exercises

1. [code] **Coordinate rotation vs. diagonal scale invariance.**
    1. Verify with `adam_2d` on `d2l.f_valley` that Adam is invariant to rescaling the objective by $c > 0$ (for $c \gg \epsilon$) and to changing $\lambda_2$ from $4$ to $400$.
    1. Rotate the valley by $45^\circ$ via $f(\mathbf{x}) = 0.05(x_1 + x_2)^2 + (x_1 - x_2)^2$. Run `adagrad_2d`, `rmsprop_2d`, and `adam_2d` at $\eta = 0.4$ for $2{,}000$ steps on the unrotated and rotated valleys (and with $\lambda_2 = 400$), and explain why rotating the eigenbasis away from the coordinate axes restores ill-conditioning for diagonal preconditioners.
   :exlabel:`ex_adam-rotation`

1. [code] **Constant-rate limit cycles and AMSGrad.** ●
    1. Show that if a 1D coordinate alternates as $x_t = (-1)^t a$ on $f(x) = \frac{\lambda}{2}x^2$ under Adam, its amplitude is $a = \frac{1-\beta_1}{1+\beta_1}\frac{\eta}{2}$ independent of $\lambda$.
    1. Run `adam_2d` on `f_valley` for $20{,}000$ steps and explain why the iterate approaches within $10^{-40}$ of the origin while $\hat{v}_t$ remembers the initial gradients, then gets ejected once $\hat{v}_t$ decays below the momentum stability threshold. Verify that replacing $\hat{v}_t$ with $\max_{s \le t} \hat{v}_s$ (AMSGrad) prevents the ejection.
   :exlabel:`ex_adam-constant-rate`

1. [extended] **Frequent and rare tokens and per-group learning rates.**
    1. Following :citet:`Kunstner.Yadav.Milligan.ea.2024`, partition the $256$ byte values of `WikiText103` into frequent ($\ge 0.1\%$) and rare ($< 0.1\%$) tokens, and compare validation loss on each group under tuned SGD with momentum, Adam, and sign descent with momentum ($\eta\,\operatorname{sign}(\mathbf{m}_t)$).
       :exlabel:`ex_adam-frequent-and-rare-tokens`
    1. Assign separate learning rates to the three parameter groups of `TinyLM` (embeddings, 2D matrices, 1D vectors) under SGD with momentum, and measure how much of the SGD–Adam gap is closed by group-level versus coordinate-level scaling.
       :exlabel:`ex_adam-per-group-learning-rates`

1. [code] **Adversarial periodic gradients and second-moment repairs.** On the 3-periodic sequence $f_t(x) = 4x$ ($t \equiv 1 \pmod 3$) and $f_t(x) = -x$ otherwise on $x \in [-1, 1]$ :cite:`Reddi.Kale.Kumar.2019`, run Adam with $\beta_1 = 0, \beta_2 = 1/17, \eta_t = 0.5/\sqrt{t}$ and compare it against AMSGrad and Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`.
   :exlabel:`ex_adam-two-repairs-variance-estimate`

1. **Two limits of AdamW and rotational equilibrium.**
   :exlabel:`ex_adamw-two-limits`
    1. For $x_{t+1} = (1 - \eta\lambda)x_t - \eta n_t$, derive the deterministic equilibrium $|x_\infty| = 1/\lambda$ when $n_t = -1$ is constant, the zero-mean noise equilibrium $\mathbb{E}[x_\infty^2] \approx \eta/(2\lambda)$ when $n_t$ is i.i.d. with $\mathbb{E}[n_t]=0, \mathbb{E}[n_t^2]=1$, and the mixed equilibrium $\mathbb{E}[x_\infty^2] \approx \bar{n}^2/\lambda^2 + \eta/(2\lambda)$ when $n_t = \bar{n} + \xi_t$.
    1. Simulate $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\mathbf{u}_t$ in $\mathbb{R}^{1000}$ with $\mathbf{u}_t \perp \mathbf{x}_t$ :cite:`Kosson.Messmer.Jaggi.2024` to verify that the equilibrium RMS matches $\sqrt{\eta/(2\lambda)}$ and the per-step rotation angle matches $\sqrt{2\eta\lambda}$.

1. [code] **Joint $(\eta, \lambda)$ search and parameter-group exemptions.** ●
    1. Train `TinyLM` on $8{,}000$ overlapping windows of `d2l.TimeMachine` for $800$ steps across a grid of $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ and $\lambda$, comparing parametrization by $\lambda$ versus per-step shrinkage $s = \eta\lambda$.
       :exlabel:`ex_adamw-joint-search`
    1. Compare validation perplexity on `WikiText103` at $\lambda \in \{0.5, 1.0\}$ when decaying all parameters versus decaying only 2D weight matrices (:numref:`subsec_adamw-what-not-to-decay`), and identify whether exempting embeddings or 1D vectors accounts for most of the gain.
       :exlabel:`ex_adamw-exemptions`

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1078)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1079)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.5]{.kicker}

Adaptive Learning-Rate Methods: AdaGrad, RMSProp, Adam, and AdamW<br>**per-coordinate RMS scaling · bias correction · step-length invariance · decoupled weight decay · timescale & equilibrium norm**
:::
:::

::: {.slide title="Why Transformers Need Per-Coordinate Learning Rates"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **40× cross-layer gradient disparity:** In a Transformer, raw gradient magnitudes differ by $15\text{--}40\times$ across token embeddings, attention projections, MLP layers, and LayerNorm parameters—starving embeddings under any scalar SGD step size.
- **Adaptive RMS preconditioning:** Dividing each coordinate's update by its historical RMS gradient scale turns $\eta$ into a **direct coordinate step length in parameter space** ($\eta^\star \approx 3 \times 10^{-3}$ vs. $0.5$ for SGD) and requires **decoupling weight decay** (AdamW).
:::

::: {.col .fig .big}
@fig:ch09-opt-adam-layer-scales
:::
:::
:::

::: {.slide title="AdaGrad: Cumulative Squared-Gradient Preconditioning"}
[Per-Coordinate Scaling]{.kicker}

::: {.cols .vc}
::: {.col}
**AdaGrad** (Duchi et al., 2011) accumulates the running elementwise sum of squared gradients $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ from $\mathbf{s}_0 = \mathbf{0}$ and scales each coordinate inversely by $\sqrt{\mathbf{s}_t + \epsilon}$:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

- **Rare vs. frequent features:** Infrequently updated coordinates (e.g., rare token embeddings) keep a small denominator $\sqrt{s_{t,i}}$ and take full steps whenever their feature appears.
- **Automatic axis-aligned preconditioning:** Because $|g_i| = \lambda_i |x_i|$ is proportional to curvature, dividing by $\sqrt{s_{t,i}}$ cancels the $\kappa = 20$ disparity between $x_1$ and $x_2$ on the valley!

@!adam-per-coordinate-learning-rates-1[out=1]
:::

::: {.col .fig}
@!adam-per-coordinate-learning-rates-1[out=2]
:::
:::
:::

::: {.slide title="RMSProp: Leaky Moving Averages of Squared Gradients"}
[Forgetting Ancient Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why AdaGrad stalls on deep networks:** Because $\mathbf{s}_t = \sum_{s=1}^t \mathbf{g}_s^2$ grows monotonically without bound ($\mathbf{s}_t \propto t$), AdaGrad's effective step size decays as $O(t^{-1/2})$ and prematurely halts learning.
- **RMSProp** (Tieleman & Hinton, 2012) replaces the cumulative sum with an exponential moving average (EMA) over a memory window of $\tau_2 = \frac{1}{1-\beta_2}$ steps:
  $$\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2$$
- By forgetting ancient gradients ($\beta_2 = 0.9 \implies 10$-step window), RMSProp maintains a non-vanishing step size and converges rapidly on the valley at $\eta = 0.4$.

@!adam-rmsprop-forgetting-on-purpose-1[out=1]
:::

::: {.col .fig}
@!adam-rmsprop-forgetting-on-purpose-1[out=2]
:::
:::
:::

::: {.slide title="Two Consequences of Uncorrected RMS Normalization"}
[Early Jumps & Late Oscillation]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Cold-start step amplification:** From $\mathbf{v}_0 = \mathbf{0}$, step 1 evaluates $\mathbf{v}_1 = (1-\beta_2)\mathbf{g}_1^2$, producing an oversized jump $|\Delta x_{1,i}| = \frac{\eta}{\sqrt{1-\beta_2}}$ ($3.16\,\eta$ at $\beta_2=0.9$; **$31.6\,\eta$** at $\beta_2=0.999$).
- **2. Non-vanishing late-stage steps:** Near a minimum where $|g_t|$ is locally steady, $\sqrt{v_t} \approx |g_t|$ and the update becomes $-\eta\,\operatorname{sign}(g_t)$ of fixed length $\eta$, hopping within $\pm\eta/2$ until $\eta_t \to 0$ (§{sec_scheduler}).
:::

::: {.col .fig .big}
@fig:ch09-opt-adam-cold-start-jump
:::
:::
:::

::: {.slide title="Adam: First and Second Moments with Bias Correction"}
[Adaptive Moment Estimation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Dual moving averages ($\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$):**
  $$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$
- **Bias-corrected update:**
  $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}, \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$$
- Dividing by $1 - \beta^t$ normalizes the geometric weights at every step $t \ge 1$ (defaults: $\beta_1 = 0.9, \beta_2 \in \{0.95, 0.999\}, \epsilon = 10^{-8}$).
:::

::: {.col .fig}
@fig:ch09-opt-adam-dual-moments
:::
:::
:::

::: {.slide title="Scale Invariance: η Is a Direct Coordinate Step Length"}
[Geometry of Adam]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exact step-1 magnitude:** At $t = 1$, $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ and $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$, so for $\sqrt{\hat{v}_{1,i}} \gg \epsilon$:
  $$\Delta x_{1,i} = -\eta\,\frac{g_{1,i}}{|g_{1,i}|} = -\eta\,\operatorname{sign}(g_{1,i})$$
  Every coordinate moves by **$\pm\eta$** on step 1 ($45^\circ$ diagonal on the valley!).
- **Diagonal scale invariance:** Multiplying the loss or rescaling coordinate $i$ by $c > 0$ scales $g_{t,i} \to c\,g_{t,i}$, $\hat{m}_{t,i} \to c\,\hat{m}_{t,i}$, and $\sqrt{\hat{v}_{t,i}} \to c\sqrt{\hat{v}_{t,i}}$, canceling $c$ from the ratio!
- **Step-length bounds:** Persistent gradients step by $\approx \eta$; zero-mean noise steps by $\approx \eta\sqrt{\frac{1-\beta_1}{1+\beta_1}} \approx 0.23\,\eta$.
:::

::: {.col .fig}
@!adam-adam-on-the-valley-1
:::
:::
:::

::: {.slide title="Implementing Adam from Scratch"}
[Code Implementation]{.kicker}

Each trainable parameter tensor keeps a step counter $t$ and two buffers $(\mathbf{m}, \mathbf{v})$ of its shape:

@-adam-adam-both-moments-debiased-1
:::

::: {.slide title="40× Gradient Disparity vs. 2× Adam Step Disparity"}
[Gradient Scales in TinyLM]{.kicker}

- Measuring the RMS gradient magnitude across $10$ minibatches for all $8$ functional parameter groups of `TinyLM` reveals why scalar-LR SGD struggles:
  - **At step 0:** Raw gradient RMS spans a **$40\times$ range** (from $1.7 \times 10^{-4}$ on `token_emb` to $6.8 \times 10^{-3}$ on `biases`).
  - **At step 500:** Gradient RMS still spans a **$15\times$ range**.
- **Adam's equalization:** Dividing by $\sqrt{\hat{\mathbf{v}}_t}$ compresses the actual RMS parameter update $\|\Delta\mathbf{x}\|_{\text{RMS}}/\eta$ across all $8$ groups into a tight **$2\times$ band ($0.14\text{--}0.28\,\eta$)**!

@!adam-gradient-scales-in-a-network-2
:::

::: {.slide title="Learning-Rate Sweep: Adam vs. SGD with Momentum"}
[3-Seed Grid Comparison]{.kicker}

- We sweep SGD with momentum ($\beta = 0.9$) across $\eta \in \{0.1, 0.3, 0.5, 1.0\}$ and Adam across $\eta \in \{3{\times}10^{-4}, 10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}\}$ over $3$ seeds ($2{,}000$ steps on `TinyLM`):
  - **100× nominal scale shift:** SGD's best rate is $\eta = 0.3$, whereas Adam's is $\eta = 3 \times 10^{-3}$ ($\approx 0.1 \times \frac{0.3}{\text{scale}}$).
  - **Graceful high-LR degradation:** SGD diverges to `inf` at $\eta = 0.5$ (less than $2\times$ above its optimum), whereas Adam remains finite even at $\eta = 3 \times 10^{-2}$ ($10\times$ above its optimum) because its step length is bounded by $\eta$.

@!adam-the-race-on-the-language-model-2
:::

::: {.slide title="Adam Reaches Target Loss in Half the Steps of SGD"}
[Speed, Range, and Stability]{.kicker}

::: {.cols .vc}
::: {.col}
- Comparing the $3$ runs at each optimizer's best learning rate shows three decisive advantages for Adam on `TinyLM`:
  1. **Convergence speed:** Reaches $1.8$ nats in **$650\text{--}775$ steps** vs. $1{,}125\text{--}1{,}550+$ steps for SGD+momentum.
  2. **Usable LR window (within $+0.15$ nats):** **$10\times$ range** ($10^{-3}$ to $10^{-2}$) vs. a single fragile grid point ($0.3$) for SGD.
  3. **Spike resilience:** Worst single-step loss rise is $<0.20$ nats under Adam vs. $>0.4\text{--}1.0$ nats under SGD.
:::

::: {.col .fig}
@!adam-the-race-on-the-language-model-3[out=2]
:::
:::

@!adam-the-race-on-the-language-model-3[out=1]
:::

::: {.slide title="Sign Descent, Role of ε, and Small-β₂ Pathologies"}
[Mechanics & Edge Cases]{.kicker}

::: {.cols .vc}
::: {.col}
- **Adam as smoothed sign descent:** Setting $\beta_1 = \beta_2 = 0, \epsilon = 0$ gives $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\operatorname{sign}(\mathbf{g}_t)$; adding momentum matches Adam's lead over SGD on Transformers (Kunstner et al., 2023).
- **$\epsilon$ as a maximum-LR cap:** $\frac{\eta}{\sqrt{\hat{v}_{t,i}} + \epsilon} \le \frac{\eta}{\epsilon}$; for $\epsilon \gg \sqrt{\hat{v}_{t,i}}$, Adam reduces to SGD with momentum at rate $(1-\beta_1)\eta/\epsilon$.
- **Why $\beta_2$ must not be too small (Reddi et al., 2019):** A short-memory $\beta_2$ lets rare large gradients inflate $\sqrt{v_t}$ while small opposite gradients step further; keeping $\beta_2 \in [0.95, 0.999]$ prevents this.
:::

::: {.col .fig .big}
@fig:ch09-opt-adam-sign-and-eps
:::
:::
:::

::: {.slide title="Why Standard ℓ₂ Regularization Breaks Under Adam"}
[Coupled vs. Decoupled Decay]{.kicker}

::: {.cols .vc}
::: {.col}
- **Coupled $\ell_2$ penalty inside Adam ($\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\mathbf{x}_t$):** Divides the decay penalty by each coordinate's preconditioner:
  $$\Delta x_{t,i}^{\text{decay}} = -\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}$$
- **Pathology:** Weights with small loss gradients are decayed up to $10^3\times$ more harshly than active weights—or normalize the penalty by itself to collapse $x_{t,i} \to 0$ at speed $\eta$/step!
:::

::: {.col .fig}
@fig:ch09-opt-adam-coupled-vs-decoupled
:::
:::
:::

::: {.slide title="AdamW: Decoupling Weight Decay from the Preconditioner"}
[Loshchilov & Hutter (2019)]{.kicker}

**AdamW** computes $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ purely from the data loss gradient $\mathbf{g}_t$ and applies uniform multiplicative shrinkage $(1 - \eta\lambda)$ directly to the weights:

$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

@-adamw-adamw-from-scratch-1
:::

::: {.slide title="Coupled ℓ₂ Collapses Weight Matrices; AdamW Trains"}
[1,000-Step Comparison at λ = 0.1]{.kicker}

::: {.cols .vc}
::: {.col}
- Training `TinyLM` for $1{,}000$ steps at $\eta = 3 \times 10^{-3}$ and $\lambda = 0.1$ from identical initial weights (initial matrix RMS $= 0.0450$):
  - **No decay ($\lambda = 0$):** Final loss **$1.68$ nats**; matrix RMS grows to $0.111$.
  - **Decoupled `AdamW` ($\lambda = 0.1$):** Final loss **$1.67$ nats**; matrix RMS bounded at $0.097$.
  - **Coupled `Adam + ℓ2` ($\lambda = 0.1$):** Collapses matrix RMS $40\times$ down to **$0.0029$**, erasing internal representations so the model stalls at **$3.20$ nats** (unigram byte frequency entropy $3.17$)!
:::

::: {.col .fig}
@!adamw-one-number-two-meanings[out=2]
:::
:::

@!adamw-one-number-two-meanings[out=1]
:::

::: {.slide title="Unrolling AdamW: The Decay Timescale τ = 1/(ηλ)"}
[Weights as an EMA of Updates]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unrolled EMA of normalized updates $\mathbf{u}_s = \frac{\hat{\mathbf{m}}_s}{\sqrt{\hat{\mathbf{v}}_s}+\epsilon}$ (Wang & Aitchison, 2025):**
  $$\mathbf{x}_t = (1 - \eta\lambda)^t\,\mathbf{x}_0 - \eta\sum_{s=0}^{t-1} (1 - \eta\lambda)^{t-1-s}\,\mathbf{u}_s$$
- **Weight-decay timescale:** $\tau = \frac{1}{\eta\lambda}$ steps ($3{,}333$ steps at $\eta = 3 \times 10^{-3}, \lambda = 0.1$).
- **Hyperparameter decoupling:** When sweeping $\eta$, hold per-step shrinkage $s = \eta\lambda = 1/\tau$ fixed ($\lambda = s/\eta$).
:::

::: {.col .fig}
@fig:ch09-opt-adam-decay-timescale
:::
:::
:::

::: {.slide title="Equilibrium Weight Norm and Rotational Equilibrium"}
[Steady-State Weight Dynamics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Stochastic norm equilibrium ($t \gtrsim 2\tau$):** Balancing decay $(1-\eta\lambda)x_t$ against unit-RMS steps $-\eta n_t$ gives:
  $$\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}}$$
- **Why $\lambda = 0$ slows angular learning:** Without decay, $\text{RMS}(\mathbf{x}_t) \propto \eta\sqrt{t}$ grows via random walk, shrinking the relative angular step across scale-invariant LayerNorm sublayers as $1/\sqrt{t}$.
- **Rotational equilibrium (Kosson et al., 2024):** Under AdamW, $\frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ stays constant across layers.
:::

::: {.col .fig}
@fig:ch09-opt-adam-rotational-eq
:::
:::
:::

::: {.slide title="Parameters Exempt from Weight Decay and State Memory"}
[Parameter Groups & 16-Byte Footprint]{.kicker}

::: {.cols .vc}
::: {.col}
- **Decay only 2D internal projection matrices ($90\%$ of `TinyLM`):**
  - **Exempt 1D biases & LayerNorm scales/shifts ($\lambda = 0$):** LayerNorm gains control feature variance directly around $1.0$; decaying them distorts activations.
  - **Exempt token & positional embeddings ($\lambda = 0$):** Rare token rows receive gradients only when sampled, so every-step decay collapses rare embeddings and inflates downstream LayerNorm gradients (OLMo 2, 2025).
- **16 bytes per parameter in `float32`:** Weight ($4\text{ B}$) $+$ Gradient ($4\text{ B}$) $+$ AdamW state $\mathbf{m}, \mathbf{v}$ ($8\text{ B}$) $= 16\text{ B}$ ($8\text{ GB}$ of optimizer state alone for a $1\text{B}$-parameter model).
:::

::: {.col .fig}
@-adamw-what-not-to-decay-1
:::
:::
:::

::: {.slide title="Concept Check: Bias Correction, Timescales, and VRAM"}
[Check Your Understanding]{.kicker}

Test your understanding with three concrete calculations:

1. **Missing bias correction:** If you run Adam with $\beta_1 = 0.9, \beta_2 = 0.999, \eta = 10^{-3}$ *without* bias correction, what is the magnitude of the parameter update $|\Delta x_{1,i}|$ on step $t = 1$ compared to corrected Adam?
2. **Timescale & relative step invariance:** A run uses AdamW at $\eta = 3 \times 10^{-3}$ and $\lambda = 0.5$. If you halve the learning rate to $\eta = 1.5 \times 10^{-3}$ and double weight decay to $\lambda = 1.0$, how do the decay timescale $\tau$, the equilibrium weight RMS, and the relative angular step $\eta / \text{RMS}$ change?
3. **VRAM accounting:** You train a $7$-billion-parameter Transformer with AdamW keeping master weights, gradients, and $(\mathbf{m}, \mathbf{v})$ in `float32`. How many gigabytes do these static buffers occupy before activations?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Without bias correction, $|\Delta x_{1,i}| = \eta\frac{1 - \beta_1}{\sqrt{1 - \beta_2}} = \eta\frac{0.1}{\sqrt{0.001}} \approx$ **$3.16\,\eta = 3.16 \times 10^{-3}$** ($3.16\times$ larger than corrected Adam's exact step length $\eta = 10^{-3}$).
2. Because $\eta\lambda = 1.5 \times 10^{-3}$ is invariant, both the **timescale $\tau = \frac{1}{\eta\lambda} \approx 667$ steps** and the **relative angular step $\sqrt{2\eta\lambda} \approx 0.055$** stay **unchanged**, while the equilibrium norm $\sqrt{\eta/(2\lambda)}$ **halves** (which scale-invariant LayerNorm sublayers ignore).
3. At $16\text{ bytes/parameter}$, static memory is $7 \times 10^9 \times 16\text{ B} =$ **$112\text{ GB}$**, of which **$56\text{ GB}$ ($50\%$)** is the AdamW state $(\mathbf{m}, \mathbf{v})$.
:::
:::

::: {.slide title="Summary: Adaptive Coordinate Methods and AdamW"}
[Key Takeaways]{.kicker}

| Optimizer / Concept | Mathematical Update / Invariant | Memory & Timescale | Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **AdaGrad** | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\; \Delta\mathbf{x}_t = -\frac{\eta}{\sqrt{\mathbf{s}_t+\epsilon}}\odot\mathbf{g}_t$ | $1$ buffer ($4\text{ B}$/param); step decays as $O(t^{-1/2})$ | Sparse convex features; stalls on deep non-convex networks |
| **RMSProp** | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | $1$ buffer; scale window $\tau_2 = \frac{1}{1-\beta_2}$ | Non-vanishing adaptive steps; lacks momentum & bias correction |
| **Adam** | $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\; \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\; \Delta\mathbf{x}_t = -\eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | $2$ buffers ($8\text{ B}$/param); $\tau_1 = 10, \tau_2 = 1{,}000$ | Invariant to diagonal rescaling; equalizes $40\times$ layer gradient spread |
| **AdamW** | $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | Decay timescale $\tau = \frac{1}{\eta\lambda}$ | Bypasses $\sqrt{\hat{v}_t}$ scaling; decay 2D weight matrices only |
| **Equilibrium Norm** | $\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}},\quad \frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ | Settles after $\sim 2\tau = \frac{2}{\eta\lambda}$ steps | Prevents $\sqrt{t}$ weight growth and $1/\sqrt{t}$ effective LR decay |
:::

