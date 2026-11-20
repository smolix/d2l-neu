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

Why is per-coordinate scaling so critical for Transformers? On `TinyLM` (:numref:`subsec_tinylm`), we measure the RMS gradient magnitude across $10$ minibatches for each of the $8$ functional parameter groups at initialization and after $500$ steps of Adam ($\eta = 3 \times 10^{-3}$), alongside the normalized step length $\|\Delta \mathbf{x}\|_{\text{RMS}} / \eta$:

```{.python .input #adam-adam-from-scratch-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

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

Raw gradient magnitudes span a **$40\times$ range** at initialization (from $\sim 2 \times 10^{-4}$ on token embeddings to $\sim 7 \times 10^{-3}$ on biases) and a **$15\times$ range** at step $500$. Under SGD, a single scalar learning rate moves the groups with the largest gradients (biases, LayerNorm, the output head) $15\text{--}40\times$ faster than the embeddings; under Adam, the normalized step size across all eight parameter groups lies within a narrow $2\times$ band ($0.1\eta\text{--}0.3\eta$, close to the pure-noise ratio $\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23$).

Because gradient scales in `TinyLM` lie around $10^{-4}\text{--}10^{-2}$ and Adam's EMA includes $(1-\beta_1) = 0.1$, Adam at $\eta \sim 3 \times 10^{-3}$ produces parameter steps comparable in magnitude to SGD with momentum $0.9$ at $\eta \sim 0.3$. Sweeping both optimizers across log-spaced learning-rate grids over $3$ seeds confirms this scale shift and demonstrates Adam's decisive advantage in usable learning-rate range and stability:

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

On `TinyLM`, Adam reaches a target loss of $1.8$ nats in nearly half the steps of SGD with momentum ($650\text{--}800$ vs. $1{,}100\text{--}1{,}500$), finishes at a lower final loss, operates across a $10\times$ usable learning-rate window ($10^{-3}$ to $10^{-2}$), and remains finite even at $\eta = 3 \times 10^{-2}$ because its normalized update length is bounded by $\eta$. SGD's best rate, $\eta = 0.3$, sits one grid step below divergence: its worst one-step loss spikes are several times Adam's, one spike can set a run back so far that it never reaches $1.8$, and at $\eta = 0.5$ most runs diverge to `inf`. On larger Transformers with subword vocabularies, this gap widens further due to block-level Hessian heterogeneity :cite:`Zhang.Chen.Ding.ea.2024` and heavy-tailed token frequencies :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024`.

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

Comparing no decay, coupled `Adam + l2`, and decoupled `AdamW` at $\eta = 3 \times 10^{-3}$ and $\lambda = 0.1$ on `TinyLM` over $1{,}000$ steps shows the dramatic difference between the two formulations:

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
\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2
\qquad \Longrightarrow \qquad
\mathbb{E}\big[x_\infty^2\big] \approx \frac{\eta\, \sigma^2}{2\lambda} \approx \frac{\eta}{2\lambda}.
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

Finally, maintaining the two per-parameter buffers $(\mathbf{m}_t, \mathbf{v}_t)$ doubles the static parameter memory footprint. In 32-bit floating point ($4$ bytes per scalar), each trainable parameter requires **$16$ bytes** of device memory: $4$ bytes for the parameter $\mathbf{x}$, $4$ bytes for the gradient $\mathbf{g}$, and $8$ bytes for the AdamW state $(\mathbf{m}, \mathbf{v})$ (:numref:`sec_memory_precision`). For a $10^9$-parameter model, the optimizer state alone consumes $8\text{ GB}$ before storing any activations. Freezing parameters (`requires_grad=False`) or quantizing/factoring $(\mathbf{m}, \mathbf{v})$ directly eliminates or compresses this $8$-byte-per-parameter overhead.

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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.7]{.kicker}

AdamW<br>**decoupled weight decay · the timescale of decay · exempt parameters · optimizer state**
:::
:::

::: {.slide title="One Coefficient, λ = 0.1, Trains One Run and Flattens Another"}
[The problem]{.kicker}

`TinyLM` (0.47 million parameters, predicts the next byte of Wikipedia
text), 1,000 steps of Adam at learning rate 0.003:

| weight decay | training loss, nats per byte |
|:--|--:|
| none | about 1.7 |
| 0.1, applied to the weights (AdamW) | about 1.7 |
| 0.1, as an $\ell_2$ penalty on the loss | about 3.2 |
| *a model that knows only how frequent each byte is* | *3.17* |

::: {.d2l-note .warn}
Under Adam, "weight decay 0.1" has two meanings, more than a nat apart.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive what an $\ell_2$ penalty does under Adam;
- write AdamW's update and say what it changes;
- compute the timescale of a recipe and tell whether decay acts in a run;
- predict the size at which decayed weights settle;
- choose which parameters to decay.

**Idea:** decay must bypass Adam's per-coordinate scaling; then
$1/(\eta\lambda)$ steps is its timescale ($\eta$: learning rate,
$\lambda$: decay coefficient).

**Running example:** `TinyLM` at $\eta = 0.003$, $\lambda = 0.1$:
timescale 3,333 steps.
:::

::: {.slide title="AdamW Completes the Optimizer of the Modern Recipe"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · **AdamW** · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |

- **Uses:** Adam's update, leaky averages with window $1/(1-\beta)$, the
  noise ball of constant-step SGD.
- **Skips:** other regularizers, learning-rate schedules, how to store the
  optimizer state in fewer bytes.
:::

::: {.slide title="Under SGD, an ℓ2 Penalty Is Weight Decay"}
[Penalty and preconditioner]{.kicker}

The penalty $\frac{\lambda}{2}\|\mathbf{x}\|^2$ adds
$\lambda\mathbf{x}_t$ to the gradient $\mathbf{g}_t$:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta(\mathbf{g}_t + \lambda\mathbf{x}_t) = (1-\eta\lambda)\,\mathbf{x}_t - \eta\,\mathbf{g}_t$$

- $\mathbf{x}$: weights; $\eta$: learning rate; $\lambda$: decay
  coefficient.
- Every weight shrinks by the same factor $1 - \eta\lambda$ per step.
- $\eta = 0.003$, $\lambda = 0.1$: each step removes 0.03% of a weight.
:::

::: {.slide title="Adam Divides the Penalty by Each Weight's Gradient Scale"}
[Penalty and preconditioner]{.kicker}

Adam:
$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$;
$\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$: running averages of the
gradient and its square. Shrinkage of weight $i$ per step:

$$\underbrace{\frac{\eta\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\, x_{t,i}}_{\textrm{$\ell_2$ through Adam}}
\qquad\textrm{vs.}\qquad
\underbrace{\eta\lambda\, x_{t,i}}_{\textrm{weight decay}}$$

::: {.d2l-note}
Scale $\sqrt{\hat{v}} = 10^{-3}$, if it stayed fixed: the penalty removes
$0.0003/10^{-3} = 30\%$ per step, weight decay 0.03%.
:::
:::

::: {.slide title="AdamW Shrinks the Weights outside the Scaling"}
[Decoupled weight decay]{.kicker}

$$\mathbf{x}_{t+1} = (1 - \eta\lambda)\,\mathbf{x}_t
- \eta\, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},
\qquad \hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \textrm{ from } \mathbf{g}_t \textrm{ alone}$$

- Coupled: add $\lambda\mathbf{x}_t$ to the gradient, then update the
  averages.
- Decoupled: shrink the weights, then take the plain Adam step.

Adam plus this one shrinkage, against the library's AdamW, on two copies
of a linear model, 100 steps at $\lambda = 0.1$:

@!adamw-adamw-from-scratch-2

[Loshchilov and Hutter, 2019]{.small}
:::

::: {.slide title="Check: What Does the Penalty Do to a Weight of 0.045?"}
[Check]{.kicker}

The weight matrices of `TinyLM` start with entries of root mean square
(RMS) 0.045 and loss gradients of about $10^{-3}$. Adam with an $\ell_2$ penalty,
$\eta = 0.003$, $\lambda = 0.1$. What happens in the first steps?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The penalty's gradient $\lambda x = 0.0045$ exceeds the loss
gradient, so the scale is mostly penalty and the fixed-scale 30% is never
reached. Adam normalizes the penalty: the weight moves toward zero by
about $\eta$ per step, $0.045/0.003 = 15$ steps. $\lambda$ sets where the
shrinkage stops, not its speed.
:::
:::
:::

::: {.slide title="At λ = 0.1 the Penalty Costs More Than a Nat; AdamW Tracks No Decay"}
[Coupled and decoupled decay]{.kicker}

Three runs of `TinyLM` from one initialization, 1,000 steps:

@!adamw-one-number-two-meanings

Entropy of the byte frequencies: 3.17 nats.
:::

::: {.slide title="The Penalty Erases the Matrices; What Is Left Predicts Byte Frequencies"}
[Coupled and decoupled decay]{.kicker}

| run | RMS of the weight matrices, step 0 → 1,000 | final loss |
|:--|:--|--:|
| no decay | 0.045 → 0.11 | about 1.7 |
| AdamW, $\lambda = 0.1$ | 0.045 → 0.10 | about 1.7 |
| Adam + $\ell_2$, $\lambda = 0.1$ | 0.045 → 0.003 | about 3.2 |

- 0.003 against 0.11: forty times smaller than without decay. Almost no
  input reaches the output, and the best single prediction is the byte
  frequencies (3.17 nats).
- The shrinkage ended where $\lambda|x| \approx 3 \times 10^{-4}$, the
  size of a loss gradient.
:::

::: {.slide title="AdamW's Weights Are a Leaky Average of Their Own Steps"}
[The timescale of decay]{.kicker}

With $\mathbf{u}_t = \hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$,
Adam's step before the learning rate:

$$\mathbf{x}_t = (1-\eta\lambda)^t\,\mathbf{x}_0 - \eta\sum_{s=0}^{t-1}(1-\eta\lambda)^{t-1-s}\,\mathbf{u}_s$$

- A step from $k$ steps ago counts with
  $(1-\eta\lambda)^k \approx e^{-\eta\lambda k}$.
- Window of the average: the **timescale** $1/(\eta\lambda)$ steps.

::: {.d2l-note}
$\eta = 0.003$, $\lambda = 0.1$: 3,333 steps. A 1,000-step run is 0.3
timescales, $e^{-0.3} = 0.74$: decay barely acts.
:::

[Wang and Aitchison, 2025]{.small}
:::

::: {.slide title="Decay and Gradients Balance at an RMS of √(η/2λ)"}
[The equilibrium norm]{.kicker}

One entry $x$ of a weight matrix. Its normalized gradient
$n_t = g_t/\sqrt{\hat{v}_t}$ has mean 0 and mean square $\sigma^2$, and is
independent of $x$ and from step to step.

I. $\;x_{t+1} \approx (1-\eta\lambda)\,x_t - \eta\,n_t$ (momentum only
delivers $n_t$ over about ten steps)

II. $\;\mathbb{E}[x_{t+1}^2] = (1-\eta\lambda)^2\,\mathbb{E}[x_t^2] + \eta^2\sigma^2$

III. Fixed point:

$$\boxed{\mathbb{E}[x_\infty^2] \approx \frac{\eta\sigma^2}{2\lambda}}$$

$\sigma^2 \approx 1$: Adam divides each gradient by its RMS.
:::

::: {.slide title="Without Decay the Weights Spread like a Random Walk"}
[The equilibrium norm]{.kicker}

$\lambda = 0$: nothing cancels the growth. Every gradient adds $\eta^2$ to
the mean square of an entry:

$$\textrm{RMS}_t = \sqrt{\textrm{RMS}_0^2 + \eta^2\, t}$$

`TinyLM`: the weight matrices start at RMS 0.045; at $\eta = 0.003$,

$$\sqrt{0.045^2 + 0.003^2 \cdot 2{,}000} = 0.14 \quad\textrm{after 2,000 steps.}$$

::: {.d2l-note}
Steps at a right angle to the weights lengthen them, by Pythagoras.
:::
:::

::: {.slide title="Check: Which Runs Settle within 2,000 Steps?"}
[Check]{.kicker}

AdamW on `TinyLM`, $\eta = 0.003$, 2,000 steps, $\lambda$ = 0.1, 0.5
and 1. Which runs reach their equilibrium, and at which RMS of the weight
matrices?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Timescales $1/(\eta\lambda)$: 3,333, 667 and 333 steps. The
first run is shorter than its timescale and cannot settle (its level
would be 0.122). The others settle, at $\sqrt{\eta/(2\lambda)} = 0.055$
and $0.039$.
:::
:::
:::

::: {.slide title="Measured: the Weights Settle after Two Timescales"}
[The equilibrium norm]{.kicker}

Two runs per $\lambda$; RMS of the weight matrices every 250 steps:

@!adamw-the-timescale-of-decay-1
:::

::: {.slide title="The Formula Has the Ratio Right and the Level 1.4 Times Too Low"}
[The equilibrium norm]{.kicker}

| $\lambda$ | RMS at step 2,000 | predicted | state | validation loss |
|--:|--:|--:|:--|:--|
| 0 | 0.14 | 0.14 | growing | reference |
| 0.1 | 0.116 | 0.122 | growing, 0.6 timescales | within 0.01 |
| 0.5 | 0.076 | 0.055 | level from step 1,500 | 0.1 worse |
| 1 | 0.054 | 0.039 | level from step 750 | 0.25 worse |

- $0.076/0.054 = \sqrt{2}$, as predicted. Both levels are 1.4 times too
  high; the $\lambda = 0$ row fits, so the excess comes with the decay:
  the loss gradient pushes back outward.

::: {.d2l-note .warn}
In 2,000 steps, slow decay does nothing and fast decay does harm (every
parameter decayed).
:::
:::

::: {.slide title="Without Decay a Run Lowers Its Own Learning Rate"}
[Choice of the coefficient]{.kicker}

- A matrix $\mathbf{W}$ whose output is normalized matters only through
  its direction:
  $\textrm{LayerNorm}(c\,\mathbf{W}\mathbf{h}) = \textrm{LayerNorm}(\mathbf{W}\mathbf{h})$
  for $c > 0$.
- **Relative step** $\eta/\textrm{RMS}$: how far one gradient moves the
  entries, relative to their size.
- No decay: $0.003/0.045 = 0.067$ falls to $0.003/0.14 = 0.021$ in 2,000
  steps, then on as $1/\sqrt{t}$.

::: {.d2l-note .rule}
A threefold cut of the learning rate, on a schedule nobody chose.
:::

[For `TinyLM` a model: none of its matrices is normalized directly]{.small}
:::

::: {.slide title="Decay Holds the Relative Step at √(2ηλ), the Same for Every Matrix"}
[Choice of the coefficient]{.kicker}

$$\frac{\eta}{\sqrt{\eta/(2\lambda)}} = \sqrt{2\eta\lambda}$$

- No initial size and no gradient scale in it: a *rotational
  equilibrium*, every matrix turns at the same rate.
- One Adam step turns a matrix by about a fifth of it (momentum).
- `TinyLM`, matrices pooled: 0.039 and 0.056 at $\lambda = 0.5$ and 1,
  predicted 0.055 and 0.077.

::: {.d2l-note .rule}
In long single-pass runs, decay lowers the training loss itself.
:::

[Kosson et al., 2024; D'Angelo et al., 2024]{.small}
:::

::: {.slide title="Check: Halve η and Double λ. What Changes?"}
[Check]{.kicker}

A run with AdamW at $(\eta, \lambda) = (0.003, 0.5)$ is repeated at
$(0.0015, 1)$. What happens to the timescale, to the RMS at which the
weights settle, and to the relative step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta\lambda$ is unchanged: the same timescale of 667 steps
and the same relative step $\sqrt{2\eta\lambda}$. The RMS
$\sqrt{\eta/(2\lambda)}$ halves, which a normalized layer ignores. So when
$\eta$ changes, hold $\eta\lambda$.
:::
:::
:::

::: {.slide title="Choose λ by Its Timescale"}
[Choice of the coefficient]{.kicker}

| rule | example |
|:--|:--|
| count timescales, $T\eta\lambda$ for $T$ steps; weights settle after about 2 | 1,000 steps at $\eta\lambda = 3 \times 10^{-4}$: 0.3, inert |
| | 300,000 steps at $\eta = 3 \times 10^{-4}$, $\lambda = 0.1$: 9, active |
| searching over $\eta$: hold $\eta\lambda$ | set $\lambda = s/\eta$, search $s$ separately |
| the best timescale is a fraction of the run | 10 times longer run: timescale 3 to 10 times longer, smaller $\lambda$ |

[Loshchilov and Hutter, 2019; Wang and Aitchison, 2025; Bergsma et al., 2025]{.small}
:::

::: {.slide title="Only the Weight Matrices Are Decayed"}
[Exempt parameters]{.kicker}

| kind in `TinyLM` | parameters | decayed | reason |
|:--|--:|:--|:--|
| weight matrices | 425,984 | yes | under normalization, size sets only the relative step |
| biases, LayerNorm scales and shifts | 3,840 | no | set scale and offset of each feature directly |
| token embeddings | 32,768 | no | a row gets a gradient only when its token occurs |
| position embeddings | 8,192 | no | by custom, with the tokens |

@!adamw-what-not-to-decay-1

[OLMo 2, 2025: decayed embeddings shrank, and the gradients through the LayerNorm after them grew]{.small}
:::

::: {.slide title="Adam's Two Averages Double the Memory of Weights and Gradients"}
[Optimizer state]{.kicker}

| per parameter, 32-bit floats | bytes |
|:--|--:|
| weight | 4 |
| gradient | 4 |
| Adam's averages $m$ and $v$ | 8 |
| **total** | **16** |

- `TinyLM`: under 4 MB for $m$ and $v$. A billion parameters: 8 GB of
  16 GB.
- Only trained parameters carry the state.
- A checkpoint needs $m$ and $v$ to resume.
:::

::: {.slide title="Check: Does a 1B-Parameter Model Fit a 24 GB GPU?"}
[Check]{.kicker}

A model with one billion parameters is trained with AdamW in 32-bit
floats. Do the weights, the gradients and the optimizer state fit on a
GPU with 24 GB?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16 \times 10^9$ bytes = 16 GB: yes, with 8 GB left for the
activations. Half of the 16 GB is $m$ and $v$.
:::
:::
:::

::: {.slide title="Once Decay Bypasses the Scaling, λ Means a Timescale"}
[Wrap-up]{.kicker}

- $\ell_2$ through Adam at $\lambda = 0.1$: weight matrices from RMS 0.045
  to 0.003, loss 3.2, the byte frequencies. AdamW: the loss of no decay.
- AdamW shrinks by $1 - \eta\lambda$ per step: timescale
  $1/(\eta\lambda)$, 3,333 steps at $\eta = 0.003$, $\lambda = 0.1$.
- Settled weights: RMS 1.4 times $\sqrt{\eta/(2\lambda)}$ on `TinyLM`;
  relative step $\eta/\textrm{RMS}$ constant, in proportion to
  $\sqrt{2\eta\lambda}$.
- Decay the matrices only, 90% of `TinyLM`; $m$ and $v$ are 8 of 16 bytes
  per parameter.

**Next:** Muon, a step chosen for a weight matrix as a whole.
:::
