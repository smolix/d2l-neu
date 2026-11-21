# Practice
:label:`sec_practice`

In our small experiments so far, training took a few seconds per run, allowing us to sweep grids of learning rates and repeat runs whenever an update diverged. Production training runs are different: when a single training run takes days or weeks, most settings must be fixed in advance, a single unguarded gradient spike can corrupt hours of progress, and the final checkpoint must not depend on the luck of the last minibatch draw.

Even at extreme scale—such as the $540$-billion-parameter PaLM model trained across $6{,}144$ TPU chips, which experienced roughly $20$ sudden loss spikes despite active gradient clipping :cite:`chowdhery2022palm`—reliable training rests on three operational pillars:

1. **Guards against transient instabilities:** Global-norm **gradient clipping** (:numref:`sec_rnn-scratch`) caps anomalous steps under SGD and protects AdamW's second-moment accumulator $\mathbf{v}_t$ from transient gradient spikes, paired with checkpoint-and-rewind recovery.
2. **Iterate smoothing:** Maintaining an **exponential moving average (EMA)** of model weights eliminates high-frequency noise-ball fluctuations without requiring a full learning-rate decay to zero.
3. **Systematic calibration from short proxy runs:** Selecting the batch size from device throughput and $b_{\text{noise}}$ (:numref:`sec_batch_size`), adopting battle-tested optimizer defaults, and calibrating the peak learning rate on a short run ($10\%$ of the horizon) before executing the full warmup-and-decay schedule (:numref:`sec_scheduler`).

In this section, we examine each of these three pillars and assemble them into a complete end-to-end configuration for an untuned $3.3$-million-parameter, $4$-block `TinyLM`.

```{.python .input #practice}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #practice}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #practice-1}
%%tab pytorch
import random
```

## Gradient Clipping and Spike Recovery

As introduced for recurrent networks in :numref:`sec_rnn-scratch`, global-norm **gradient clipping** treats the concatenated gradient across all parameters as a single vector $\mathbf{g}$ and rescales it whenever its Euclidean norm $\|\mathbf{g}\|_2$ exceeds a threshold $\theta$:

$$
\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}.
$$
:eqlabel:`eq_practice_clip`

Because the scaling factor $\theta / \|\mathbf{g}\|_2$ is global, clipping preserves the relative direction across all layers while capping the Euclidean norm of the gradient entering the optimizer at $\theta$.

```{.python .input #practice-gradient-clipping-1}
%%tab pytorch
class Clipped:
    """Clip the global gradient norm before every optimizer step."""
    def __init__(self, optimizer, params, max_norm=1.0):
        self.optimizer, self.params = optimizer, list(params)
        self.max_norm, self.norms = max_norm, []

    def step(self):
        self.norms.append(nn.utils.clip_grad_norm_(self.params, self.max_norm))
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
```

```{.python .input #practice-gradient-clipping-1}
%%tab jax
def record_norms(num_steps):
    """Pass the gradients through unchanged, storing each step's norm."""
    def init(params):
        return dict(norms=jnp.zeros(num_steps), step=jnp.zeros((), jnp.int32))

    def update(grads, state, params=None):
        norm = optax.global_norm(grads)
        norms = state['norms'].at[state['step']].set(norm)
        return grads, dict(norms=norms, step=state['step'] + 1)

    return optax.GradientTransformation(init, update)

def clipped(tx, num_steps, max_norm=1.0):
    """Clip the global gradient norm before an optimizer's update, recording
    the norm before clipping on each of the first num_steps steps."""
    return optax.chain(record_norms(num_steps),
                       optax.clip_by_global_norm(max_norm), tx)
```

### Clipping as a Guard vs. a Constant Brake
:label:`subsec_practice-nan`

In :numref:`sec_adam`, SGD with momentum $0.9$ on `TinyLM` diverged to `inf` at $\eta = 1.0$ within the first $20$ steps. Is $\eta = 1.0$ too large for the entire loss surface, or only during the initial transient steps where freshly initialized weights produce large gradients? We test $(\eta, \theta) \in \{(1.0, \infty), (1.0, 1.0), (0.3, \infty), (3.0, 1.0)\}$ across two seeds, logging how many steps trigger clipping ($\|\mathbf{g}_t\|_2 > \theta$), the last step on which clipping fires, and the median gradient norm:

```{.python .input #practice-a-nan-averted-1}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
```

```{.python .input #practice-preventing-a-numerical-failure-1}
%%tab pytorch
new_seed = lambda: random.getrandbits(31)

def run(seed, lr, max_norm, num_steps=2000):
    """SGD with momentum 0.9 on TinyLM, clipped at max_norm (math.inf: not
    at all); return the losses and the gradient norms before clipping."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    model = d2l.TinyLM(len(data.vocab))
    optimizer = Clipped(torch.optim.SGD(model.parameters(), lr, momentum=0.9),
                        model.parameters(), max_norm)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    return losses, torch.stack(optimizer.norms).cpu().numpy()
```

```{.python .input #practice-preventing-a-numerical-failure-1}
%%tab jax
new_seed = d2l.get_seed

def run(seed, lr, max_norm, num_steps=2000):
    """SGD with momentum 0.9 on TinyLM, clipped at max_norm (math.inf: not
    at all); return the losses and the gradient norms before clipping."""
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    tx = clipped(optax.sgd(lr, momentum=0.9), num_steps, max_norm)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    return losses, np.asarray(optimizer.opt_state[0]['norms'][...])
```

```{.python .input #practice-a-nan-averted-2}
%%tab pytorch, jax
settings = [(1.0, math.inf), (1.0, 1.0), (0.3, math.inf), (3.0, 1.0)]
seeds, runs = [new_seed() for _ in range(2)], {}
print(f'{"lr":>4}{"clip at":>9}{"final loss":>12}{"steps clipped":>15}'
      f'{"last":>6}{"median norm":>13}{"largest / median":>18}')
for seed in seeds:
    for lr, max_norm in settings:
        losses, norms = runs[seed, lr, max_norm] = run(seed, lr, max_norm)
        line = f'{lr:4g}{max_norm:9g}{d2l.final_loss(losses):12.3f}'
        if max_norm < math.inf:
            fired = np.nonzero(norms > max_norm)[0]  # Steps clipping changed
            median = np.median(norms)
            line += (f'{len(fired):15d}{fired.max(initial=0):6d}{median:13.2f}'
                     f'{norms[100:].max() / median:18.1f}')  # After step 100
        print(line)
```

At $\eta = 1.0$, clipping at $\theta = 1.0$ fires on only the first few dozen steps (all before step $50$) and never fires again over the remaining $1{,}950$ steps (where the median gradient norm is $\sim 0.25$), rescuing the run from `inf` down to $\sim 1.61\text{--}1.64$ nats. Here clipping acts as a **guard**: it intervenes on $<2\%$ of steps to prevent early transient blow-up (and handle heavy-tailed gradient noise :cite:`Zhang.Karimireddy.Veit.ea.2020`). By contrast, at $\eta = 3.0$, clipping fires on every step for as long as the gradient norm stays finite, and the run still fails to train (its loss stalls at $100\text{--}200$ nats or overflows to `inf`): when clipping is permanently active as a **brake**, the learning rate $\eta$ is simply too large :cite:`Godbole.Dahl.Gilmer.ea.2023`. In practice, set $\theta$ roughly $3\text{--}5\times$ above the steady-state median gradient norm (typically $\theta = 1.0$) and monitor the clipping fraction.

### Clipping Under AdamW and Operational Spike Recovery
:label:`subsec_practice-spikes`

Under AdamW (:numref:`sec_adam`), a single anomalous gradient $g \gg \sqrt{v_{t-1}}$ already produces a bounded step of at most $\eta(1-\beta_1)/\sqrt{1-\beta_2}$ ($\approx 3.2\,\eta$ at $\beta_2 = 0.999$, or $0.45\,\eta$ at $\beta_2 = 0.95$). Instead of bounding the immediate step, **clipping under AdamW protects the second-moment state $\mathbf{v}_t$**: an unclipped $100\times$ gradient spike injects $(1-\beta_2) \times 10^4$ times the normal squared gradient into $v_t$, inflating the denominator $\sqrt{\hat{v}_t}$ and paralyzing learning for the next $\sim 1/(1-\beta_2)$ steps ($1{,}000$ steps at $\beta_2 = 0.999$; $20$ steps at $\beta_2 = 0.95$).

When a sudden loss spike occurs despite gradient clipping (and auxiliary logit regularizers such as PaLM's $z$-loss $10^{-4}\log^2(\sum_j e^{o_j})$; Exercise :exref:`ex_practice-logit-drift`), the standard operational remedy :cite:`chowdhery2022palm,zhang2022opt` is to **rewind to a checkpoint saved $\sim 100$ steps prior to the spike and skip the $200\text{--}500$ minibatches surrounding the trigger**, which requires checkpointing the model weights, optimizer moments $(\mathbf{m}_t, \mathbf{v}_t)$, scheduler step counter, and data-stream cursor at regular intervals.

## Weight Averaging (EMA)
:label:`subsec_practice-weight-averaging`

At a constant or non-zero learning rate, stochastic iterates fluctuate continuously around the optimum inside the noise ball (:numref:`sec_sgd`), where a mode of curvature $\lambda_{\text{curv}}$ has autocorrelation time $\sim 1/(\eta\lambda_{\text{curv}})$ steps. Maintaining an **exponential moving average (EMA)** $\bar{\mathbf{x}}_t$ of the live parameters $\mathbf{x}_t$ with decay coefficient $\alpha \in [0.99, 0.9999]$ :cite:`Polyak.Juditsky.1992`,

$$
\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1 - \alpha)\, \mathbf{x}_t,
$$
:eqlabel:`eq_practice_ema`

averages over a trailing window of $1/(1-\alpha)$ steps to cancel noise-ball fluctuations at evaluation time without altering the live training trajectory.

We compare constant-rate SGD ($\eta = 0.3$) with and without weight EMA ($\alpha = 0.999$, window $1{,}000$ steps $\approx 4.2$ epochs, started after epoch $5$ to avoid initial lag) against cosine learning-rate decay over $15$ epochs ($3{,}525$ steps) on `FashionCNN`:

```{.python .input #practice-weight-averaging-1}
%%tab pytorch, jax
fashion = d2l.FashionMNIST(batch_size=256)
num_epochs, num_batches = 15, len(fashion.train_dataloader())
test_acc = lambda model: d2l.evaluate_accuracy(model, fashion.val_dataloader())
```

```{.python .input #practice-weight-averaging-4}
%%tab pytorch
def warmup_cosine(peak, warmup, num_steps):
    """A linear warmup to peak over warmup steps, then a cosine to zero."""
    def schedule(t):
        if t < warmup:
            return peak * t / warmup
        return peak * (1 + math.cos(math.pi * (t - warmup)
                                    / (num_steps - warmup))) / 2
    return schedule
```

```{.python .input #practice-weight-averaging-4}
%%tab jax
warmup_cosine = lambda peak, warmup, num_steps: (
    optax.warmup_cosine_decay_schedule(0.0, peak, warmup, num_steps))

def ema_of_weights(alpha=0.999, start=0):
    """Pass the updates through unchanged, keeping an EMA of the updated
    weights that averages from step start on (copies them before)."""
    def init(params):
        return dict(ema=params, step=jnp.zeros((), jnp.int32))

    def update(updates, state, params):
        new = optax.apply_updates(params, updates)
        rate = jnp.where(state['step'] < start, 1.0, 1 - alpha)
        ema = optax.incremental_update(new, state['ema'], rate)
        return updates, dict(ema=ema, step=state['step'] + 1)

    return optax.GradientTransformation(init, update)

def averaged(model, optimizer):
    """A copy of model with the EMA kept by the last transformation."""
    ema = nnx.clone(model)
    nnx.update(ema, optimizer.opt_state[-1]['ema'])
    return ema
```

```{.python .input #practice-averaging-at-a-constant-learning-rate-1}
%%tab pytorch
def train_fashion(seed, schedule, alpha=0.999, start=5):
    """Plain SGD on FashionCNN at the rate schedule(t); the test accuracy
    after every epoch of the weights and of their EMA from epoch start on."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    model = d2l.FashionCNN().to(d2l.try_gpu())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # Times schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.register_step_post_hook(lambda *_: scheduler.step())
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(alpha))
    live, avg = [], []
    for epoch in range(num_epochs):
        if epoch == start:  # Its first update copies the current weights
            optimizer.register_step_post_hook(
                lambda *_: ema.update_parameters(model))
        d2l.train_steps(model, fashion, optimizer, num_batches)
        live.append(test_acc(model))
        avg.append(test_acc(ema) if epoch >= start else math.nan)
    return live, avg
```

```{.python .input #practice-averaging-at-a-constant-learning-rate-1}
%%tab jax
def train_fashion(seed, schedule, alpha=0.999, start=5):
    """Plain SGD on FashionCNN at the rate schedule(t); the test accuracy
    after every epoch of the weights and of their EMA from epoch start on."""
    model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
    tx = optax.chain(optax.sgd(schedule),
                     ema_of_weights(alpha, start * num_batches))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    live, avg = [], []
    for epoch in range(num_epochs):
        d2l.train_steps(model, fashion, optimizer, num_batches)
        live.append(test_acc(model))
        avg.append(test_acc(averaged(model, optimizer)) if epoch >= start
                   else math.nan)
    return live, avg
```

```{.python .input #practice-weight-averaging-2}
%%tab pytorch, jax
schedules = {'constant rate': lambda t: 0.3,
             'cosine decay': warmup_cosine(0.3, 0, num_epochs * num_batches)}
cnn_seeds, curves = [new_seed() for _ in range(2)], {}
print(f'{"test accuracy":<15}{"live, last five epochs":>24}{"live, final":>13}'
      f'{"EMA, final":>12}')
for seed in cnn_seeds:
    for name, schedule in schedules.items():
        live, avg = curves[seed, name] = train_fashion(seed, schedule)
        print(f'{name:<15}{min(live[-5:]):15.3f} to {max(live[-5:]):.3f}'
              f'{live[-1]:13.3f}{avg[-1]:12.3f}')
```

```{.python .input #practice-weight-averaging-3}
%%tab pytorch, jax
(live, avg), (decayed, _) = (curves[cnn_seeds[0], name] for name in schedules)
d2l.plot(list(range(1, num_epochs + 1)), [live, avg, decayed], 'epoch',
         'test accuracy', ylim=[0.85, 0.925], figsize=(5, 2.5),
         legend=['constant rate, live weights', 'constant rate, EMA',
                 'cosine decay, live weights'], legend_loc='outside')
```

At a constant learning rate, the live weights jitter over a test-accuracy band of up to $\sim 5\%$ across the last five epochs, whereas their EMA tracks the top of the band smoothly at $>91\%$—matching the final accuracy of the cosine-decayed run without ever lowering the learning rate. Thus, weight EMA provides an immediate, zero-step alternative to cooldown branching during the constant plateau of a WSD schedule (:numref:`subsec_scheduler-branching`).

## A Worked End-to-End Configuration

We now synthesize the entire chapter to configure a $3{,}000$-step training run for an untuned $4$-block, width-$256$, $4$-head `TinyLM` ($3.3$ million parameters). :numref:`tab_practice_recipes` summarizes the published optimizer configurations of four frontier pretraining runs, and :numref:`tab_practice_symptoms` provides a diagnostic lookup table mapping training log symptoms to their remedies.

:Published optimizer configurations for four large language model pretraining runs.
:label:`tab_practice_recipes`

| Model | Optimizer | $\beta_1, \beta_2$ | Peak LR & schedule | Warmup | Clip $\theta$ | Weight decay $\lambda$ | Batch size |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` | AdamW | — | $8{\times}10^{-5}$, cosine to $8{\times}10^{-7}$ | 8k steps | — | — | ramp 4M $\to$ 16M tokens |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, WSD cosine to $2.2{\times}10^{-5}$ | 2k steps | 1.0 | 0.1 | ramp 3,072 $\to$ 15,360 seqs |
| OLMo 2 7B :cite:`OLMo.2025` | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, cosine to $3{\times}10^{-5}$ | 2k steps | 1.0 | 0.1 (no emb) | 1,024 seqs |
| Kimi K2 :cite:`Kimi.Team.2025` | MuonClip | — | $2{\times}10^{-4}$, WSD cosine to $2{\times}10^{-5}$ | 500 steps | — | 0.1 | 67M tokens |

:Diagnosing optimization failures from logged metrics (loss, pre-clip gradient norm, clipping fraction, and relative step size $\|\Delta\mathbf{W}\|_{\text{RMS}}/\|\mathbf{W}\|_{\text{RMS}}$).
:label:`tab_practice_symptoms`

| Logged symptom | Diagnosis | Primary remedy |
|:--|:--|:--|
| Loss diverges in first few steps with exploding gradient norms | Step size exceeds initial local sharpness $2/L$ | Add linear warmup or global-norm clipping |
| Loss stalls high under Adam; initial relative weight updates $>10\%$ | Cold-start step corrupts initialization | Add linear warmup ($1\%\text{--}5\%$ of steps) |
| Loss stalls high under Adam even behind warmup | Peak learning rate $\eta_0$ too large | Lower $\eta_0$ by $2\text{--}3\times$ |
| Clipping fires on $>20\%\text{--}50\%$ of steps throughout training | Clipping acting as a constant brake | Lower $\eta_0$ before adjusting $\theta$ |
| Loss plateaus early with high step-to-step variance | Stationary noise ball at current $\eta / b$ | Decay $\eta_t \to 0$, evaluate weight EMA, or increase $b$ |
| Loss descends smoothly at constant slope until final step | Under-budgeted cumulative travel $B_T$ | Increase peak $\eta_0$ or total steps $T$ |

Following :numref:`tab_practice_recipes`, we configure our $3.3\text{M}$-parameter `TinyLM` with:

1. **Batch size:** $b = 256$ sequences (where hardware throughput saturates and comparable to the post-warmup gradient-noise scale $b_{\text{noise}} \sim 100\text{--}200$).
2. **Optimizer & guards:** AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$, decoupled weight decay $\lambda = 0.1$ applied exclusively to 2D hidden matrices (:numref:`subsec_adamw-what-not-to-decay`), global-norm gradient clipping at $\theta = 1.0$, and a short-window weight EMA ($\alpha = 0.995$).
3. **Peak learning rate calibration:** Three $300$-step ($10\%$-horizon) trial runs across $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ identify $\eta_0 = 3 \times 10^{-3}$ as optimal (with $<5\%$ of steps clipped, confirming that $\theta = 1.0$ acts purely as a guard).
4. **Schedule:** $60$-step linear warmup ($2\%$ of $T = 3{,}000$) followed by cosine decay to zero, which lowers validation loss by $0.05\text{--}0.06$ nats below the constant-rate baseline ($<0.01$-nat seed spread):

```{.python .input #practice-a-worked-configuration-1}
%%tab pytorch
def tiny_lm(seed):
    """A TinyLM of four blocks of width 256 on the GPU."""
    torch.manual_seed(seed)  # The initialization and the minibatch order
    return d2l.TinyLM(len(data.vocab), d_model=256, num_heads=4,
                      num_blks=4).to(d2l.try_gpu())

def sample_lm(b):  # b windows at random offsets in the training text
    starts = torch.randint(len(data.train) - data.num_steps, (b,))
    X, Y = data.windows(data.train, starts.numpy())
    return X.to(d2l.try_gpu()), Y.to(d2l.try_gpu())
```

```{.python .input #practice-a-worked-configuration-1}
%%tab jax
def tiny_lm(seed):
    """A TinyLM of four blocks of width 256."""
    return d2l.TinyLM(len(data.vocab), d_model=256, num_heads=4, num_blks=4,
                      rngs=nnx.Rngs(seed))

rng = np.random.default_rng(d2l.get_seed())

def sample_lm(b):  # b windows at random offsets in the training text, on the GPU
    X, Y = data.windows(data.train,
                        rng.integers(len(data.train) - data.num_steps, size=b))
    return jnp.asarray(X), jnp.asarray(Y)
```

```{.python .input #practice-a-worked-configuration-3}
%%tab pytorch
def train(seed, schedule, num_steps, max_norm=1.0, alpha=0.995):
    """AdamW with betas (0.9, 0.95) and weight decay 0.1 on the matrices at
    the rate schedule(t), clipped at max_norm, with an EMA of the weights;
    return the losses, the model, the EMA and the fraction of clipped steps."""
    model = tiny_lm(seed)
    matrices = [p for name, p in model.named_parameters()
                if p.ndim == 2 and 'emb' not in name]
    others = [p for name, p in model.named_parameters()
              if p.ndim != 2 or 'emb' in name]
    adamw = torch.optim.AdamW([{'params': matrices, 'weight_decay': 0.1},
                               {'params': others, 'weight_decay': 0.0}],
                              lr=1.0, betas=(0.9, 0.95))  # Times schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, schedule)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(alpha))
    adamw.register_step_post_hook(lambda *_: scheduler.step())
    adamw.register_step_post_hook(lambda *_: ema.update_parameters(model))
    optimizer = Clipped(adamw, model.parameters(), max_norm)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    clip_rate = (torch.stack(optimizer.norms) > max_norm).float().mean()
    return losses, model, ema, float(clip_rate)
```

```{.python .input #practice-a-worked-configuration-3}
%%tab jax
def decay_mask(params):  # True for the matrices, False for the rest
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

def train(seed, schedule, num_steps, max_norm=1.0, alpha=0.995):
    """AdamW with betas (0.9, 0.95) and weight decay 0.1 on the matrices at
    the rate schedule(t), clipped at max_norm, with an EMA of the weights;
    return the losses, the model, the EMA and the fraction of clipped steps."""
    model = tiny_lm(seed)
    adamw = optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1,
                        mask=decay_mask)
    tx = optax.chain(clipped(adamw, num_steps, max_norm), ema_of_weights(alpha))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses = d2l.train_steps(model, data, optimizer, num_steps)
    norms = optimizer.opt_state[0][0]['norms'][...]  # Recorded by clipped
    clip_rate = (norms > max_norm).mean()
    return losses, model, averaged(model, optimizer), float(clip_rate)
```

```{.python .input #practice-a-worked-configuration-4}
%%tab pytorch, jax
data.batch_size = 256  # Where the throughput levels off
seed, short = new_seed(), {}
print('noise scale at initialization: '
      f'{d2l.noise_scale(tiny_lm(seed), sample_lm, b_small=8, b_big=256):.0f}')
for lr in [1e-3, 3e-3, 1e-2]:
    losses, model, _, clip_rate = train(seed, lambda t: lr, num_steps=300)
    short[lr] = (d2l.final_loss(losses), model)
    print(f'lr {lr:g}: loss {short[lr][0]:.3f} after 300 steps, '
          f'{clip_rate:.0%} of the steps clipped')
peak = min(short, key=lambda lr: short[lr][0])
print(f'peak rate {peak:g}; noise scale after its 300 steps: '
      f'{d2l.noise_scale(short[peak][1], sample_lm, b_small=8, b_big=256):.0f}')
```

```{.python .input #practice-a-worked-configuration-5}
%%tab pytorch, jax
num_steps = 3000
val_loss = lambda model: math.log(d2l.lm_perplexity(model, data.val_dataloader()))
print(f'{"validation loss":<16}{"configured":>11}{"its EMA":>9}'
      f'{"clipped":>9}{"baseline":>10}{"gap":>7}')
results = []
for seed in [new_seed() for _ in range(2)]:
    _, model, ema, clip_rate = train(
        seed, warmup_cosine(peak, num_steps // 50, num_steps), num_steps)
    _, baseline, _, _ = train(seed, lambda t: peak, num_steps)
    results.append((val_loss(model), val_loss(baseline)))
    print(f'{"seed " + str(len(results)):<16}{results[-1][0]:11.3f}'
          f'{val_loss(ema):9.3f}{clip_rate:9.1%}{results[-1][1]:10.3f}'
          f'{results[-1][1] - results[-1][0]:7.3f}')
(c1, b1), (c2, b2) = results
print(f'spread between the seeds: configured {abs(c1 - c2):.3f}, '
      f'baseline {abs(b1 - b2):.3f}')
```

## Summary

* **Global-norm gradient clipping** ($\mathbf{g} \leftarrow \min(1, \theta/\|\mathbf{g}\|_2)\mathbf{g}$) acts as a **guard** when it fires on a small fraction ($<5\%$) of early or anomalous steps—preventing initial blow-up under SGD and protecting AdamW's second-moment buffer $\mathbf{v}_t$ from gradient spikes—and as a **brake** when it fires continuously (indicating that $\eta$ is too high).
* **Exponential moving averaging (EMA)** of model weights ($\bar{\mathbf{x}}_t = \alpha\bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$) smooths out stationary noise-ball fluctuations, matching the accuracy of a full cosine decay on constant-rate trajectories.
* Production training runs combine hardware-saturated batch sizes below or near $b_{\text{noise}}$, AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$ and matrix-only weight decay $\lambda = 0.1$, clipping at $\theta = 1.0$, short-horizon peak-LR calibration, and linear warmup followed by cosine or WSD cooldown.

## Exercises

1. [code] **Clipping regimes: guard vs. brake.** Sweep $\theta \in \{0.01, 0.1, 0.5, 1.0, 4.0, \infty\}$ at $\eta = 1.0$ with `run` on `TinyLM`, recording the final loss and clipping fraction, and compare clipping against replacing clipping with a $20$-step or $100$-step linear warmup.
   :exlabel:`ex_practice-clipping-regimes`

1. **Softmax logit drift and PaLM's $z$-loss.** For output logits $\mathbf{o} \in \mathbb{R}^V$ and partition function $Z = \sum_{j=1}^V e^{o_j}$, show that the cross-entropy gradient has zero sum ($\mathbf{1}^\top \nabla_{\mathbf{o}}\ell_{\text{CE}} = 0$) whereas adding the auxiliary penalty $10^{-4}\log^2 Z$ :cite:`chowdhery2022palm` introduces a restoring gradient $2 \times 10^{-4}(\log Z)\,\mathbf{p}$ that pulls $\log Z$ toward $0$ without altering the uniform-shift invariance of $\mathbf{p} = \operatorname{softmax}(\mathbf{o})$.
   :exlabel:`ex_practice-logit-drift`

1. [code] **Weight EMA window and start time.** Sweep the EMA decay $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ and start epoch `start` $\in \{0, 5, 10\}$ on `train_fashion` at constant $\eta = 0.3$, relating the accuracy trajectories to the effective window $1/(1-\alpha)$ in epochs.
   :exlabel:`ex_practice-averaging-window`

1. [code] **Ablating the worked configuration.** ● On the $4$-block width-$256$ `TinyLM` of `train`, run two-seed ablations testing: (a) WSD late linear decay vs. cosine decay, (b) removing warmup or removing gradient clipping, (c) shifting the peak rate to $10^{-3}$ or $10^{-2}$, and (d) evaluating the weight EMA of the constant-rate baseline at $\alpha \in \{0.995, 0.999\}$.
   :exlabel:`ex_practice-worked-configuration`

[Discussions](https://d2l.discourse.group/t/36767)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.10]{.kicker}

Practical Optimization Diagnostics and Recipes<br>**global-norm gradient clipping · guard vs. brake · AdamW spike recovery · weight EMA · end-to-end configuration**
:::
:::

::: {.slide title="Three Pillars of a Production Pretraining Run"}
[Motivation]{.kicker}

In small experiments, a diverged run costs seconds to restart. When pretraining a foundation model across thousands of accelerators for weeks—such as the $540\text{B}$-parameter PaLM model on $6{,}144$ TPUs, which suffered $\sim 20$ sudden loss spikes despite active clipping (Chowdhery et al., 2022)—reliable execution rests on three operational pillars:

::: {.cols}
::: {.col}
- **1. Guards against transient instabilities:**
  - Global-norm **gradient clipping** prevents early high-curvature blow-ups under SGD and protects AdamW's second-moment accumulator $\mathbf{v}_t$ from anomalous gradient spikes, paired with checkpoint-and-rewind recovery.
- **2. Iterate smoothing (Weight EMA):**
  - Maintaining an exponential moving average of model weights cancels late-stage noise-ball fluctuations without requiring a full decay to zero.
:::

::: {.col}
- **3. Calibration from short proxy runs:**
  - Rather than sweeping a full $3{,}000$-step (or $300{,}000$-step) run across a grid, we select $b$ from hardware throughput and $b_{\text{noise}}$ (§9.8), adopt published AdamW defaults, calibrate peak $\eta_0$ on a $10\%$-horizon run, and execute a warmup + cosine/WSD schedule (§9.7).
:::
:::
:::

::: {.slide title="Global-Norm Gradient Clipping: Direction-Preserving Cap"}
[Clipping Mechanics]{.kicker}

::: {.cols .vc}
::: {.col}
Global-norm gradient clipping treats the concatenated gradient across all parameters as a single vector $\mathbf{g} \in \mathbb{R}^d$ and rescales it whenever its Euclidean norm $\|\mathbf{g}\|_2$ exceeds a threshold $\theta > 0$:

$$\mathbf{g} \leftarrow \min\!\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right)\mathbf{g}$$

- **Preserves cross-layer relative magnitudes:** Unlike elementwise clipping (which distorts each layer's direction into a sign vector), global-norm clipping multiplies every parameter's gradient by the same scalar $\theta / \|\mathbf{g}\|_2 < 1$.
- **Bounds SGD step length:** Under plain SGD, no parameter update can exceed Euclidean length $\eta\theta$.
:::

::: {.col .fig}
@-practice-gradient-clipping-1
:::
:::
:::

::: {.slide title="Testing Two Hypotheses for SGD Divergence at η = 1.0"}
[Guard vs. Brake Experiment]{.kicker}

In §9.5, SGD with momentum ($\beta = 0.9$) on `TinyLM` trained at $\eta = 0.3$ but diverged to `inf` within $20$ steps at $\eta = 1.0$. Why did $\eta = 1.0$ diverge?

| Competing Hypothesis | Prediction Under Global-Norm Clipping ($\theta = 1.0$) | Resulting Training Behavior |
| :--- | :--- | :--- |
| **Hypothesis A — Transient initial sharpness:** $\eta = 1.0$ is only too large during the opening steps before the iterate leaves the steep initial region | Clipping fires on the **first few dozen steps**, then falls completely silent ($\|\mathbf{g}_t\|_2 < \theta$) for the rest of the run | **Clipping acts as a guard:** The run survives the opening transient and trains cleanly at $\eta = 1.0$! |
| **Hypothesis B — Globally excessive learning rate:** $\eta = 1.0$ exceeds $2/L$ throughout the entire loss landscape | Clipping fires on **almost every step** ($>50\text{--}100\%$), permanently rescaling the step to fixed length $\eta\theta$ | **Clipping acts as a brake:** Changes the algorithm into fixed-step normalized descent and still fails to converge |
:::

::: {.slide title="Clipping Fires on <2% of Steps and Rescues η = 1.0"}
[Empirical Clipping Diagnostics on TinyLM]{.kicker}

::: {.cols .vc}
::: {.col}
- Running $2{,}000$ steps across $(\eta, \theta) \in \{(1.0, \infty), (1.0, 1.0), (0.3, \infty), (3.0, 1.0)\}$ over two seeds confirms **Hypothesis A** at $\eta = 1.0$:
  - **At $\eta = 1.0, \theta = \infty$:** Diverges to `inf` within $20$ steps.
  - **At $\eta = 1.0, \theta = 1.0$ (Guard):** Clipping fires on only **$15\text{--}35$ steps (all before step $50$)**, never fires again over the remaining $1{,}950$ steps (median $\|\mathbf{g}_t\|_2 \approx 0.25$), and reaches **$1.61\text{--}1.64$ nats** (beating unclipped $\eta = 0.3$)!
  - **At $\eta = 3.0, \theta = 1.0$ (Brake):** Fires on **every single step** until overflowing or stalling above $100$ nats.
:::

::: {.col .fig}
@!practice-a-nan-averted-2
:::
:::
:::

::: {.slide title="Distinguishing a Guard from a Brake in Training Logs"}
[Setting the Clipping Threshold θ]{.kicker}

| Diagnostic Dimension | Clipping as a **Guard** ($\eta = 1.0, \theta = 1.0$) | Clipping as a **Constant Brake** ($\eta = 3.0, \theta = 1.0$) |
| :--- | :--- | :--- |
| **Fraction of steps clipped** | **$<2\%\text{--}5\%$** (concentrated in early transient or rare spikes) | **$>50\%\text{--}100\%$** (active continuously throughout training) |
| **Steady-state median $\|\mathbf{g}_t\|_2$** | $\approx 0.25 \approx \theta / 4$ (well below threshold $\theta$) | Tens or hundreds of $\theta$ (permanently saturated) |
| **Effective update rule** | Standard SGD/AdamW with rare outlier steps capped | Fixed-step normalized descent; masks an oversized learning rate |
| **Engineering action** | **Healthy setting:** Keep $\theta \approx 3\text{--}5\times$ median norm (typically $\theta = 1.0$) | **Lower the learning rate $\eta$ first** before tuning $\theta$! |
:::

::: {.slide title="Why Clipping Under AdamW Protects the Second Moment v_t"}
[AdamW Spike Dynamics]{.kicker}

Under AdamW (§9.5), a single massive gradient coordinate $g_t \gg \sqrt{v_{t-1}}$ *already* produces a bounded immediate parameter step because $g_t$ appears in both $\mathbf{m}_t$ and $\sqrt{\mathbf{v}_t}$:

$$|\Delta x_t| \approx \eta\,\frac{(1 - \beta_1)\,|g_t|}{\sqrt{(1 - \beta_2)\,g_t^2}} = \eta\,\frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \approx \begin{cases} 3.16\,\eta & (\beta_1 = 0.9,\; \beta_2 = 0.999) \\ 0.45\,\eta & (\beta_1 = 0.9,\; \beta_2 = 0.95) \end{cases}$$

- **What clipping actually protects in AdamW is the denominator state $\mathbf{v}_t$:**
  - An unclipped **$100\times$ gradient spike** injects $(1 - \beta_2) \times 100^2$ times the typical squared gradient into $v_t$:
  - At $\beta_2 = 0.999$: $v_t$ jumps to $11\times$ its normal level, shrinking subsequent effective learning rates to $1/\sqrt{11} \approx 30\%$ and taking **$t \approx \frac{\ln 10}{1 - 0.999} \approx 2{,}300$ steps** just to decay back below $2\times$!
  - Clipping at $\theta = 4\times$ median norm limits the injection into $v_t$ to $0.001 \times 16 < 2\%$, and using **$\beta_2 = 0.95$** shortens any residual recovery window **$20\times$** (from $2{,}300$ steps to $\sim 120$ steps).
:::

::: {.slide title="Operational Spike Recovery and Softmax z-Loss"}
[Surviving Mid-Run Instabilities]{.kicker}

When a sudden loss spike occurs mid-run despite gradient clipping ($\theta = 1.0$) and shorter second-moment memory ($\beta_2 = 0.95$):

- **Coincidence of parameter state and anomalous minibatch (PaLM, Chowdhery et al., 2022):**
  - Replaying the offending minibatch from an *earlier* checkpoint does not spike, and continuing from just *before* the spike with different minibatches does not spike—the instability is triggered by the interaction of a marginal parameter state with a specific data batch.
- **Standard operational recovery protocol:**
  1. **Rewind** to a checkpoint saved **$\sim 100$ steps before** the spike onset.
  2. **Skip** the **$200\text{--}500$ minibatches** surrounding the spike window.
  3. **Full state checkpointing:** Every checkpoint must serialize model weights, optimizer moments $(\mathbf{m}_t, \mathbf{v}_t)$, scheduler step $t$, and the deterministic data-loader offset!
- **Auxiliary logit stabilization ($z$-loss):** Adding $10^{-4}\log^2\!\left(\sum_{j=1}^V e^{o_j}\right)$ pulls the log-partition function $\log Z \to 0$ via restoring gradient $2{\times}10^{-4}(\log Z)\,\mathbf{p}$, preventing softmax logit drift without altering $\mathbf{p} = \operatorname{softmax}(\mathbf{o})$.
:::

::: {.slide title="Weight Averaging (EMA): Smoothing the Noise Ball"}
[Polyak–Ruppert Iterate Averaging]{.kicker}

::: {.cols .vc}
::: {.col}
- At any non-zero learning rate $\eta$, the live weights $\mathbf{x}_t$ represent a single random draw from the stationary noise ball (§9.2).
- Maintaining an **exponential moving average (EMA)** $\bar{\mathbf{x}}_t$ of the live parameters with decay $\alpha \in [0.99, 0.9999]$ (Polyak & Juditsky, 1992):
  $$\bar{\mathbf{x}}_t = \alpha\,\bar{\mathbf{x}}_{t-1} + (1 - \alpha)\,\mathbf{x}_t$$
  averages over a trailing window of **$\tau_{\text{EMA}} = \frac{1}{1 - \alpha}$ steps** without altering the live training trajectory.
- **Choosing $\alpha$ and start step:** $\tau_{\text{EMA}}$ must exceed the noise-ball autocorrelation time $\frac{1}{\eta\lambda_{\text{curv}}}$ (e.g., $\alpha = 0.999 \implies 1{,}000$ steps), and averaging should start after the initial rapid descent phase so early high-loss weights do not lag the average.
:::

::: {.col .fig}
@-practice-weight-averaging-4
:::
:::
:::

::: {.slide title="Constant-Rate Weight EMA vs. Live Weights on FashionCNN"}
[Eliminating Epoch-to-Epoch Jitter]{.kicker}

::: {.cols .vc}
::: {.col}
- Training `FashionCNN` for $15$ epochs ($235$ steps/epoch $= 3{,}525$ steps) with plain SGD at a constant learning rate $\eta = 0.3$:
  - **Live weights (blue):** Bounce wildly inside the noise ball, jittering across a **$2\text{--}6$ percentage-point test-accuracy band** ($86\%\text{--}91\%$) over the last five epochs!
  - **Weight EMA ($\alpha = 0.999$, started at epoch 5, orange):** Smoothly tracks the very top of the band at **$>91\%$ test accuracy** on every single epoch, matching the full cosine-decayed trajectory (green)!
:::

::: {.col .fig}
@!practice-weight-averaging-3
:::
:::
:::

::: {.slide title="Weight EMA Matches Cosine Decay Without Lowering η"}
[Two-Seed Quantitative Comparison]{.kicker}

::: {.cols .vc}
::: {.col}
- Comparing constant-rate SGD ($\eta = 0.3$) against cosine decay ($\eta: 0.3 \to 0$) across two seeds on `FashionCNN`:
  - **At a constant learning rate:** The EMA improves final test accuracy by $+1\text{--}5$ percentage points over the noisy live checkpoint, matching the **$91.1\%\text{--}91.4\%$** accuracy of a full cosine decay!
  - **After cosine decay to zero:** The live weights have already settled at the center of the noise ball ($\eta_T = 0$), so taking an EMA on top of a fully decayed run adds nothing (or lags slightly by $0.1\text{--}0.3\%$).
- **When to use Weight EMA:** During constant-rate / WSD plateau training to obtain an immediately deployable model at any step without waiting for a cooldown branch!
:::

::: {.col .fig}
@!practice-weight-averaging-2
:::
:::
:::

::: {.slide title="Published Frontier Pretraining Recipes: What Agrees"}
[Industry Reference Configurations]{.kicker}

Comparing four frontier LLM pretraining runs shows remarkable agreement on optimizer structure alongside scale-dependent learning rates and batch sizes:

| Model & Scale | Optimizer | Moments $(\beta_1, \beta_2)$ | Peak LR $\eta_0$ & Schedule | Warmup | Clip $\theta$ | Weight Decay $\lambda$ | Batch Size Policy |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Llama 3 405B** (Grattafiori et al., 2024) | AdamW | — | $8{\times}10^{-5}$, cosine to $8{\times}10^{-7}$ | $8{,}000$ steps | — | — | Ramp $4\text{M} \to 16\text{M}$ tokens |
| **DeepSeek-V3 671B** (Liu et al., 2024) | AdamW | $(0.9, 0.95)$ | $2.2{\times}10^{-4}$, WSD cosine to $2.2{\times}10^{-5}$ | $2{,}000$ steps | $1.0$ | $0.1$ | Ramp $3{,}072 \to 15{,}360$ seqs |
| **OLMo 2 7B** (OLMo Team, 2025) | AdamW | $(0.9, 0.95)$ | $3{\times}10^{-4}$, cosine to $3{\times}10^{-5}$ | $2{,}000$ steps | $1.0$ | $0.1$ (no emb) | $1{,}024$ seqs ($\sim 4\text{M}$ tokens) |
| **Kimi K2 1T** (Kimi Team, 2025) | MuonClip | — | $2{\times}10^{-4}$, WSD cosine to $2{\times}10^{-5}$ | $500$ steps | — | $0.1$ | $67\text{M}$ tokens |
:::

::: {.slide title="Diagnostic Playbook: Reading Failure Signatures in Logs"}
[From Telemetry to Action]{.kicker}

Always log four curves on every run: **training loss**, **pre-clip global gradient norm $\|\mathbf{g}_t\|_2$**, **clipping indicator ($\|\mathbf{g}_t\|_2 > \theta$)**, and **relative weight update $\|\Delta\mathbf{W}\|_{\text{RMS}}/\|\mathbf{W}\|_{\text{RMS}}$**:

| Logged Telemetry Signature | Root-Cause Diagnosis | Primary Engineering Remedy |
| :--- | :--- | :--- |
| **Loss diverges in first few steps; $\|\mathbf{g}_t\|_2$ explodes** | Step size exceeds initial local sharpness ceiling $2/L$ | Add linear warmup ($1\text{--}5\%$) or global-norm clipping ($\theta = 1.0$) |
| **Loss stalls high under AdamW; step-1 $\|\Delta\mathbf{W}\|/\|\mathbf{W}\| > 10\%$** | Unaveraged cold-start $\pm\eta_0$ jump corrupts initialization | Add linear warmup over $W \ge \frac{1}{1-\beta_1}$ steps ($1\text{--}5\%$ of $T$) |
| **Loss stalls high under AdamW *behind* a warmup** | Peak learning rate $\eta_0$ is globally too large | Lower peak learning rate $\eta_0$ by $2\text{--}3\times$ ($1$ grid step) |
| **Clipping fires on $>20\text{--}50\%$ of steps throughout run** | Clipping is acting as a permanent brake, not a guard | Lower $\eta_0$ first before adjusting threshold $\theta$ |
| **Loss plateaus early with high step-to-step jitter** | Iterates trapped in stationary noise ball at current $\eta/b$ | Decay $\eta_t \to 0$, evaluate weight EMA $\bar{\mathbf{x}}_t$, or increase batch size $b$ |
| **Loss descends smoothly at constant slope until step $T$** | Under-budgeted cumulative travel $B_T = \sum_t \eta_t$ | Raise peak rate $\eta_0$ or extend total step horizon $T$ |
:::

::: {.slide title="Case Study: Configuring a 3.3M-Parameter TinyLM"}
[Putting All 10 Sections Together]{.kicker}

::: {.cols .vc}
::: {.col}
We configure a $3{,}000$-step training run for a new $4$-block, width-$256$, $4$-head `TinyLM` ($3.3\text{M}$ parameters, $7\times$ larger than our $2$-block model) without repeating the full $3{,}000$-step run across a grid:

1. **Batch size ($b = 256$ sequences):** Chosen where GPU throughput saturates and comparable to post-warmup $b_{\text{noise}} \sim 100$.
2. **Optimizer & guards:** AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$, decoupled weight decay $\lambda = 0.1$ on 2D matrices only, global-norm clipping $\theta = 1.0$, and weight EMA ($\alpha = 0.995$).
3. **Short-horizon peak LR calibration ($10\%$ budget):** Sweep $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}\}$ for $300$ steps.
4. **Schedule:** $60$-step linear warmup ($2\%$) + cosine decay to zero.
:::

::: {.col .fig}
@-practice-a-worked-configuration-3
:::
:::
:::

::: {.slide title="300-Step Proxy Calibration of Peak LR and Noise Scale"}
[10%-Horizon Calibration]{.kicker}

::: {.cols .vc}
::: {.col}
- Before launching the $3{,}000$-step run, we execute three $300$-step trial runs ($900$ steps total, $<30\%$ of one full run) across $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}\}$:
  - **$\eta = 10^{-3}$:** Slow ($2.14$ nats, $0\%$ clipped).
  - **$\eta = 3 \times 10^{-3}$:** **Optimal ($1.84$ nats, $4\%$ early steps clipped—a healthy guard!)**.
  - **$\eta = 10^{-2}$:** Stalls at $2.50$ nats ($23\%$ clipped).
- **Noise-scale check:** $b_{\text{noise}}$ grows from $\approx 2$ sequences at step 0 to $\sim 100$ sequences at step 300, confirming that $b = 256$ sits right at the critical batch size knee.
:::

::: {.col .fig}
@!practice-a-worked-configuration-4
:::
:::
:::

::: {.slide title="Full 3,000-Step Verification Across Seeds"}
[Configured Recipe vs. Constant-Rate Baseline]{.kicker}

::: {.cols .vc}
::: {.col}
- Running the full $3{,}000$-step warmup + cosine schedule at the calibrated peak rate $\eta_0 = 3 \times 10^{-3}$ across two independent seeds:
  - **Configured recipe:** Reaches **$1.337\text{--}1.339$ validation nats**, beating the constant-rate baseline ($1.390\text{--}1.395$) by **$0.052\text{--}0.056$ nats** ($>25\times$ the $0.002$-nat inter-seed spread!).
  - **Guard verification:** Clipping fires on only **$0.6\%$ of steps** ($\sim 18$ steps out of $3{,}000$).
  - **Post-decay EMA:** Matches the live weights to within $0.001$ nats because cosine decay already reduced $\eta_T \to 0$.
:::

::: {.col .fig}
@!practice-a-worked-configuration-5
:::
:::
:::

::: {.slide title="Concept Check: Clipping, Spike Recovery, and Weight EMA"}
[Check Your Understanding]{.kicker}

Diagnose and solve each practical training scenario:

1. **Diagnosing a stalled AdamW run:** On the $3.3\text{M}$-parameter `TinyLM` at $\eta = 10^{-2}$ with clipping at $\theta = 1.0$, the loss stalls near $2.5$ nats and clipping fires on $23\%$ of steps. A colleague proposes lowering the clipping threshold to $\theta = 0.2$ to shorten the steps. Will that fix the stall?
2. **AdamW second-moment recovery time:** Suppose an unclipped gradient spike of $100\times$ normal magnitude hits AdamW and multiplies $v_t$ by $1 + (1-\beta_2)10^4$. How many steps does it take for that excess contribution in $v_t$ to decay by $e^{-3} \approx 5\%$ under $\beta_2 = 0.999$ versus $\beta_2 = 0.95$?
3. **When does Weight EMA add zero gain?** In our final $3{,}000$-step verification table, why did the configured run's live weights and its weight EMA achieve the exact same validation loss ($1.338$ vs. $1.339$)?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **No!** Because AdamW divides $\hat{\mathbf{m}}_t$ by $\sqrt{\hat{\mathbf{v}}_t}$, rescaling $\mathbf{g}_t$ by a tighter global clipping threshold $\theta = 0.2$ rescales both numerator and denominator and leaves the steady-state coordinate step length $\approx \eta$ unchanged! You must **lower the learning rate $\eta$** (e.g., to $3 \times 10^{-3}$).
2. Decay by $e^{-3}$ takes $3\,\tau_2 = \frac{3}{1 - \beta_2}$ steps: **$3{,}000$ steps at $\beta_2 = 0.999$** versus only **$60$ steps at $\beta_2 = 0.95$** ($20\times$ faster recovery!).
3. Because the configured run ended with **cosine decay to $\eta_T = 0$**, the noise ball had already shrunk to zero during the final steps; weight EMA shines when evaluating **uncooled constant-rate or WSD-plateau checkpoints**.
:::
:::

::: {.slide title="Summary: Practical Optimization Diagnostics and Recipes"}
[Key Takeaways]{.kicker}

| Operational Pillar | Mathematical Rule / Mechanism | Healthy Telemetry Target | Production Role |
| :--- | :--- | :--- | :--- |
| **Global-Norm Clipping** | $\mathbf{g} \leftarrow \min\!\left(1, \frac{\theta}{\|\mathbf{g}\|_2}\right)\mathbf{g}$ ($\theta = 1.0$) | Fires on $<2\%\text{--}5\%$ of steps (guard); if $>20\%$, lower $\eta$ | Caps early SGD steps; protects AdamW's $\mathbf{v}_t$ from spikes |
| **$\beta_2 = 0.95$ & Rewind** | $\tau_2 = \frac{1}{1-\beta_2} = 20$ steps (vs. $1{,}000$ at $0.999$) | Rewind $\sim 100$ steps & skip $200\text{--}500$ batches on spike | Prevents a single spike from paralyzing $\sqrt{\hat{\mathbf{v}}_t}$ for $2{,}300$ steps |
| **Weight Averaging (EMA)** | $\bar{\mathbf{x}}_t = \alpha\bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$ ($\alpha \in [0.995, 0.9999]$) | Window $\frac{1}{1-\alpha} > \frac{1}{\eta\lambda_{\text{curv}}}$; start after early descent | Matches cosine-decayed accuracy on constant-rate trajectories |
| **Short-Run Calibration** | Sweep $\eta_0$ over $10\%$ horizon ($300$ of $3{,}000$ steps) | Pick $\eta_0$ $3\times$ below stall threshold; check $b_{\text{noise}}$ | Calibrates peak LR for $<30\%$ of a single full training run |
| **Modern Default Recipe** | AdamW $(0.9, 0.95)$, matrix-only $\lambda = 0.1$, clip $1.0$ | $1\text{--}5\%$ linear warmup + cosine or WSD decay to $\le 0.1\eta_0$ | Reliable baseline across Transformers and modern deep networks |
:::

