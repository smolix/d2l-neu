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

At $\eta = 1.0$, clipping at $\theta = 1.0$ fires on only the first few dozen steps (all before step $50$) and never fires again over the remaining $1{,}950$ steps (where the median gradient norm is $\sim 0.25$), rescuing the run from `inf` down to $\sim 1.61\text{--}1.64$ nats. Here clipping acts as a **guard**: it intervenes on $<2\%$ of steps to prevent early transient blow-up (and handle heavy-tailed gradient noise :cite:`Zhang.Karimireddy.Veit.ea.2020`). By contrast, at $\eta = 3.0$, clipping fires on nearly $100\%$ of steps and the run still fails to train: when clipping is permanently active as a **brake**, the learning rate $\eta$ is simply too large :cite:`Godbole.Dahl.Gilmer.ea.2023`. In practice, set $\theta$ roughly $3\text{--}5\times$ above the steady-state median gradient norm (typically $\theta = 1.0$) and monitor the clipping fraction.

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

At a constant learning rate, the live weights jitter over a $2\%\text{--}6\%$ test-accuracy band across the last five epochs, whereas their EMA tracks the top of the band smoothly at $>91\%$—matching the final accuracy of the cosine-decayed run without ever lowering the learning rate. Thus, weight EMA provides an immediate, zero-step alternative to cooldown branching during the constant plateau of a WSD schedule (:numref:`subsec_scheduler-branching`).

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
1. **Batch size:** $b = 256$ sequences (where hardware throughput saturates and comparable to the post-warmup gradient-noise scale $b_{\text{noise}} \sim 100$).
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
[Dive into Deep Learning · §9.12]{.kicker}

Practice<br>
**gradient clipping · weight averaging · a worked configuration**
:::
:::

::: {.slide title="PaLM's Loss Spiked about 20 Times with Gradient Clipping On"}
[The problem]{.kicker}

A run that takes a day on one GPU:

- **Guard:** one oversized update can undo hours. PaLM, 540 billion
  parameters on 6,144 TPU chips: about 20 loss spikes, with clipping
  enabled.
- **Finish:** the final weights are one draw from a noisy trajectory.
- **Configure:** a sweep over four learning rates costs four days.

::: {.d2l-note .warn}
How do we guard, finish and configure a run that we get to do once?
:::

[Chowdhery et al., 2022]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- set a clipping threshold and tell a guard from a brake;
- compute how long a gradient spike stays in Adam's state;
- recover from a loss spike;
- choose the window and the start of a weight average;
- configure a new model: batch size, optimizer, peak rate, schedule.

**Idea:** the settings of a long run come from short runs of the same
model; a guard and an average cover what short runs cannot foresee.

**Running example:** `TinyLM` (transformer, 0.47 million parameters, next
byte of Wikipedia text) under SGD at a diverging rate; then a `TinyLM` of
3.3 million parameters, configured from nothing.
:::

::: {.slide title="Practice Assembles the Chapter into One Configured Run"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · **Practice** |

- **Uses:** SGD's diverging rates on `TinyLM`; Adam's two averages; the
  noise ball; warmup and cosine decay; the gradient-noise scale.
- **Skips:** automated hyperparameter search; training on several
  devices.
:::

::: {.slide title="Global-Norm Clipping Caps the Step and Keeps Its Direction"}
[Gradient clipping]{.kicker}

All gradients as one vector $\mathbf{g}$, threshold $\theta$:

$$\mathbf{g} \leftarrow \min\left(1,\; \frac{\theta}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

- $\|\mathbf{g}\|_2 = 5$, $\theta = 1$: multiply by 0.2.
- One norm over all layers: their relative sizes are kept.
- Plain SGD with learning rate $\eta$: no step is longer than
  $\eta\theta$.
:::

::: {.slide title="Two Readings of a Diverging Rate Predict Different Clip Counts"}
[Gradient clipping]{.kicker}

SGD with momentum 0.9 on `TinyLM`: best at $\eta = 0.3$, every run
diverges at $\eta = 1$.

| if $\eta = 1$ is too large | clipping at $\theta = 1$ acts | and the run |
|:--|:--|:--|
| for the first steps only | early, then never | trains at $\eta = 1$ |
| throughout | on most steps | is no longer SGD at $\eta = 1$ |

**Test:** $\eta = 1$ unclipped and clipped at $\theta = 1$, $\eta = 0.3$
unclipped, $\eta = 3$ clipped; two seeds, 2,000 steps; count the clipped
steps.
:::

::: {.slide title="At η = 1, Clipping Separates Divergence from Training"}
[Gradient clipping · TinyLM, first 50 steps]{.kicker}

@!practice-a-nan-averted-3

- Unclipped: the loss falls for a step or two, then grows by orders of
  magnitude and is gone within about 20 steps.
- Clipped at 1: hard to tell from $\eta = 0.3$ on this scale.
:::

::: {.slide title="Clipping Changed a Few Dozen Early Steps, Then Fell Silent" layout="tight"}
[Gradient clipping]{.kicker}

@!practice-a-nan-averted-2

[`last`: last clipped step. `largest / median`: after step 100.]{.small}

::: {.d2l-note}
$\eta = 1$ was too large for the first 50 steps only, and SGD's limit
moved from about 0.5 to between 1 and 3. When a run of a sweep diverges,
ask when.
:::
:::

::: {.slide title="A Guard Fires Rarely; a Brake Fires Always and Changes the Algorithm"}
[Gradient clipping]{.kicker}

| | guard ($\eta = 1$) | brake ($\eta = 3$) |
|:--|:--|:--|
| clipped steps | a few dozen of 2,000 | practically all |
| gradient norm | median $\theta/4$ | tens of $\theta$ |
| update | SGD, a few steps shortened | fixed length: normalized descent |
| run | trains | fails |

- **Threshold:** a few times the median norm of a run that trains. Log
  the share of clipped steps.
- **Fires on most steps:** lower $\eta$ first.
:::

::: {.slide title="Check: Does Clipping Shorten an Adam Step?"}
[Check]{.kicker}

One gradient coordinate $g$ arrives that dwarfs all earlier ones. Adam:
$m \leftarrow \beta_1 m + (1-\beta_1) g$,
$v \leftarrow \beta_2 v + (1-\beta_2) g^2$, step $\eta\, m/\sqrt{v}$, with
$\beta_1 = 0.9$, $\beta_2 = 0.999$. How long is the step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $m/\sqrt{v} \approx (1-\beta_1)/\sqrt{1-\beta_2} =
0.1/\sqrt{0.001} \approx 3.2$ for any $g$. The step is bounded at
$3.2\,\eta$, and clipping $g$ barely changes it. So clipping does not
rescue Adam from a learning rate that is too large.
:::
:::
:::

::: {.slide title="Clipping Protects Adam's State: a 100× Spike Stays in v for 2,300 Steps"}
[Gradient clipping]{.kicker}

An unclipped gradient 100 times the typical one, $\beta_2 = 0.999$:

- $v$ gains $(1-\beta_2) \cdot 100^2 = 10$ times its usual value: the
  steps shrink to $1/\sqrt{11} \approx 30\%$;
- $10 \cdot 0.999^t < 1$ only after $t \approx 2{,}300$ steps, and the
  steps are still at 70%;
- clipped at 4 times the median norm: $v$ gains $0.001 \cdot 4^2$, under
  2%.

$\beta_2 = 0.95$: $v$ gains 500 times, the steps fall below 5%, and the
spike is forgotten within about 120 steps.
:::

::: {.slide title="When a Spike Gets Through, Rewind and Skip"}
[Gradient clipping]{.kicker}

- PaLM: restart about **100 steps** before the spike, skip **200–500
  minibatches**. No spike there again.
- Same minibatches, earlier checkpoint: no spike. Data and parameter
  state cause it together; the report gives no further cause.
- A penalty on drifting logits (z-loss) was on throughout and did not
  prevent the spikes.

::: {.d2l-note}
A checkpoint holds model, optimizer state, step count and data position.
:::

[Chowdhery et al., 2022]{.small}
:::

::: {.slide title="An Average Needs Iterates That Are Different Draws"}
[Weight averaging]{.kicker}

Which weights do we keep? The last ones are one draw from the *noise
ball*, in which the weights fluctuate at a constant rate.

Exponential moving average (EMA) of the weights $\mathbf{x}_t$, decay
$\alpha$:

$$\bar{\mathbf{x}}_t = \alpha\, \bar{\mathbf{x}}_{t-1} + (1-\alpha)\, \mathbf{x}_t,
\qquad \textrm{window} \approx \frac{1}{1-\alpha}\ \textrm{steps}$$

- Valley, curvature $c = 0.2$, learning rate $\eta = 0.1$: a coordinate
  forgets its position in $1/(\eta c) = 50$ steps.
- So the window is long: $\alpha = 0.999$, 1,000 steps.
:::

::: {.slide title="At a Constant Rate the EMA Is Steadier Than the Live Weights"}
[Weight averaging · CNN on Fashion-MNIST, SGD, η = 0.3, 235 steps per epoch]{.kicker}

@!practice-weight-averaging-3

- EMA ($\alpha = 0.999$) started after epoch 5. From step 0, the
  initialization would still weigh $0.999^{235} \approx 0.79$ after one
  epoch.
- Each epoch the live weights, the ones that training updates, show one
  point of the noise ball.
:::

::: {.slide title="The EMA Matches a Decayed Run; after a Decay It Adds Nothing" layout="tight"}
[Weight averaging]{.kicker}

Two seeds per schedule, test accuracy:

@!practice-weight-averaging-2

- Constant rate, live: two to six points of scatter. EMA: a little above
  91%.
- Cosine decay to zero: about 91% as well. A tie.
- EMA of the decayed run: a few tenths of a point behind at most.

::: {.d2l-note}
Use an EMA when no decay is planned, or for a model needed before the
decay.
:::
:::

::: {.slide title="Check: When Does an EMA Not Help?"}
[Check]{.kicker}

Name two situations in which the EMA of the weights is no better than
the last iterate.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** (1) After the schedule has decayed the rate to nearly zero:
the weights barely move, and the average only lags. (2) While the weights
improve faster than the window follows: start the average late.
:::
:::
:::

::: {.slide title="Four Large Runs Agree on Most Settings and Differ on the Peak Rate"}
[A worked configuration]{.kicker}

| run | optimizer | $\beta_1,\beta_2$ | peak rate | warmup | clip | weight decay |
|:--|:--|:--|:--|:--|:--|:--|
| Llama 3 405B | AdamW | — | $8{\times}10^{-5}$ | 8k | — | — |
| DeepSeek-V3 | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| OLMo 2 7B | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$ | 2k | 1.0 | 0.1 |
| Kimi K2 | MuonClip | — | $2{\times}10^{-4}$ | 500 | — | 0.1 |

- All four warm up and later lower the rate along a cosine.
- The widest model has the smallest peak rate.
- MuonClip: Muon with a cap on attention scores. A dash is missing
  evidence, not a default.

[Grattafiori et al., 2024; Liu et al., 2024; OLMo Team, 2025; Kimi Team, 2025]{.small}
:::

::: {.slide title="A New Model Is Configured from Short Runs, One Decision at a Time"}
[A worked configuration · TinyLM, 4 blocks, width 256, 3.3M parameters]{.kicker}

The run we pretend we cannot repeat: 3,000 steps.

| decision | choice | source |
|:--|:--|:--|
| batch size | fills the device, not above the noise scale | Batch Size |
| optimizer | AdamW (0.9, 0.95), weight decay 0.1 on matrices | published runs, AdamW |
| precautions | clip at 1, log the clip rate; EMA, window 200 | this deck |
| peak rate | best of three 300-step runs, 3× apart | Two Test Models, Adam |
| schedule | 60-step warmup (2%), cosine to zero | Schedules, published runs |
| verdict | two seeds: gap against spread | Two Test Models |
:::

::: {.slide title="The Device Is Full by 256 Sequences" layout="tight"}
[A worked configuration]{.kicker}

Time of one gradient of the 3.3M-parameter model:

@!practice-a-worked-configuration-2

- 32 to 128: four times the batch, far less than four times the time.
- From 256: the time doubles with the batch, and the sequences per second
  stop rising.
- 256 is the floor for the batch. The ceiling is the *noise scale*: the
  batch at which gradient noise equals the gradient.
:::

::: {.slide title="Three Runs of 300 Steps Pick the Peak Rate" layout="tight"}
[A worked configuration]{.kicker}

@!practice-a-worked-configuration-4

- 0.001: slow. 0.01: stalls near 2.5, though clipped most often. 0.003:
  best, a factor of 3 below the first rate that stalls.
- Noise scale: 2 sequences, then of the order of 100. The floor of 256
  is at or above it: stay at 256.
- Cost: 900 steps, under a third of the final run.

[Cold start, constant rate. Bold assumption: the best rate of 300 constant steps is the right peak for 3,000 decayed ones.]{.small}
:::

::: {.slide title="Four Logged Curves Tell the Failures of a Run Apart" layout="tight"}
[A worked configuration]{.kicker}

Log the gradient norm before clipping, the share of clipped steps, the
learning rate, and the update size relative to the weights.

| in the log | reading | first remedy |
|:--|:--|:--|
| loss leaves in the first steps, norms climb | steps too long for the start | warmup or clipping |
| loss stalls under Adam, first updates large against the weights | start too rough | warmup |
| loss stalls under Adam behind a warmup | rate too large throughout | one grid point down |
| clipping acts on most steps | a brake, not a guard | lower rate |
| loss flat and jittery | noise of the current rate | decay or average |
| loss smooth, still falling | slow | higher rate, more steps |
:::

::: {.slide title="The Schedule Lowers the Loss by 0.05 Nats, Many Times the Seed Spread" layout="tight"}
[A worked configuration]{.kicker}

Configured: 60-step warmup, cosine to zero. Baseline: the same AdamW and
clipping at a constant 0.003.

@!practice-a-worked-configuration-5

- EMA = final weights: nothing left to average after a decay to zero.
- Clipped under 1% of the steps: a guard. Whether the run needed it, the
  table cannot say.
- Known to have done work: the peak rate and the decay.
:::

::: {.slide title="Check: Diagnose a Stalled Run"}
[Check]{.kicker}

AdamW at a rate of 0.01 on the 3.3M-parameter `TinyLM`: the loss stalls
near 2.5 without diverging, and clipping at 1 fires more often than at
lower rates. What do you change?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Not the threshold: Adam moves a coordinate by about $\eta$
whatever the size of its gradient, so tighter clipping does not shorten
the step. Lower the rate by a grid point: 0.003 trains.
:::
:::
:::

::: {.slide title="A Long Run Is Configured Short, Guarded, and Finished with an Average or a Decay"}
[Wrap-up]{.kicker}

- **Guard:** clipping at $\theta = 1$ changed a few dozen early steps and
  let SGD train `TinyLM` at $\eta = 1$; at $\eta = 3$ it fired on
  practically every step and the run failed.
- **Adam:** step bounded at $3.2\,\eta$; an unclipped 100× spike stays
  in $v$ for 2,300 steps (120 at $\beta_2 = 0.95$).
- **Finish:** EMA at a constant rate = cosine decay, about 91% for the
  CNN on Fashion-MNIST.
- **Configure:** batch 256 from the device; peak 0.003 from three 300-step
  runs; warmup and cosine: 0.05 nats below the constant rate, seed spread
  under 0.01.

**Still open:** what a step costs, in seconds and in bytes. **Next
chapter:** attention, the mechanism inside `TinyLM`.
:::
